"""语料检索：跨分片句子级 BM25 索引与增量同步。

问答系统的检索底座，解决三个问题：

1. **跨分片检索**：语料按分片文件存放且持续增多，索引在构建时遍历
   :class:`~storage.sharded.ShardedStore` 的全部分片，把每篇文档切成句子，
   因此检索不遗漏任何分片。
2. **可核实的出处**：每个索引句子都带有
   ``文档 id / 文档名 / 分片文件 / 片内位置 / 句序号 / 字符偏移``，
   答案可直接定位到原文哪一篇、哪一句。
3. **增量更新**：以 ``文档 id -> 内容哈希`` 作为版本签名，新增 / 修改 /
   删除文档后只需重建发生变化的文档句子；签名落盘，重启后仍然有效。
   这样「旧问题」在新文档入库后重新提问就能查到新答案。

算法为纯 Python 实现的 BM25（k1=1.5, b=0.75），分词复用平台自带分词器，
无需任何外部模型或服务。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections import Counter
from typing import Optional

from .lexicon import STOPWORDS
from .segmenter import Segmenter


# BM25 参数
BM25_K1 = 1.5
BM25_B = 0.75

# 只出现在问句中的停用词：它们对定位原文没有帮助，仅保留在分类阶段使用
QUESTION_STOP = {
    "谁", "什么", "怎么", "怎样", "怎么样", "为什么", "哪里", "哪儿", "哪个",
    "哪些", "哪里人", "何时", "多少", "几", "几号", "几年", "请问", "麻烦",
    "时间", "时候", "地点", "地方", "位置", "原因", "定义", "含义", "意思",
    "是不是", "有没有", "吗", "呢", "啊", "请问一下", "哪", "何",
}

_SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+|(?<=[.!?])\s+")


def split_sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """切句并保留每个句子在原文中的字符偏移 ``(句子, start, end)``。"""
    result: list[tuple[str, int, int]] = []
    pos = 0
    for match in _SENT_SPLIT_RE.finditer(text):
        raw = text[pos:match.start()]
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        if raw.strip():
            start = pos + lead
            end = pos + len(raw) - trail
            result.append((text[start:end], start, end))
        pos = match.end()
    raw = text[pos:]
    lead = len(raw) - len(raw.lstrip())
    trail = len(raw) - len(raw.rstrip())
    if raw.strip():
        start = pos + lead
        end = pos + len(raw) - trail
        result.append((text[start:end], start, end))
    return result


def _doc_hash(text: str, name: str) -> str:
    return hashlib.md5(f"{name}\x00{text}".encode("utf-8")).hexdigest()


class CorpusRetriever:
    """对某个 :class:`ShardedStore`（通常是 ``corpus`` 任务）建立句子索引。"""

    def __init__(self, store, index_path: Optional[str] = None,
                 segmenter: Optional[Segmenter] = None):
        self.store = store
        self.index_path = index_path
        self.segmenter = segmenter or Segmenter()
        # passage 字段：
        # pid, doc_id, doc_name, shard_index, shard_file, position,
        # sent_index, start, end, text, tokens
        self._passages: list[dict] = []
        self._by_doc: dict[str, list[int]] = {}
        self._avg_len: float = 0.0
        self._signatures: dict[str, str] = {}
        self._lock = threading.RLock()
        self._loaded = False

    # -- 分词 -------------------------------------------------------------
    def _tokens(self, text: str, question: bool = False) -> list[str]:
        words = self.segmenter.cut(text)
        out: list[str] = []
        for w in words:
            w = w.strip()
            if not w:
                continue
            if w in STOPWORDS:
                continue
            if question and w in QUESTION_STOP:
                continue
            # 纯标点 / 空白丢弃；单字中文保留（专名、动词都可能是单字）
            if not re.search(r"[一-鿿A-Za-z0-9]", w):
                continue
            out.append(w.lower())
        return out

    # -- 索引构建 / 增量同步 ----------------------------------------------
    def _build_doc_passages(self, record: dict, shard_index: int,
                            position: int) -> list[dict]:
        doc_id = record.get("id")
        doc_name = record.get("name", "未命名")
        shard_file = os.path.basename(self.store._shard_path(shard_index))
        passages = []
        for sent_index, (sent, start, end) in enumerate(
                split_sentences_with_offsets(record.get("text", ""))):
            tokens = self._tokens(sent)
            if not tokens:
                continue
            passages.append({
                "doc_id": doc_id,
                "doc_name": doc_name,
                "shard_index": shard_index,
                "shard_file": shard_file,
                "position": position,
                "sent_index": sent_index,
                "start": start,
                "end": end,
                "text": sent,
                "tokens": tokens,
                "token_text": "".join(tokens),
                "token_set": set(tokens),
            })
        return passages

    def _reindex(self, live_records: list[tuple[dict, int, int]]) -> dict:
        """根据当前全量存活记录重建内存索引（跨全部分片）。"""
        passages: list[dict] = []
        by_doc: dict[str, list[int]] = {}
        signatures: dict[str, str] = {}
        total_len = 0

        for record, shard_index, position in live_records:
            if record.get("_deleted"):
                continue
            doc_id = record.get("id")
            text = record.get("text", "")
            name = record.get("name", "未命名")
            signatures[doc_id] = _doc_hash(text, name)
            base = len(passages)
            doc_passages = self._build_doc_passages(record, shard_index, position)
            for i, passage in enumerate(doc_passages):
                passage["pid"] = base + i
                passages.append(passage)
                total_len += len(passage["tokens"])
            by_doc[doc_id] = list(range(base, base + len(doc_passages)))

        self._passages = passages
        self._by_doc = by_doc
        self._avg_len = (total_len / len(passages)) if passages else 0.0
        self._signatures = signatures
        return {
            "documents": len(signatures),
            "passages": len(passages),
        }

    def sync(self, force: bool = False) -> dict:
        """与分片存储同步索引。

        增量策略：计算当前全部存活文档（跨分片）的内容哈希签名，与已索引
        签名比较；完全一致则跳过，否则对发生增 / 删 / 改的语料重建索引。

        :param force: 强制全量重建
        """
        with self._lock:
            self._ensure_loaded()
            live = [(r, si, pos)
                    for r, si, pos in self.store.iter_with_location()
                    if not r.get("_deleted")]
            current = {
                r.get("id"): _doc_hash(r.get("text", ""), r.get("name", "未命名"))
                for r, _, _ in live
            }
            if not force and current == self._signatures and self._passages is not None:
                return {"changed": False, "documents": len(current),
                        "passages": len(self._passages)}

            stats = self._reindex(live)
            stats["changed"] = True
            self._save_snapshot()
            return stats

    # -- 持久化 -----------------------------------------------------------
    def _snapshot_payload(self) -> dict:
        return {
            "version": 1,
            "built_at": time.time(),
            "signatures": self._signatures,
            "passages": [{k: p[k] for k in (
                "pid", "doc_id", "doc_name", "shard_index", "shard_file",
                "position", "sent_index", "start", "end", "text", "tokens")
            } for p in self._passages],
        }

    @staticmethod
    def _hydrate(passage: dict) -> dict:
        """补齐不写入快照的派生字段。"""
        passage["token_text"] = "".join(passage["tokens"])
        passage["token_set"] = set(passage["tokens"])
        return passage

    def _save_snapshot(self) -> None:
        if not self.index_path:
            return
        tmp = f"{self.index_path}.{os.getpid()}.tmp"
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._snapshot_payload(), fh, ensure_ascii=False)
        os.replace(tmp, self.index_path)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.index_path and os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                passages = payload.get("passages", [])
                self._passages = [self._hydrate(p) for p in passages]
                self._by_doc = {}
                total_len = 0
                for p in self._passages:
                    self._by_doc.setdefault(p["doc_id"], []).append(p["pid"])
                    total_len += len(p["tokens"])
                self._avg_len = (total_len / len(self._passages)
                                 if self._passages else 0.0)
                self._signatures = payload.get("signatures", {})
            except (json.JSONDecodeError, OSError, KeyError):
                pass

    # -- 检索 -------------------------------------------------------------
    @staticmethod
    def _term_hit(term: str, token_text: str, token_set: set) -> bool:
        """词项命中：精确命中，或在该句词序列拼接串中形成子串包含。

        子串匹配用于弥合分词粒度差异（问题切成「成立」、文档切成「成立于」），
        仅对长度 >= 2 的中文/字母词项生效，避免单字噪声。
        """
        if term in token_set:
            return True
        if len(term) >= 2 and re.search(r"[一-鿿a-zA-Z]", term):
            return term in token_text
        return False

    def search(self, query: str, top_k: int = 8,
               min_score: float = 0.0) -> list[dict]:
        """跨全部分片检索最相关的句子段落，按 BM25 分数降序返回。

        命中判定在精确词项之外增加子串包含匹配（分词粒度容错）；
        ``min_score`` 为绝对门槛，调用方还可用相对门槛（如 top1 的比例）
        做二次筛选。
        """
        with self._lock:
            self.sync()
            q_terms = self._tokens(query, question=True)
            if not q_terms or not self._passages:
                return []
            q_counts = Counter(q_terms)
            n = len(self._passages)
            idf_cache: dict[str, float] = {}
            for term in q_counts:
                # DF 同样按「精确或子串」统计，保证 IDF 与命中口径一致
                df = sum(1 for p in self._passages
                         if self._term_hit(term, p["token_text"], p["token_set"]))
                idf_cache[term] = math.log((n - df + 0.5) / (df + 0.5) + 1.0) if df else 0.0

            scored: list[tuple[float, dict, list[str]]] = []
            for passage in self._passages:
                token_text = passage["token_text"]
                token_set = passage["token_set"]
                dl = len(passage["tokens"]) or 1
                norm = BM25_K1 * (1 - BM25_B + BM25_B * dl / (self._avg_len or 1))
                score = 0.0
                hits: list[str] = []
                for term in q_counts:
                    if not self._term_hit(term, token_text, token_set):
                        continue
                    tf = passage["tokens"].count(term)
                    if term not in token_set:
                        tf = 1  # 子串命中给一次词频
                    score += idf_cache.get(term, 0.0) * (
                        tf * (BM25_K1 + 1)) / (tf + norm)
                    hits.append(term)
                if score > 0:
                    scored.append((score, passage, sorted(set(hits))))

            scored.sort(key=lambda x: x[0], reverse=True)
            results = []
            for score, passage, hits in scored[:top_k]:
                if score < min_score:
                    continue
                results.append({
                    "score": round(score, 4),
                    "doc_id": passage["doc_id"],
                    "doc_name": passage["doc_name"],
                    "shard_index": passage["shard_index"],
                    "shard_file": passage["shard_file"],
                    "position": passage["position"],
                    "sent_index": passage["sent_index"],
                    "start": passage["start"],
                    "end": passage["end"],
                    "text": passage["text"],
                    "matched_terms": hits,
                })
            return results

    # -- 状态 -------------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            self.sync()
            return {
                "documents": len(self._signatures),
                "passages": len(self._passages),
                "indexed_at": (os.path.getmtime(self.index_path)
                               if self.index_path and os.path.exists(self.index_path)
                               else None),
            }
