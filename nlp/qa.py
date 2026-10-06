"""基于语料库的事实性问答（检索 + 抽取，轻量级 RAG）。

不依赖外部大模型，整个链路为纯 Python：

1. **索引**：把分片语料库（``ShardedStore`` 中的 ``corpus`` 任务）逐篇切句、
   分词，建立 BM25 倒排索引。索引按文档 id/文本指纹做**增量更新**——
   新增文档后旧问题立刻能查到新答案；索引快照落盘，重启后快速载入。
2. **检索**：对问题分词后**跨全部分片**检索最相关的句子（段落），
   按 BM25 分数合并排序，而不是命中一个就返回。
3. **答案抽取**：按问题类型（定义 / 人物 / 机构 / 地点 / 时间事件 / 其他）
   在候选句中做规则抽取，并复用 :mod:`nlp.ner` 识别人名、地名、机构、时间。
4. **拒答（不编造）**：最高候选分数过低、或问题关键词覆盖率不足时，
   明确返回 ``found=False``，不产生答案。
5. **溯源与多答案辨析**：每个答案都附「文档 id / 文档名 / 句子序号 / 原文」，
   同一问题在不同文档中有不同说法时分组列出并标记冲突。

典型用法::

    engine = QAEngine(registry)
    result = engine.answer("什么是自然语言处理？")
    if result["found"]:
        for a in result["answers"]:
            print(a["answer"], a["sources"])
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections import Counter
from typing import Optional

from .lexicon import STOPWORDS
from .ner import NERExtractor
from .pos import POSTagger
from .segmenter import Segmenter
from storage import FileLock, _atomic_write_json, _read_json, lock_path_for


# ---------------------------------------------------------------------------
# 句子切分（保留字符偏移与句序，供精确定位）
# ---------------------------------------------------------------------------

# 句末标点或换行；英文标点后若有空白也算句末
_SENT_END_RE = re.compile(r"[。！？!?；;\n]+|(?<=[.!?])\s+")


def split_sentences_with_offsets(text: str) -> list[dict]:
    """把文本切成句子，返回 ``[{index, text, start, end}]``。

    ``index`` 为句序（从 0 起），``start/end`` 为相对文档的字符偏移，
    与切分标点的处理方式无关，可直接用于高亮与定位。
    """
    sentences: list[dict] = []
    pos = 0
    index = 0
    for m in _SENT_END_RE.finditer(text):
        raw = text[pos:m.start()]
        stripped = raw.strip()
        if stripped:
            lead = len(raw) - len(raw.lstrip())
            sentences.append({
                "index": index,
                "text": stripped,
                "start": pos + lead,
                "end": pos + lead + len(stripped),
            })
            index += 1
        pos = m.end()
    tail = text[pos:]
    stripped = tail.strip()
    if stripped:
        lead = len(tail) - len(tail.lstrip())
        sentences.append({
            "index": index,
            "text": stripped,
            "start": pos + lead,
            "end": pos + lead + len(stripped),
        })
    return sentences


# ---------------------------------------------------------------------------
# 分词 / 索引项
# ---------------------------------------------------------------------------

_CJK_CHAR = re.compile(r"[一-鿿]")
_CJK_RUN = re.compile(r"[一-鿿]+")
_ASCII_WORD = re.compile(r"[a-zA-Z0-9]+(?:[.\-@][a-zA-Z0-9]+)*")


def analyze_tokens(text: str, segmenter: Segmenter) -> list[str]:
    """把文本切成用于检索的词项。

    中文用平台分词器，再对**连续汉字串整体**补充二元字串（bigram），
    保证未登录的专有名词（人名、术语）、以及分词边界两侧也能被字面匹配；
    英文/数字整体保留并小写。
    """
    terms: list[str] = []
    for w in segmenter.cut(text):
        w = w.strip()
        if not w:
            continue
        if _CJK_CHAR.search(w):
            if w not in STOPWORDS:
                terms.append(w)
        else:
            for tok in _ASCII_WORD.findall(w):
                terms.append(tok.lower())
    # 字级 bigram：按原文连续汉字串生成，不受分词边界影响
    for run in _CJK_RUN.findall(text):
        for i in range(len(run) - 1):
            terms.append(run[i:i + 2])
    return terms


def query_terms(text: str, segmenter: Segmenter) -> list[str]:
    """问题的关键词项（去停用词），同时保留词与 bigram。"""
    return [t for t in analyze_tokens(text, segmenter)
            if t not in STOPWORDS and len(t) >= 1]


# ---------------------------------------------------------------------------
# 问题分类
# ---------------------------------------------------------------------------

class QType:
    DEFINITION = "definition"   # 「X 是什么 / 什么是 X / X 的定义」
    PERSON = "person"           # 「谁」
    ORGANIZATION = "organization"
    LOCATION = "location"       # 「哪里 / 在哪儿」
    TIME = "time"               # 「什么时候 / 何时 / 哪年」
    FACT = "fact"               # 其它事实性问题


_DEF_PATTERNS = [
    re.compile(r"什么是「(?P<c>.+?)」"),
    re.compile(r"什么是【(?P<c>.+?)】"),
    re.compile(r"什么是[“\"']?(?P<c>.+?)[”\"']?(?:呢|啊)?(?:？|\?|。|$)"),
    re.compile(r"(?P<c>.+?)是什么(?P<cat>东西|概念|意思|含义|设备|技术|系统|"
               r"方法|算法|模型|语言|工具|产品|组织|机构|公司|地方|城市|"
               r"国家|现象|物质|元素|动物|植物|疾病|药物|指标|协议|服务)?"
               r"(?:呢|啊)?(?:？|\?|。|$)"),
    re.compile(r"(?P<c>.+?)的(?:定义|含义|概念|意思)是什么(?:呢|啊)?(?:？|\?|。|$)"),
    re.compile(r"(?P<c>.+?)指(?:的是|什么)(?:呢|啊)?(?:？|\?|。|$)"),
    re.compile(r"如何(?:理解|定义)(?P<c>.+?)(?:呢|啊)?(?:？|\?|。|$)"),
]

# 定义句中引导被定义项的模式（X 是 / X 指的是 / 所谓 X）
_DEF_COPULA = ("是指", "指的是", "定义为", "所谓", "是", "即", "指", "属于")

_LEAD_CLAUSE = re.compile(
    r"^(?:其实|简单来说|简单地说|一般来说|通常来说|通常|所谓)?[,，]?\s*")
_TAIL_PUNCT = re.compile(r"[。！？!?；;，,、\s]+$")

# 时间表达式（日期 / 年份 / 相对时间）
_TIME_PATTERNS = [
    re.compile(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?"),
    re.compile(r"\d{4}\s*年\s*\d{1,2}\s*月"),
    re.compile(r"\d{4}\s*年"),
    re.compile(r"\d{1,2}\s*月\s*\d{1,2}\s*日"),
    re.compile(r"[一二三四五六七八九十零两\d]+\s*年前"),
    re.compile(r"(?:去年|今年|明年|昨天|今天|明天|前天|后天)"),
    re.compile(r"\d{1,2}\s*[:：]\s*\d{2}(?:\s*[:：]\s*\d{2})?"),
]

# 问时间的疑问词
_QTIME_RE = re.compile(r"(什么时候|何时|哪一?年|哪一?天|几点|什么时间)")
# 问地点
_QLOC_RE = re.compile(r"(哪里|哪儿|在哪|何处|什么地方|哪个地方)")
# 问方面/领域（形式像地点问，实际是问范围，按事实问处理）
_QASPECT_RE = re.compile(r"(?:哪个|哪种|哪些|什么)(领域|方面|行业|场景)")
# 问人物
_QWHO_RE = re.compile(r"(?:是谁|谁是|谁在|谁曾|谁能|谁可|谁将|谁会|谁没|由谁|何人|谁)")
# 问机构
_QORG_RE = re.compile(r"(哪家公司|哪个机构|哪家机构|什么公司|什么机构|哪个组织)")
# 问金额
_QMONEY_RE = re.compile(r"(多少钱|什么价格|价格是多少|售价(?:是)?多少|花费多少|费用多少)")
_MONEY_SIGNAL_RE = re.compile(r"(售价|价格|价钱|花费|费用|标价|定价|花了?|值)")


def classify_question(question: str) -> str:
    q = question.strip()
    for pat in _DEF_PATTERNS:
        if pat.search(q):
            return QType.DEFINITION
    if _QASPECT_RE.search(q):
        return QType.FACT
    if _QORG_RE.search(q):
        return QType.ORGANIZATION
    if _QWHO_RE.search(q):
        return QType.PERSON
    if _QLOC_RE.search(q):
        return QType.LOCATION
    if _QTIME_RE.search(q):
        return QType.TIME
    return QType.FACT


def _definition_target(question: str, segmenter: Segmenter) -> list[str]:
    """从定义类问题中抽取被问概念（可能多种切分）。"""
    q = question.strip().rstrip("？?。！!")
    for pat in _DEF_PATTERNS:
        m = pat.search(q)
        if m:
            concept = m.group("c").strip("“”\"'「」【】 ")
            if concept and concept not in ("什么", "谁"):
                return [concept]
    return []


# ---------------------------------------------------------------------------
# BM25 索引（增量、可落盘）
# ---------------------------------------------------------------------------

BM25_K1 = 1.5
BM25_B = 0.75


class QAIndex:
    """对语料库全部文档的句子建立 BM25 倒排索引。"""

    def __init__(self, segmenter: Optional[Segmenter] = None):
        self.segmenter = segmenter or Segmenter()
        # 句子级数据
        self.passages: list[dict] = []          # 见 _index_doc
        self.doc_names: dict[str, str] = {}
        # 倒排表 term -> {passage_idx: tf}
        self.postings: dict[str, dict[int, int]] = {}
        self.df: Counter = Counter()
        self.tf: list[Counter] = []
        self.doc_lens: list[int] = []
        # 文档指纹：id -> (name, text_hash, passage 起始下标, 句数)
        self.doc_fingerprints: dict[str, dict] = {}
        self.avgdl = 0.0

    # -- 索引构建 ---------------------------------------------------------
    def _hash_text(self, text: str) -> str:
        import hashlib
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def index_doc(self, doc_id: str, name: str, text: str) -> bool:
        """（重新）索引单篇文档；若指纹未变则跳过。返回是否发生了更新。"""
        fp = self._hash_text(text)
        old = self.doc_fingerprints.get(doc_id)
        if old and old["hash"] == fp and old.get("name") == name:
            return False
        if old:
            self._remove_doc(doc_id)
        sentences = split_sentences_with_offsets(text)
        start = len(self.passages)
        for sent in sentences:
            terms = analyze_tokens(sent["text"], self.segmenter)
            tf = Counter(terms)
            self.tf.append(tf)
            self.doc_lens.append(len(terms))
            idx = len(self.passages)
            self.passages.append({
                "doc_id": doc_id,
                "doc_name": name,
                "sent_index": sent["index"],
                "text": sent["text"],
                "start": sent["start"],
                "end": sent["end"],
            })
            for term, freq in tf.items():
                self.postings.setdefault(term, {})[idx] = freq
            for term in tf:
                self.df[term] += 1
        self.doc_names[doc_id] = name
        self.doc_fingerprints[doc_id] = {
            "hash": fp, "name": name,
            "start": start, "count": len(sentences),
        }
        self._recompute_avgdl()
        return True

    def _remove_doc(self, doc_id: str) -> None:
        info = self.doc_fingerprints.pop(doc_id, None)
        self.doc_names.pop(doc_id, None)
        if not info:
            return
        lo, hi = info["start"], info["start"] + info["count"]
        drop = set(range(lo, hi))
        # 重建受影响的结构（语料规模为演示级，直接全量压缩最稳妥）
        self.passages = [p for i, p in enumerate(self.passages) if i not in drop]
        self.tf = [t for i, t in enumerate(self.tf) if i not in drop]
        self.doc_lens = [l for i, l in enumerate(self.doc_lens) if i not in drop]
        self.postings = {}
        self.df = Counter()
        for idx, tf in enumerate(self.tf):
            for term, freq in tf.items():
                self.postings.setdefault(term, {})[idx] = freq
            for term in tf:
                self.df[term] += 1
        # 重排各文档的起始偏移
        cursor = 0
        for d_id, fp in self.doc_fingerprints.items():
            fp["start"] = cursor
            cursor += fp["count"]
        self._recompute_avgdl()

    def _recompute_avgdl(self) -> None:
        total = sum(self.doc_lens)
        self.avgdl = total / len(self.doc_lens) if self.doc_lens else 0.0

    def sync_store(self, store) -> dict:
        """与分片语料库做增量同步：新增/变更的重新索引，删除的剔除。"""
        live_ids: set[str] = set()
        added, updated, unchanged = 0, 0, 0
        for record in store.all():
            if record.get("_deleted"):
                continue
            doc_id = record.get("id")
            text = record.get("text", "") or ""
            name = record.get("name", "未命名")
            if not doc_id or not text.strip():
                continue
            live_ids.add(doc_id)
            fp = self._hash_text(text)
            old = self.doc_fingerprints.get(doc_id)
            if self.index_doc(doc_id, name, text):
                if old is None:
                    added += 1
                else:
                    updated += 1
            else:
                unchanged += 1
        removed = 0
        for doc_id in list(self.doc_fingerprints):
            if doc_id not in live_ids:
                self._remove_doc(doc_id)
                removed += 1
        return {"documents": len(live_ids), "passages": len(self.passages),
                "added": added, "updated": updated, "removed": removed,
                "unchanged": unchanged}

    # -- 检索 -------------------------------------------------------------
    def search(self, terms: list[str], top_k: int = 8) -> list[dict]:
        """跨全部文档检索与查询词项最相关的句子。

        返回 ``[{passage, score, coverage, matched_terms}]``，按分数降序。
        ``coverage`` 为查询关键词在该句中出现的比例（去重词项口径）。
        """
        if not self.passages or not terms:
            return []
        q_terms = list(dict.fromkeys(terms))
        n = len(self.passages)
        scores: dict[int, float] = {}
        matched: dict[int, set[str]] = {}
        for term in q_terms:
            posts = self.postings.get(term)
            if not posts:
                continue
            idf = math.log(1 + (n - len(posts) + 0.5) / (len(posts) + 0.5))
            for idx, freq in posts.items():
                denom = freq + BM25_K1 * (
                    1 - BM25_B + BM25_B * self.doc_lens[idx] /
                    (self.avgdl or 1.0))
                score = idf * (freq * (BM25_K1 + 1)) / denom
                scores[idx] = scores.get(idx, 0.0) + score
                matched.setdefault(idx, set()).add(term)

        results = []
        for idx, score in scores.items():
            cov = len(matched[idx]) / len(q_terms) if q_terms else 0.0
            results.append({
                "passage": self.passages[idx],
                "score": score,
                "norm_score": self._normalize(score),
                "coverage": cov,
                "matched_terms": sorted(matched[idx]),
            })
        results.sort(key=lambda r: (r["norm_score"], r["coverage"]), reverse=True)
        return results[:top_k]

    @staticmethod
    def _normalize(score: float) -> float:
        """把 BM25 原始分压到 0~1，用于阈值判定与展示。"""
        if score <= 0:
            return 0.0
        return round(score / (score + 6.0), 4)

    # -- 上下文 -----------------------------------------------------------
    def neighbors(self, doc_id: str, sent_index: int, span: int = 1) -> list[dict]:
        """取某句前后相邻的句子（供展示语境，不参与答案判定）。"""
        out = []
        for p in self.passages:
            if (p["doc_id"] == doc_id
                    and abs(p["sent_index"] - sent_index) <= span
                    and p["sent_index"] != sent_index):
                out.append({"sent_index": p["sent_index"], "text": p["text"]})
        out.sort(key=lambda x: x["sent_index"])
        return out

    # -- 落盘 -------------------------------------------------------------
    def to_snapshot(self) -> dict:
        return {
            "version": 1,
            "passages": self.passages,
            "doc_fingerprints": self.doc_fingerprints,
            "doc_names": self.doc_names,
            "tf": [dict(c) for c in self.tf],
            "doc_lens": self.doc_lens,
            "avgdl": self.avgdl,
        }

    @classmethod
    def from_snapshot(cls, data: dict,
                      segmenter: Optional[Segmenter] = None) -> "QAIndex":
        idx = cls(segmenter)
        idx.passages = data.get("passages", [])
        idx.doc_fingerprints = data.get("doc_fingerprints", {})
        idx.doc_names = data.get("doc_names", {})
        idx.tf = [Counter(c) for c in data.get("tf", [])]
        idx.doc_lens = data.get("doc_lens", [])
        idx.avgdl = data.get("avgdl", 0.0)
        idx.postings = {}
        idx.df = Counter()
        for i, tf in enumerate(idx.tf):
            for term, freq in tf.items():
                idx.postings.setdefault(term, {})[i] = freq
            for term in tf:
                idx.df[term] += 1
        return idx


# ---------------------------------------------------------------------------
# 答案抽取
# ---------------------------------------------------------------------------

class AnswerExtractor:
    """按问题类型从候选句中抽取答案片段。"""

    def __init__(self, segmenter: Optional[Segmenter] = None,
                 ner: Optional[NERExtractor] = None,
                 tagger: Optional[POSTagger] = None):
        self.segmenter = segmenter or Segmenter()
        self.ner = ner or NERExtractor(self.segmenter)
        self.tagger = tagger or POSTagger()

    def extract(self, qtype: str, question: str, sentence: str) -> dict:
        """返回 ``{answer, start, end, entities}``；抽不到时 answer 为空串。"""
        if qtype == QType.DEFINITION:
            return self._extract_definition(question, sentence)
        if qtype == QType.TIME:
            return self._extract_time_event(question, sentence)
        etype = {
            QType.PERSON: "PERSON",
            QType.ORGANIZATION: "ORGANIZATION",
            QType.LOCATION: "LOCATION",
        }.get(qtype)
        if etype:
            hit = self._extract_entity(sentence, etype)
            # 「谁在什么时候做了什么」式事件问：以整句为答案，
            # 但保留目标实体的偏移（focus_*）用于前端高亮
            if hit["answer"] and self._is_event_question(question):
                hit["focus_start"] = hit["start"]
                hit["focus_end"] = hit["end"]
                hit["start"], hit["end"] = 0, len(sentence)
            return hit
        return self._extract_fact(question, sentence)

    @staticmethod
    def _is_event_question(question: str) -> bool:
        return bool(re.search(r"(做了什么|干什么|做了啥|做什么|有什么成就|"
                              r"发生了什么|干了什么|提出了什么|发现了什么)",
                              question))

    # -- 定义 -------------------------------------------------------------
    def _extract_definition(self, question: str, sentence: str) -> dict:
        targets = _definition_target(question, self.segmenter)
        entities = self.ner.recognize(sentence)
        # 句中必须提到被定义概念，否则不算答案
        if targets and not any(t in sentence for t in targets):
            return {"answer": "", "start": -1, "end": -1, "entities": entities}

        for marker in ("是指", "指的是", "定义为"):
            pos = sentence.find(marker)
            if pos >= 0:
                return self._pack(sentence, pos + len(marker), len(sentence),
                                  entities)
        for marker in ("所谓",):
            pos = sentence.find(marker)
            if pos >= 0:
                rest_start = pos + len(marker)
                cut = sentence.find("是", rest_start)
                if cut >= 0:
                    return self._pack(sentence, cut + 1, len(sentence), entities)
        # 「X 是 …」「X 即 …」「X 指 …」：被定义项须位于句首，系动词紧随
        # 其后（至多隔一个停顿标点），避免「X 的售价是 1999 元」被误判为定义
        for t in sorted(targets, key=len, reverse=True):
            if not sentence.startswith(t):
                continue
            after = len(t)
            offset = 1 if sentence[after:after + 1] in ("，", ",", " ") else 0
            ch = sentence[after + offset:after + offset + 1]
            if ch in ("是", "即", "指"):
                return self._pack(sentence, after + offset + 1,
                                  len(sentence), entities)
        # 「X 用于 / X 通过 / X 具有 …」：句首即被定义项，且其后紧跟
        # 描述性动词时，后续描述即为释义（不能是「X 的售价是…」这类句子）
        desc_markers = ("用于", "用来", "通过", "具有", "属于", "由",
                        "利用", "描述", "表示", "代表")
        if targets:
            for t in sorted(targets, key=len, reverse=True):
                if not sentence.startswith(t):
                    continue
                rest = sentence[len(t):]
                if any(rest.startswith(m) for m in desc_markers):
                    return self._pack(sentence, len(t), len(sentence), entities)
        return {"answer": "", "start": -1, "end": -1, "entities": entities}

    def _find_copula_after_target(self, sentence: str, targets: list[str]) -> int:
        search_from = 0
        if targets:
            earliest = -1
            for t in targets:
                p = sentence.find(t)
                if p >= 0 and (earliest < 0 or p < earliest):
                    earliest = p
            if earliest >= 0:
                search_from = earliest + len(targets[0])
        for ch in ("即", "指"):
            p = sentence.find(ch, search_from)
            if p >= 0:
                return p
        p = sentence.find("是", search_from)
        return p

    # -- 实体类（谁 / 哪家机构 / 哪里） -----------------------------------
    def _extract_entity(self, sentence: str, etype: str) -> dict:
        entities = self.ner.recognize(sentence)
        wanted = [e for e in entities if e["type"] == etype]
        if wanted:
            # 取句中第一个目标类型实体作为核心答案
            e = wanted[0]
            return {"answer": self._clean_phrase(e["text"]),
                    "start": e["start"], "end": e["end"], "entities": entities}
        # NER 未命中：用词性回退（人名 nr / 机构 nt / 地名 ns），
        # 并把相邻的同类实体词合并（「李/明」两个 nr → 「李明」）
        tag_map = {"PERSON": {"nr"}, "ORGANIZATION": {"nt"},
                   "LOCATION": {"ns"}}
        words = self.tagger.tag(sentence)
        tags = tag_map.get(etype, set())
        pos = 0
        i = 0
        tagged = []
        for w, t in words:
            start = sentence.find(w, pos)
            if start < 0:
                start = pos
            tagged.append((w, t, start, start + len(w)))
            pos = start + len(w)
        while i < len(tagged):
            w, t, start, end = tagged[i]
            if t in tags:
                j = i + 1
                while j < len(tagged) and tagged[j][1] in tags \
                        and tagged[j][2] == tagged[j - 1][3]:
                    end = tagged[j][3]
                    j += 1
                phrase = sentence[start:end]
                if len(phrase) >= 2:
                    return {"answer": self._clean_phrase(phrase),
                            "start": start, "end": end,
                            "entities": entities}
            i += 1
        return {"answer": "", "start": -1, "end": -1, "entities": entities}

    # -- 时间 / 事件 ------------------------------------------------------
    def _extract_time_event(self, question: str, sentence: str) -> dict:
        entities = self.ner.recognize(sentence)
        times = self._find_times(sentence)
        asks_time = bool(_QTIME_RE.search(question))
        # 问「什么时候做了X」：若句中有时间，答案给时间；
        # 若问题自带时间（「X 年发生了什么」），给事件本身。
        q_has_time = bool(self._find_times(question))
        if asks_time and times:
            t = times[0]
            return {"answer": self._clean_phrase(t["text"]),
                    "start": t["start"], "end": t["end"], "entities": entities}
        # 事件：优先返回整句作为可核实的事实，去掉开头的时间状语
        start = 0
        if times and q_has_time:
            first = times[0]
            if first["start"] <= 4:
                start = first["end"]
        answer = sentence[start:].lstrip("，,。. ")
        return {"answer": self._clean_phrase(answer) or sentence,
                "start": start, "end": len(sentence), "entities": entities}

    @staticmethod
    def _find_times(sentence: str) -> list[dict]:
        found = []
        for pat in _TIME_PATTERNS:
            for m in pat.finditer(sentence):
                found.append({"text": m.group(), "start": m.start(),
                              "end": m.end()})
        found.sort(key=lambda x: x["start"])
        return found

    # -- 一般事实 ---------------------------------------------------------
    def _extract_fact(self, question: str, sentence: str) -> dict:
        entities = self.ner.recognize(sentence)
        # 若句中有时间/数字/机构等实体，优先给最突出的那个
        priority = ("PERCENT", "MONEY", "DATE", "TIME", "NUMBER",
                    "PERSON", "ORGANIZATION", "LOCATION")
        present = {e["type"]: e for e in entities}
        for etype in priority:
            if etype in present:
                e = present[etype]
                return {"answer": self._clean_phrase(e["text"]),
                        "start": e["start"], "end": e["end"],
                        "entities": entities}
        # 回退：「X 是 …」的表语部分
        pos = self._find_copula_after_target(sentence, [])
        if pos >= 0:
            return self._pack(sentence, pos + 1, len(sentence), entities)
        # 一般事实问（怎么做 / 利用什么 / 为什么 …）：整句即事实，
        # 交还给上层以原句作答并附出处，避免有证据却误判为「无答案」
        return {"answer": sentence, "start": 0, "end": len(sentence),
                "entities": entities}

    # -- 工具 -------------------------------------------------------------
    def _pack(self, sentence: str, start: int, end: int,
              entities: list) -> dict:
        answer = sentence[start:end]
        answer = _LEAD_CLAUSE.sub("", answer)
        answer = _TAIL_PUNCT.sub("", answer).strip()
        return {"answer": answer, "start": start, "end": end,
                "entities": entities}

    @staticmethod
    def _clean_phrase(text: str) -> str:
        text = _TAIL_PUNCT.sub("", text).strip()
        return text.strip("“”\"' ")


# ---------------------------------------------------------------------------
# 问答引擎：索引 + 检索 + 抽取 + 多答案辨析
# ---------------------------------------------------------------------------

# 拒答阈值：归一化 BM25 分数与查询词覆盖率
MIN_SCORE = 0.08
MIN_COVERAGE = 0.34
# 次要候选与最高可答句分差超过此值则不采信（跨文档的不同说法不受此裁剪）
SECONDARY_GAP = 0.25


class QAEngine:
    """对外问答门面：维护索引（增量同步、快照缓存）并回答问题。"""

    def __init__(self, registry=None, data_root: Optional[str] = None,
                 top_k: int = 8):
        self.registry = registry
        self.data_root = data_root
        self.top_k = top_k
        self._lock = threading.RLock()
        self._index: Optional[QAIndex] = None
        self._segmenter = Segmenter()
        self._extractor = AnswerExtractor(self._segmenter)

    # -- 索引管理 ---------------------------------------------------------
    def _snapshot_path(self) -> str:
        root = self.data_root or (
            self.registry.root if self.registry is not None else ".")
        qa_dir = os.path.join(root, "qa")
        os.makedirs(qa_dir, exist_ok=True)
        return os.path.join(qa_dir, "index.json")

    def _load_snapshot(self) -> Optional[QAIndex]:
        path = self._snapshot_path()
        if not os.path.exists(path):
            return None
        try:
            data = _read_json(path, None)
            if not data:
                return None
            return QAIndex.from_snapshot(data, self._segmenter)
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            return None

    def _save_snapshot(self, index: QAIndex) -> None:
        path = self._snapshot_path()
        with FileLock(lock_path_for(path)):
            _atomic_write_json(path, index.to_snapshot())

    def reindex(self) -> dict:
        """强制与语料库同步（通常无需手动调用，answer 前会自动同步）。"""
        with self._lock:
            if self._index is None:
                self._index = self._load_snapshot() or QAIndex(self._segmenter)
            stats = {"documents": 0, "passages": 0, "added": 0,
                     "updated": 0, "removed": 0, "unchanged": 0}
            if self.registry is not None:
                store = self.registry.task("corpus")
                stats = self._index.sync_store(store)
                if stats["added"] or stats["updated"] or stats["removed"]:
                    self._save_snapshot(self._index)
            return stats

    def index_stats(self) -> dict:
        with self._lock:
            if self._index is None:
                self._index = self._load_snapshot() or QAIndex(self._segmenter)
            return {
                "documents": len(self._index.doc_fingerprints),
                "passages": len(self._index.passages),
                "terms": len(self._index.postings),
            }

    # -- 问答 -------------------------------------------------------------
    def answer(self, question: str, top_k: Optional[int] = None) -> dict:
        question = (question or "").strip()
        if not question:
            return self._not_found(question, QType.FACT, "问题为空")
        with self._lock:
            stats = self.reindex()
            index = self._index
            if len(index.passages) == 0:
                return self._not_found(question, classify_question(question),
                                       "语料库为空，请先添加文档", stats)

            qtype = classify_question(question)
            terms = query_terms(question, self._segmenter)
            candidates = index.search(terms, top_k or self.top_k)
            decision = self._decide(qtype, question, candidates, index)
            decision["index"] = stats
            return decision

    # -- 判定与多答案辨析 -------------------------------------------------
    def _decide(self, qtype: str, question: str, candidates: list[dict],
                index: QAIndex) -> dict:
        if not candidates:
            return self._not_found(question, qtype,
                                   "语料中没有与问题关键词匹配的句子")
        best = candidates[0]
        if best["norm_score"] < MIN_SCORE:
            return self._not_found(
                question, qtype,
                f"最相关段落的相关度仅 {best['norm_score']}，低于可信度阈值",
                candidates=candidates)

        # 先在所有候选上抽答案
        enriched = [(c, self._extractor.extract(
            qtype, question, c["passage"]["text"])) for c in candidates]

        # 不同问题类型对「能否作答」的判据不同
        typed_types = {QType.DEFINITION, QType.PERSON, QType.ORGANIZATION,
                       QType.LOCATION, QType.TIME}
        if qtype in typed_types:
            answerable = [(c, e) for c, e in enriched if e.get("answer")]
            # 定义问：售价/数量/时间等带度量信号的句子不是定义
            if qtype == QType.DEFINITION:
                answerable = [(c, e) for c, e in answerable
                              if self._is_definitional(
                                  c["passage"]["text"], e, question)]
            # 问题自带的限定成分（年份/日期/数字/地名/机构）必须在答案句中出现，
            # 例如问「谁在2015年…」时，只含「2019年」的句子不予采纳
            constraints = self._question_constraints(question)
            if constraints:
                answerable = [(c, e) for c, e in answerable
                              if self._satisfies_constraints(
                                  c["passage"]["text"], constraints)]
            # 时间/事件问里，问题中的动作词（成立、发布、接任…）也要共现，
            # 避免「公司哪年成立」答成另一事件的年份
            if qtype == QType.TIME:
                actions = self._question_actions(question)
                if actions:
                    answerable = [(c, e) for c, e in answerable
                                  if self._satisfies_actions(
                                      c["passage"]["text"], actions)]
            if not answerable:
                return self._not_found(
                    question, qtype,
                    "虽然有相关段落，但无法从中抽取到与问题类型对应的答案",
                    candidates=candidates)
        else:
            money_question = bool(_QMONEY_RE.search(question))
            # 一般事实问：要求关键词覆盖率达标，防止词面碰撞误答
            if best["coverage"] < MIN_COVERAGE and not money_question:
                return self._not_found(
                    question, qtype,
                    f"问题关键词在最相关段落中覆盖率仅 {best['coverage']:.0%}，"
                    "无法确认语料包含答案",
                    candidates=candidates)
            answerable = [(c, e) for c, e in enriched if e.get("answer")]
            # 「多少钱/价格」类问题：答案句必须真正含金额
            if money_question:
                answerable = [
                    (c, e) for c, e in answerable
                    if any(en["type"] in ("MONEY", "NUMBER")
                           for en in e.get("entities", []))
                    and self._has_money_signal(c["passage"]["text"])]
                if not answerable:
                    return self._not_found(
                        question, qtype, "相关段落中没有可核实的价格/金额信息",
                        candidates=candidates)

        # 只保留分数与最高可答句接近的候选（跨文档的不同说法不会被轻易丢掉）
        top_score = max(c["norm_score"] for c, _ in answerable)
        cutoff = max(MIN_SCORE, top_score - SECONDARY_GAP)
        answerable = [(c, e) for c, e in answerable
                      if c["norm_score"] >= cutoff]

        groups = self._group_answers(qtype, answerable, index)
        if not groups:
            return self._not_found(question, qtype,
                                   "无法从相关段落中确认答案",
                                   candidates=candidates)

        groups.sort(key=lambda g: (g["doc_count"], g["support"],
                                   g["max_score"]), reverse=True)
        top_group = groups[0]
        conflicting = len(groups) > 1
        return {
            "found": True,
            "question": question,
            "qtype": qtype,
            "answer": top_group["answer"],
            "conflict": conflicting,
            "answer_count": len(groups),
            "answers": groups,
            "evidence": [self._evidence(c) for c, _ in answerable][:6],
            "candidates": [self._evidence(c)
                           for c, _ in enriched[:6]],
            "reason": ("同一问题在不同文档中存在不同说法，已分组列出，请按出处核实"
                       if conflicting else "已在语料中找到可核实的出处"),
        }

    def _question_constraints(self, question: str) -> list[dict]:
        """抽取问题中的限定成分（时间/数字/地名/机构），用于校验候选句。

        例如「谁在2015年发布了产品？」中的 ``2015年`` 就是限定成分：
        只含 2019 年的句子即便含人名，也不能回答这个问题。
        """
        constraints: list[dict] = []
        for pat in _TIME_PATTERNS:
            for m in pat.finditer(question):
                text = m.group().replace(" ", "")
                if len(text) >= 2:
                    constraints.append({"type": "TIME", "text": text})
        for ent in self._extractor.ner.recognize(question):
            if ent["type"] in ("LOCATION", "ORGANIZATION"):
                constraints.append({"type": ent["type"], "text": ent["text"]})
        # 去重
        deduped, seen = [], set()
        for c in constraints:
            key = (c["type"], c["text"])
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        return deduped

    @staticmethod
    def _satisfies_constraints(sentence: str, constraints: list[dict]) -> bool:
        compact = re.sub(r"\s+", "", sentence)
        return all(c["text"] in compact for c in constraints)

    # 问题中用于限定「在什么情境下」的动作/关系词；
    # 与年份等实体同时出现时，答案句也必须含有该动作才算同一事件
    _ACTION_WORDS = ("成立", "创办", "创立", "接任", "继任", "上任", "离职",
                     "发布", "发表", "推出", "发行", "收购", "合并", "上市",
                     "出生", "逝世", "去世", "毕业", "加入", "访问", "签署",
                     "举办", "召开", "开工", "建成", "通车", "获奖", "发明",
                     "发现", "提出", "接任", "任命", "担任", "就任")

    def _question_actions(self, question: str) -> list[str]:
        return [w for w in self._ACTION_WORDS if w in question]

    @staticmethod
    def _satisfies_actions(sentence: str, actions: list[str]) -> bool:
        return any(a in sentence for a in actions)

    @staticmethod
    def _has_money_signal(sentence: str) -> bool:
        return bool(_MONEY_SIGNAL_RE.search(sentence))

    def _is_definitional(self, sentence: str, extracted: dict,
                         question: str) -> bool:
        """判断一个候选句是否真的是在下定义，而非提及概念的其它事实。"""
        # 抽取结果必须落在「是/指/即…」这类系表结构上
        answer = extracted.get("answer", "")
        if not answer:
            return False
        # 含金额/百分比/数量且问题没在问数值的，不是定义
        ents = extracted.get("entities", [])
        if any(e["type"] in ("MONEY", "PERCENT") for e in ents):
            return False
        if _MONEY_SIGNAL_RE.search(sentence):
            return False
        if any(e["type"] == "NUMBER" for e in ents) \
                and not re.search(r"(几|多少)(个|条|种|类|步)", question):
            return False
        return True

    def _group_answers(self, qtype: str, items: list,
                       index: QAIndex) -> list[dict]:
        """把同一答案的多个出处归并；不同答案保留为不同组。"""
        groups: dict[str, dict] = {}
        for cand, extracted in items:
            key = self._normalize_answer(extracted["answer"], qtype)
            if not key:
                continue
            passage = cand["passage"]
            source = {
                "doc_id": passage["doc_id"],
                "doc_name": passage["doc_name"],
                "sent_index": passage["sent_index"],
                "sentence": passage["text"],
                "char_start": passage["start"],
                "char_end": passage["end"],
                "answer": extracted["answer"],
                "answer_start": extracted.get("start", -1),
                "answer_end": extracted.get("end", -1),
                "focus_start": extracted.get("focus_start", -1),
                "focus_end": extracted.get("focus_end", -1),
                "score": cand["norm_score"],
                "context": index.neighbors(passage["doc_id"],
                                           passage["sent_index"]),
            }
            group = groups.get(key)
            if group is None:
                groups[key] = {
                    "key": key,
                    "answer": extracted["answer"],
                    "qtype": qtype,
                    "support": 1,
                    "doc_count": 1,
                    "max_score": cand["norm_score"],
                    "sources": [source],
                }
            else:
                group["support"] += 1
                group["max_score"] = max(group["max_score"],
                                         cand["norm_score"])
                if source["doc_id"] not in {s["doc_id"]
                                            for s in group["sources"]}:
                    group["doc_count"] += 1
                group["sources"].append(source)
        result = list(groups.values())
        # 组内按相关度排序出处
        for g in result:
            g["sources"].sort(key=lambda s: s["score"], reverse=True)
        return result

    @staticmethod
    def _normalize_answer(answer: str, qtype: str) -> str:
        """归一化答案文本，用于判断是否同一说法。"""
        a = re.sub(r"[\s，,。.、；;：:“”\"'（）()【】\[\]]+", "", answer)
        return a.lower()

    def _evidence(self, cand: dict) -> dict:
        p = cand["passage"]
        return {
            "doc_id": p["doc_id"],
            "doc_name": p["doc_name"],
            "sent_index": p["sent_index"],
            "sentence": p["text"],
            "char_start": p["start"],
            "char_end": p["end"],
            "score": cand["norm_score"],
            "coverage": round(cand["coverage"], 3),
        }

    def _not_found(self, question: str, qtype: str, reason: str,
                   stats: Optional[dict] = None,
                   candidates: Optional[list] = None) -> dict:
        return {
            "found": False,
            "question": question,
            "qtype": qtype,
            "answer": None,
            "conflict": False,
            "answer_count": 0,
            "answers": [],
            "reason": f"在语料库中找不到答案：{reason}",
            "index": stats,
            "candidates": [self._evidence(c) for c in (candidates or [])][:4],
        }


# ---------------------------------------------------------------------------
# 进程内单例（与其它 nlp 模块一致，按数据目录缓存）
# ---------------------------------------------------------------------------

_engines: dict[str, QAEngine] = {}
_engines_lock = threading.Lock()


def get_qa_engine(registry=None, data_root: Optional[str] = None) -> QAEngine:
    key = data_root or (registry.root if registry is not None else ".")
    with _engines_lock:
        eng = _engines.get(key)
        if eng is None:
            eng = QAEngine(registry, data_root)
            _engines[key] = eng
        elif registry is not None:
            eng.registry = registry
        return eng
