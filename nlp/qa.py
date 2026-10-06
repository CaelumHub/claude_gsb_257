"""检索式问答（Retrieval-based QA）。

在 :mod:`nlp.retrieval` 的 BM25 句子检索之上做**抽取式**问答，严格遵守
「语料里有才答、没有就明说找不到」的原则：

1. **问句分类**：用规则把问题识别为 人 / 时间 / 地点 / 机构 / 数字 /
   定义 / 事实 七类，决定从证据句里抽取什么类型的答案；
2. **答案抽取**：答案永远是证据原句的一个子串（实体或定义短语），
   绝不自由生成；抽不到合格答案就返回 :meth:`QAEngine.ask` 的
   ``found=False``，并附上最相关的原文供人工判断；
3. **多源分辨**：同一问题在不同文档有不同答案时，按答案值分组，
   一致答案合并为「多处印证」，不一致则显式标记 ``conflict=True``，
   每个答案各自带可核实的出处（文档、分片文件、句序号、字符偏移）。

NER / 分词均复用平台自带的纯 Python 实现。
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Optional

from .lexicon import (GIVEN_NAME_CHARS, LOCATIONS, LOC_SUFFIXES, PERSONS,
                      SURNAMES)
from .ner import NERExtractor
from .retrieval import CorpusRetriever, QUESTION_STOP
from .segmenter import Segmenter


# 相对相关度门槛：只保留分数不低于 top1 该比例的证据（跨文档答案需要并列保留）
REL_CUTOFF = 0.35


# 问句类型判定（按优先级排列，先匹配先生效）
_TIME_PATTERNS = [
    re.compile(r"什么时候|何时|哪一?年|哪一?天|哪一?个月|几点|多少号|什么时间"),
]
# 真正询问地点的说法；「哪个 + 任务/领域/方面」等抽象名词不属于地点
_LOC_PATTERNS = [
    re.compile(r"哪里|哪儿|何地"),
    re.compile(r"哪个?地方|什么地方"),
    re.compile(r"哪个?(?:城市|国家|地区|省份|位置|地址|站点|学校|公园|大楼)"),
    re.compile(r"在哪(?:里|儿)?(?:$|[，,。？?\s])|地处"),
]
_PERSON_PATTERNS = [
    re.compile(r"^谁|是谁|谁人|由谁|被谁|让谁|给谁|谁是"),
]
_ORG_PATTERNS = [
    re.compile(r"哪个(?:机构|组织|公司|单位|学校|大学|部门)|什么(?:机构|组织|公司|单位|学校|大学|部门)"),
]
# 「多少」后需接真正的量词，避免「有多少/多少钱？」之外的宽泛误报；
# 年份（2015 年）不会带量词，因此也不会被误当成数量答案
_NUM_PATTERNS = [
    re.compile(r"多少(?:个|条|位|本|张|块|斤|米|公里|元|岁|年时间|次|台|份|名|家|页|题|分)"),
    re.compile(r"几(?:个|条|年|位|本|张|块|斤|米|公里|元|岁|次|台|份|名|家|页|题|分|点|号)"),
    re.compile(r"百分之|比例是多少|占比多少|数量(?:是)?多少|多大面积"),
]
_DEF_PATTERNS = [
    re.compile(r"是什么|是啥|什么是|啥是|的定义|指的是|所谓|是指|定义是|含义是|什么意思"),
]

# 定义抽取的句式模板（focus 为被定义术语）
_DEF_CUE_PATTERNS = [
    re.compile(r"是指(?P<def>[^，。；,;！？!?]{2,60})"),
    re.compile(r"指的是(?P<def>[^，。；,;！？!?]{2,60})"),
    re.compile(r"定义(?:为|是)(?P<def>[^，。；,;！？!?]{2,60})"),
    re.compile(r"所谓(?P<term>[^，。；,;！？!?]{1,30})[，,]?是(?P<def>[^，。；,;！？!?]{2,60})"),
    re.compile(r"(?P<term>[^，。；,;！？!?]{1,30})是(?P<def>[^，。；,;！？!?]{2,60}?(?:的)?(?:一种|一类|一门|技术|方法|学科|系统|工具|概念|过程|活动|分支|手段))"),
]

# 时间兜底：年份 / 年代 / 世纪 / 相对时间（NER 正则只覆盖具体年月日与时刻）
_TIME_FALLBACK = re.compile(
    r"\d{4}\s*年(?:\s*代|\s*世纪)?"
    r"|[一二两三四五六七八九十百零〇]+\s*年(?:\s*代|\s*世纪)?"
    r"|\d{1,4}\s*年代"
    r"|(?:清朝|明朝|唐朝|宋朝|元朝|汉朝|秦朝|古代|近代|现代|当代|改革开放以来|建国初期)")

# 人名兜底：NER 在词层面识别，若分词把人名切成单字（如「李/明」）会漏报；
# 这里在原文字符串上直接补扫。两字名要求名字用字命中常见名字表；
# 三字名（如少数民族 / 复姓风格）只要求首尾合理，中间放宽。
_GIVEN_CLASS = "".join(sorted(GIVEN_NAME_CHARS))
_PERSON_FALLBACK = re.compile(
    rf"[{''.join(sorted(SURNAMES))}](?:[{_GIVEN_CLASS}]|[^\s，。；,;：:、的了是在和与及！？!?（）()《》\"'][{_GIVEN_CLASS}])")

# 不可能作为姓名首字（NER 偶发把「后李」这类分词误判为人名，答案层再过滤一道）
_PERSON_BAD_FIRST = set("后前后里外内上下中大小新旧好坏多少高低长短远近早晚期初末内外")

# 姓名左侧应是句首、标点、称谓引导或动词等边界，避免从句中截取（如「然后李…」）
_PERSON_LEFT_OK = re.compile(
    r"(?:^|[\s，。；,;：:、（）(）]|创始人|创办人|创立者|负责人|主席|总统|主任|"
    r"是|由|被|让|给|和|与|跟|及|叫|名为|名字是|名叫|作者|教授|先生|女士)$")

# 常见称谓，命中后把后缀剥掉，避免「李明先生」整体当作答案
_TITLE_SUFFIX = re.compile(r"(先生|女士|小姐|同志|老师|教授|博士|经理|局长|主席|书记)$")
_FOCUS_PREFIX_RE = re.compile(
    r"^(?:请问一下|请问|谁能告诉我|我想知道|我想了解|想知道|了解一下|咨询一下)?")
_FOCUS_STRIP_RE = re.compile(
    r"(?:是什么|是啥|什么是|啥是|的定义|定义是|的含义|含义是|什么意思|指的是|是指|"
    r"是谁|谁是|在哪里|在哪儿|在什么地方|地处哪里|"
    r"什么时候|什么时间|何时|哪一?年|哪一?个月|哪一?天|哪天|几点|多少号|"
    r"在哪里成立|哪儿成立|"
    r"多少[^，。？?！!]{0,6}|几[^，。？?！!]{0,6}|哪里|哪儿|哪儿|哪个|哪些|什么|谁|为何|为什么)?"
    r"(?:举行|举办|召开|发生|成立|出生|去世|逝世|离世|发布|出台|完成|开始|结束)?"
    r"(?:呢|吗|啊|？|\?|。|！|!)$")


class QAEngine:
    """检索 + 抽取的问答引擎。"""

    def __init__(self, retriever: CorpusRetriever,
                 ner: Optional[NERExtractor] = None,
                 segmenter: Optional[Segmenter] = None):
        self.retriever = retriever
        self.ner = ner or NERExtractor()
        self.segmenter = segmenter or Segmenter()

    # ------------------------------------------------------------------
    # 问句分类与焦点
    # ------------------------------------------------------------------
    def classify(self, question: str) -> str:
        q = question.strip()
        for pat in _TIME_PATTERNS:
            if pat.search(q):
                return "time"
        for pat in _LOC_PATTERNS:
            if pat.search(q):
                return "location"
        for pat in _PERSON_PATTERNS:
            if pat.search(q):
                return "person"
        for pat in _ORG_PATTERNS:
            if pat.search(q):
                return "organization"
        for pat in _NUM_PATTERNS:
            if pat.search(q):
                return "number"
        for pat in _DEF_PATTERNS:
            if pat.search(q):
                return "definition"
        return "fact"

    def focus(self, question: str, qtype: str) -> str:
        """提取问句焦点（被问对象），用于检验证据句是否真的在谈这个主题。"""
        q = question.strip()
        q = _FOCUS_PREFIX_RE.sub("", q)
        # 「什么是 X」「谁是 X」：焦点在引导词之后
        m = re.match(r"^(?:什么是|啥是|谁是)(.+?)(?:呢|吗|啊)?$", q)
        if m:
            return m.group(1).strip()
        q = _FOCUS_STRIP_RE.sub("", q)
        return q.strip()

    # ------------------------------------------------------------------
    # 答案抽取（答案必须是证据原句的子串）
    # ------------------------------------------------------------------
    def _entities_in(self, sentence: str) -> list[dict]:
        return self.ner.recognize(sentence)

    @staticmethod
    def _in_question(text: str, question: str) -> bool:
        return text and text in question

    @staticmethod
    def _valid_person(name: str) -> bool:
        """过滤 NER 的人名误报（如 HMM 分词产生的「后李」）。"""
        if len(name) < 2:
            return False
        if name[0] in _PERSON_BAD_FIRST or name[0] not in SURNAMES:
            return False
        return True

    @staticmethod
    def _valid_location(name: str) -> bool:
        """过滤地名误报：NER 后缀规则会把「星河（科技）」这类专名截短为地名。

        可信地名：在内置地名词典中；或以省/市/区等行政区划后缀结尾；
        或长度 >= 2 且以山河湖海等自然地名后缀结尾且首字不是常见专名动词。
        """
        if name in LOCATIONS:
            return True
        if name.endswith(("省", "市", "区", "县", "镇", "村", "自治区", "州")):
            return True
        # 「星河 / 银河 / 先河」这类是公司 / 产品名常见前缀，不可单独作地点
        if name in {"星河", "银河", "先河", "天河"} and name not in LOCATIONS:
            return False
        if any(name.endswith(s) for s in LOC_SUFFIXES) and len(name) >= 2:
            return True
        return False

    def _extract_typed(self, qtype: str, sentence: str,
                       question: str) -> list[str]:
        """按问句类型从证据句抽取候选答案实体（原文子串）。"""
        entities = self._entities_in(sentence)
        answers: list[str] = []

        def add_type(types):
            for ent in entities:
                if ent["type"] not in types:
                    continue
                if (types == {"PERSON"} and ent["text"] not in PERSONS
                        and not self._valid_person(ent["text"])):
                    continue
                if types == {"LOCATION"} and not self._valid_location(ent["text"]):
                    continue
                if not self._in_question(ent["text"], question):
                    answers.append(ent["text"])

        if qtype == "person":
            add_type({"PERSON"})
            # NER 漏报时的原文兜底（匹配姓氏开头的 2~3 字姓名）
            for m in _PERSON_FALLBACK.finditer(sentence):
                token = _TITLE_SUFFIX.sub("", m.group())
                if len(token) < 2 or self._in_question(token, question):
                    continue
                # 左边界检查：姓名前应是句首 / 标点 / 称谓引导，防止从句中截取
                if not _PERSON_LEFT_OK.search(sentence[:m.start()]):
                    continue
                # 已被 NER 识别为更长人名实体的子串则不重复加
                if any(token in e["text"] and e["type"] == "PERSON"
                       and e["text"] != token for e in entities):
                    continue
                answers.append(token)
        elif qtype == "location":
            add_type({"LOCATION"})
        elif qtype == "organization":
            add_type({"ORGANIZATION"})
        elif qtype == "time":
            add_type({"TIME", "DATE"})
            for m in _TIME_FALLBACK.finditer(sentence):
                token = m.group().strip()
                if token and not self._in_question(token, question):
                    answers.append(token)
        elif qtype == "number":
            add_type({"NUMBER", "PERCENT", "MONEY"})

        # 去重保序
        seen, uniq = set(), []
        for a in answers:
            if a not in seen:
                seen.add(a)
                uniq.append(a)
        return uniq

    def _extract_definition(self, sentence: str, focus: str,
                            question: str) -> Optional[str]:
        for pat in _DEF_CUE_PATTERNS:
            m = pat.search(sentence)
            if not m:
                continue
            groups = m.groupdict()
            term = groups.get("term")
            definition = groups.get("def")
            if not definition:
                continue
            # 带 term 的句式，term 必须与焦点相关（同一术语）
            if term is not None and focus and focus not in term and term not in focus:
                continue
            definition = definition.strip("，。；,;：: ")
            if len(definition) >= 2 and not self._in_question(definition, question):
                return definition

        # 兜底：「<focus>是<名词短语>」——焦点后紧跟「是」
        if focus and focus in sentence:
            idx = sentence.index(focus) + len(focus)
            rest = sentence[idx:]
            m = re.match(r"是(.{2,60})", rest)
            if m:
                tail = m.group(1).strip("，。；,;：: ")
                # 简单名词性判断：不含明显谓语续接
                if tail and not self._in_question(tail, question):
                    return tail
        return None

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def ask(self, question: str, top_k: int = 8,
            min_score: float = 1.5) -> dict:
        """对自然语言问题作答。

        :param min_score: 绝对相关度门槛；在此之上还会按 top1 分数的
            ``REL_CUTOFF`` 比例做相对截断，避免小语料下固定阈值失灵。

        返回结构::

            {
              "question": ..., "type": ..., "found": bool,
              "answer": 主答案（找不到时为空串）,
              "conflict": 不同文档是否给出不一致答案,
              "groups": [{"answer": ..., "sources": [出处...]} ...],
              "evidence": [所有相关原文段落（含分数）],
              "message": 找不到 / 冲突时的说明,
            }
        """
        question = re.sub(r"\s+", " ", question).strip()
        qtype = self.classify(question)
        focus = self.focus(question, qtype)

        evidence = self.retriever.search(question, top_k=top_k, min_score=0.0)
        if evidence:
            top = evidence[0]["score"]
            # 以相对门槛为主（小语料绝对分天然偏低）；top1 足够强时再叠加
            # 绝对门槛，top1 较弱时只要求一个很低的非零分，随后交给焦点覆盖把关
            floor = top * REL_CUTOFF if top < min_score else max(
                min_score, top * REL_CUTOFF)
            evidence = [e for e in evidence if e["score"] >= floor]

        # 焦点覆盖：证据句必须在原文中包含焦点实义词（字面包含即可，
        # 与分词粒度无关），防止只靠「公司/问题」等宽泛词撞分。
        focus_terms = self._focus_terms(focus)
        if focus_terms:
            related = [e for e in evidence
                       if any(t in e["text"].lower() for t in focus_terms)]
            if related:
                evidence = related

        # 冲突分辨：实体类问题可能有多篇文档各说各话，弱相关文档会被相对
        # 门槛截掉。补回「句中含同类实体」的段落（仍需命中焦点），保证
        # 不同答案都能被呈现出来供核实。
        evidence = self._recover_alternative_sources(
            question, qtype, evidence, focus_terms)

        groups: "OrderedDict[str, dict]" = OrderedDict()
        for ev in evidence:
            answers = self._answers_for(qtype, ev, focus, question)
            # 一句可同时支持多个答案值（如「张三和李四」两个 PER）
            for ans in answers:
                key = self._normalize_answer(ans)
                if key not in groups:
                    groups[key] = {"answer": ans, "sources": []}
                groups[key]["sources"].append(self._source(ev))

        found = bool(groups)
        conflict = len(groups) > 1
        groups_list = list(groups.values())
        # 主答案取「最强证据」支持的答案；同等强度时出处多者优先
        groups_list.sort(
            key=lambda g: (max(s["score"] for s in g["sources"]),
                           len(g["sources"])),
            reverse=True)

        message = ""
        if not found:
            if evidence:
                message = "在语料中找到了相关段落，但未能从中抽取到确定答案，请查看下方原文核实。"
            else:
                message = "在全部语料分片中未找到与该问题相关的内容，无法作答（不做猜测）。"
        elif conflict:
            message = (f"不同文档对该问题给出了 {len(groups_list)} 种不同答案，"
                       "已分组列出，请按出处核实。")

        return {
            "question": question,
            "type": qtype,
            "focus": focus,
            "found": found,
            "answer": groups_list[0]["answer"] if found else "",
            "conflict": conflict,
            "groups": groups_list,
            "evidence": [self._evidence_view(e) for e in evidence],
            "message": message,
        }

    _ENTITY_TYPES = {
        "person": {"PERSON"},
        "location": {"LOCATION"},
        "organization": {"ORGANIZATION"},
        "time": {"TIME", "DATE"},
        "number": {"NUMBER", "PERCENT", "MONEY"},
    }

    def _recover_alternative_sources(self, question: str, qtype: str,
                                     evidence: list[dict],
                                     focus_terms: list[str]) -> list[dict]:
        """补回被相对门槛截掉、但确实含同类答案实体的弱相关段落。

        做法：用较宽的候选池（不设相对门槛、扩大 top_k）重新检索，凡是
        命中焦点且句中带有「问题所问实体类型」的段落都保留——它们可能
        正是另一篇文档给出的不同答案。为防止「年/在」这类单字焦点造成
        误纳，要求候选必须与焦点有至少一个长度 >= 2 的实义重叠，或者
        命中了检索器的子串匹配。
        """
        wanted = self._ENTITY_TYPES.get(qtype)
        if not wanted:
            return evidence
        kept = list(evidence)
        known = {(e["doc_id"], e["sent_index"]) for e in kept}
        # 只有长度 >= 2 的实义焦点才参与弱候选把关，防止「年 / 在」
        # 这类单字焦点把无关句子误纳为冲突来源
        strong_terms = [t for t in focus_terms if len(t) >= 2]
        wider = self.retriever.search(question, top_k=20, min_score=0.0)
        for cand in wider:
            key = (cand["doc_id"], cand["sent_index"])
            if key in known:
                continue
            text = cand["text"]
            if strong_terms and not any(t in text.lower() for t in strong_terms):
                continue
            answers_here = self._extract_typed(qtype, text, question)
            if not answers_here:
                continue
            cand = dict(cand)
            cand["recovered"] = True
            kept.append(cand)
            known.add(key)
        return kept

    def _answers_for(self, qtype: str, ev: dict, focus: str,
                     question: str) -> list[str]:
        sentence = ev["text"]
        if qtype == "definition":
            definition = self._extract_definition(sentence, focus, question)
            return [definition] if definition else []
        if qtype in ("person", "location", "organization", "time", "number"):
            return self._extract_typed(qtype, sentence, question)
        # fact：没有可抽取的结构化答案，不做猜测
        return []

    def _focus_terms(self, focus: str) -> list[str]:
        if not focus:
            return []
        return [t.lower() for t in self.retriever._tokens(focus)
                if t not in QUESTION_STOP]

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        return re.sub(r"[\s，,。.、；;：:]+", "", answer)

    @staticmethod
    def _source(ev: dict) -> dict:
        """出处：定位到具体文档、分片文件、第几句与字符偏移。"""
        return {
            "doc_id": ev["doc_id"],
            "doc_name": ev["doc_name"],
            "shard_index": ev["shard_index"],
            "shard_file": ev["shard_file"],
            "position": ev["position"],
            "sent_index": ev["sent_index"],
            "start": ev["start"],
            "end": ev["end"],
            "sentence": ev["text"],
            "score": ev["score"],
        }

    def _evidence_view(self, ev: dict) -> dict:
        return {
            "score": ev["score"],
            "doc_id": ev["doc_id"],
            "doc_name": ev["doc_name"],
            "shard_index": ev["shard_index"],
            "shard_file": ev["shard_file"],
            "sent_index": ev["sent_index"],
            "start": ev["start"],
            "end": ev["end"],
            "text": ev["text"],
            "matched_terms": ev.get("matched_terms", []),
        }

    def stats(self) -> dict:
        return self.retriever.stats()

    def reindex(self, force: bool = True) -> dict:
        return self.retriever.sync(force=force)
