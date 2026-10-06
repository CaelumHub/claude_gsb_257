"""检索式问答（QA）与跨分片 BM25 检索的单元测试。

运行：``python -m unittest discover -s tests -v``

覆盖：
- 句子偏移与出处定位（文档 / 分片文件 / 第几句 / 字符偏移）；
- 跨分片检索（shard_size=1 强制每文档一个分片）；
- 事实性问题的抽取式作答（人物 / 时间 / 地点 / 定义）；
- 语料无答案时明确 ``found=False``，不编造；
- 不同文档给出不同答案时的分组与冲突标记；
- 新增 / 修改 / 删除文档后的增量同步（旧问题查到新答案）；
- 快照持久化（重建引擎后索引仍可复用）。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp import CorpusRetriever, QAEngine
from nlp.retrieval import split_sentences_with_offsets
from storage import StoreRegistry


def _build_registry(root: str, docs: list[tuple[str, str]],
                    shard_size: int = 1) -> StoreRegistry:
    reg = StoreRegistry(root, shard_size=shard_size)
    for name, text in docs:
        reg.task("corpus").insert({"name": name, "text": text})
    return reg


class TestSentenceOffsets(unittest.TestCase):
    def test_offsets_point_to_original(self):
        text = "第一句话。第二句话！第三句？"
        sents = split_sentences_with_offsets(text)
        self.assertEqual([s[0] for s in sents],
                         ["第一句话", "第二句话", "第三句"])
        for sent, start, end in sents:
            self.assertEqual(text[start:end], sent)

    def test_newline_split(self):
        text = "标题行\n正文内容。"
        sents = split_sentences_with_offsets(text)
        self.assertEqual([s[0] for s in sents], ["标题行", "正文内容"])
        for sent, start, end in sents:
            self.assertEqual(text[start:end], sent)

    def test_repeated_substring_offsets(self):
        # 同一句子文本重复出现时，偏移必须各自指向正确位置（不能用 str.index 回溯）
        text = "重复。重复。重复。"
        sents = split_sentences_with_offsets(text)
        self.assertEqual(len(sents), 3)
        self.assertEqual([(s[1], s[2]) for s in sents],
                         [(0, 2), (3, 5), (6, 8)])
        for sent, start, end in sents:
            self.assertEqual(text[start:end], sent)


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.reg = _build_registry(self.tmp, [
            ("科技新闻", "自然语言处理是人工智能的重要分支。深度学习模型在图像识别中表现出色。"),
            ("评论", "这个产品非常好用，但是价格有点贵。"),
        ])
        self.retriever = CorpusRetriever(
            self.reg.task("corpus"),
            os.path.join(self.tmp, "qa_index.json"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cross_shard_search(self):
        hits = self.retriever.search("深度学习图像识别", top_k=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["doc_name"], "科技新闻")
        # 出处信息齐全，且跨分片定位正确
        top = hits[0]
        self.assertIn("shard_file", top)
        self.assertGreaterEqual(top["sent_index"], 0)
        self.assertEqual(
            "深度学习模型在图像识别中表现出色",
            self.reg.task("corpus").get(top["doc_id"])["text"][top["start"]:top["end"]])

    def test_no_hit(self):
        self.assertEqual(self.retriever.search("火星航海烹饪玄学"), [])

    def test_locate(self):
        loc = self.reg.task("corpus").locate("corpus_2")
        self.assertIsNotNone(loc)
        self.assertEqual(loc["shard_file"], "shard_000001.json")
        self.assertEqual(loc["position"], 0)


class TestQAAnswers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.reg = _build_registry(self.tmp, [
            ("公司简介A", "智汇科技有限公司成立于2015年，总部位于深圳。"),
            ("公司简介B", "宏图科技成立于2008年，总部位于杭州。"),
            ("术语表", "自然语言处理是人工智能的重要分支，研究人与计算机之间用自然语言进行有效通信的理论与方法。"
                       "分词是指将连续的文本切分成词语序列的过程。"),
            ("事件", "2020年3月，张伟带领团队发布了开源深度学习框架星辰。"),
        ], shard_size=1)
        self.qa = QAEngine(CorpusRetriever(
            self.reg.task("corpus"),
            os.path.join(self.tmp, "qa_index.json")))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_person_what_when(self):
        r = self.qa.ask("谁在2020年发布了深度学习框架？")
        self.assertTrue(r["found"])
        self.assertEqual(r["answer"], "张伟")
        src = r["groups"][0]["sources"][0]
        self.assertEqual(src["doc_name"], "事件")
        # 句子与字符偏移可核实
        doc = self.reg.task("corpus").get(src["doc_id"])["text"]
        self.assertEqual(doc[src["start"]:src["end"]], src["sentence"])

    def test_time(self):
        r = self.qa.ask("智汇科技什么时候成立？")
        self.assertTrue(r["found"])
        self.assertEqual(r["answer"], "2015年")

    def test_definition(self):
        r = self.qa.ask("分词是什么？")
        self.assertTrue(r["found"])
        self.assertIn("切分成词语序列", r["answer"])
        # 定义答案必须是原句子串
        src = r["groups"][0]["sources"][0]
        self.assertIn(r["answer"], src["sentence"])

    def test_conflicting_answers_grouped(self):
        r = self.qa.ask("公司总部在哪里？")
        self.assertTrue(r["found"])
        self.assertTrue(r["conflict"])
        answers = {g["answer"] for g in r["groups"]}
        self.assertEqual(answers, {"深圳", "杭州"})
        # 每个答案各自带可核实出处，且来自不同文档 / 分片
        docs = {src["doc_name"] for g in r["groups"] for src in g["sources"]}
        self.assertEqual(docs, {"公司简介A", "公司简介B"})
        for g in r["groups"]:
            for src in g["sources"]:
                self.assertTrue(src["shard_file"].startswith("shard_"))

    def test_not_found_does_not_fabricate(self):
        r = self.qa.ask("火星的平均地表温度是多少？")
        self.assertFalse(r["found"])
        self.assertEqual(r["answer"], "")
        self.assertEqual(r["evidence"], [])
        self.assertIn("未找到", r["message"])

    def test_question_classification(self):
        self.assertEqual(self.qa.classify("他什么时候来的？"), "time")
        self.assertEqual(self.qa.classify("公司在哪里？"), "location")
        self.assertEqual(self.qa.classify("是谁发明了电话？"), "person")
        self.assertEqual(self.qa.classify("熵的定义是什么？"), "definition")
        self.assertEqual(self.qa.classify("这个框架好用吗？"), "fact")


class TestIncrementalIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.reg = _build_registry(self.tmp, [
            ("公告A", "项目发布会于2019年在北京举行。"),
        ], shard_size=1)
        index_path = os.path.join(self.tmp, "qa_index.json")
        self.qa = QAEngine(CorpusRetriever(self.reg.task("corpus"), index_path))
        self.index_path = index_path

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_new_document_adds_answer(self):
        r = self.qa.ask("项目发布会什么时候举行？")
        self.assertEqual(r["answer"], "2019年")

        # 新增一篇给出不同答案的文档（落入新分片），旧问题应能查到新答案
        self.reg.task("corpus").insert({
            "name": "公告B", "text": "项目发布会推迟至2021年在上海举行。"})
        r = self.qa.ask("项目发布会什么时候举行？")
        self.assertTrue(r["conflict"])
        self.assertEqual({g["answer"] for g in r["groups"]},
                         {"2019年", "2021年"})

    def test_modified_document_refreshes_answer(self):
        self.reg.task("corpus").insert({
            "name": "公告B", "text": "项目发布会推迟至2021年在上海举行。"})
        rec = [x for x in self.reg.task("corpus").all()
               if not x.get("_deleted") and x["name"] == "公告B"][0]
        self.reg.task("corpus").delete(rec["id"])
        self.reg.task("corpus").insert({
            "name": "公告B", "text": "项目发布会推迟至2022年在上海举行。"})
        r = self.qa.ask("项目发布会什么时候举行？")
        self.assertEqual({g["answer"] for g in r["groups"]},
                         {"2019年", "2022年"})

    def test_deleted_document_removes_answer(self):
        self.reg.task("corpus").insert({
            "name": "公告B", "text": "项目发布会推迟至2021年在上海举行。"})
        rec = [x for x in self.reg.task("corpus").all()
               if not x.get("_deleted") and x["name"] == "公告B"][0]
        self.reg.task("corpus").delete(rec["id"])
        r = self.qa.ask("项目发布会什么时候举行？")
        self.assertFalse(r["conflict"])
        self.assertEqual(r["answer"], "2019年")

    def test_snapshot_persistence(self):
        self.qa.ask("项目发布会什么时候举行？")
        self.assertTrue(os.path.exists(self.index_path))
        # 重新构建检索器（模拟进程重启），快照可复用且答案一致
        qa2 = QAEngine(CorpusRetriever(self.reg.task("corpus"), self.index_path))
        r = qa2.ask("项目发布会什么时候举行？")
        self.assertEqual(r["answer"], "2019年")


if __name__ == "__main__":
    unittest.main()
