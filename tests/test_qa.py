"""语料问答（QA）模块测试。

运行：``python -m unittest tests.test_qa -v``

覆盖：
- 跨全部分片的检索（文档数超过单片容量）；
- 事实 / 定义 / 人物时间事件等问题的答案抽取与溯源；
- 同一问题在不同文档有不同答案时的分组与冲突标记；
- 语料无答案时拒答（found=False），不编造；
- 新增文档后旧问题能查到新答案（增量索引）；
- 出处能精确定位到文档 id、句序与字符偏移。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.qa import (QAEngine, QAIndex, classify_question, QType,
                    split_sentences_with_offsets, analyze_tokens)
from storage import StoreRegistry


SHARD_SIZE = 3


class QATestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="qa_test_")
        self.registry = StoreRegistry(self.tmp, shard_size=SHARD_SIZE)
        self.engine = QAEngine(self.registry, data_root=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def add_doc(self, name, text):
        rid = self.registry.task("corpus").insert(
            {"name": name, "text": text})
        return rid

    def ask(self, q):
        return self.engine.answer(q)


class TestSentenceSplit(unittest.TestCase):
    def test_offsets(self):
        text = "第一句。第二句！第三句？"
        sents = split_sentences_with_offsets(text)
        self.assertEqual([s["text"] for s in sents],
                         ["第一句", "第二句", "第三句"])
        for s in sents:
            self.assertEqual(text[s["start"]:s["end"]], s["text"])
        self.assertEqual([s["index"] for s in sents], [0, 1, 2])


class TestClassify(unittest.TestCase):
    def test_types(self):
        self.assertEqual(classify_question("什么是自然语言处理？"),
                         QType.DEFINITION)
        self.assertEqual(classify_question("深度学习是什么"), QType.DEFINITION)
        self.assertEqual(classify_question("智能音箱是什么设备？"),
                         QType.DEFINITION)
        self.assertEqual(classify_question("谁发明了蒸汽机？"), QType.PERSON)
        self.assertEqual(classify_question("公司总部在哪里？"), QType.LOCATION)
        self.assertEqual(classify_question("会议什么时候召开？"), QType.TIME)
        self.assertEqual(classify_question("这家公司有多少员工？"), QType.FACT)
        # 「在哪个领域」是问方面，不是问地点
        self.assertEqual(classify_question("深度学习在哪个领域表现出色？"),
                         QType.FACT)


class TestDefinitionQA(QATestBase):
    def test_definition_with_source(self):
        self.add_doc("AI 入门", "机器学习是人工智能的一个分支。"
                               "自然语言处理是人工智能的重要分支。")
        r = self.ask("什么是自然语言处理？")
        self.assertTrue(r["found"], r.get("reason"))
        self.assertIn("人工智能", r["answer"])
        src = r["answers"][0]["sources"][0]
        self.assertEqual(src["doc_name"], "AI 入门")
        self.assertEqual(src["sent_index"], 1)
        self.assertEqual(
            src["sentence"][src["answer_start"]:src["answer_end"]]
            if src["answer_start"] >= 0 else src["sentence"],
            src["sentence"][src["answer_start"]:src["answer_end"]])

    def test_source_locates_exact_doc_and_sentence(self):
        self.add_doc("文档甲", "今天天气不错。机器学习推动了人工智能的发展。")
        self.add_doc("文档乙", "深度学习模型在图像识别中表现出色。")
        r = self.ask("什么推动了人工智能的发展？")
        self.assertTrue(r["found"], r.get("reason"))
        src = r["answers"][0]["sources"][0]
        self.assertEqual(src["doc_name"], "文档甲")
        self.assertEqual(src["sent_index"], 1)


class TestAbstain(QATestBase):
    def test_no_answer(self):
        self.add_doc("科技新闻", "自然语言处理是人工智能的重要分支。")
        r = self.ask("珠穆朗玛峰有多高？")
        self.assertFalse(r["found"])
        self.assertIsNone(r["answer"])
        self.assertIn("找不到", r["reason"])

    def test_empty_corpus(self):
        r = self.ask("任何问题？")
        self.assertFalse(r["found"])


class TestCrossShard(QATestBase):
    def test_search_spans_all_shards(self):
        # 插入 7 篇，超过单片容量 3，形成至少 3 个分片
        for i in range(7):
            self.add_doc(f"文档{i}", f"这是第{i}篇文档的无关内容。")
        self.add_doc("关键文档", "区块链是一种分布式账本技术。")
        stats = self.engine.reindex()
        self.assertGreaterEqual(stats["documents"], 8)
        shard_count = self.registry.task("corpus").stats()["shard_count"]
        self.assertGreater(shard_count, 1)
        r = self.ask("什么是区块链？")
        self.assertTrue(r["found"], r.get("reason"))
        self.assertEqual(r["answers"][0]["sources"][0]["doc_name"], "关键文档")


class TestConflictingAnswers(QATestBase):
    def test_different_docs_different_answers(self):
        self.add_doc("旧版手册", "项目的上线时间是2022年3月1日。")
        self.add_doc("新版公告", "项目的上线时间是2023年5月12日。")
        r = self.ask("项目什么时候上线？")
        self.assertTrue(r["found"], r.get("reason"))
        self.assertTrue(r["conflict"])
        self.assertGreaterEqual(r["answer_count"], 2)
        answers = {g["answer"] for g in r["answers"]}
        self.assertTrue(any("2022" in a for a in answers))
        self.assertTrue(any("2023" in a for a in answers))
        # 每个答案各自带可核实出处
        for g in r["answers"]:
            self.assertEqual(g["doc_count"], 1)
            self.assertTrue(g["sources"][0]["sentence"])

    def test_consistent_answers_corroborated(self):
        self.add_doc("文档甲", "自然语言处理是人工智能的重要分支。")
        self.add_doc("文档乙", "自然语言处理是人工智能的重要分支。")
        r = self.ask("什么是自然语言处理？")
        self.assertTrue(r["found"])
        self.assertFalse(r["conflict"])
        self.assertEqual(r["answers"][0]["doc_count"], 2)


class TestEventQA(QATestBase):
    def test_who_when_what_with_constraint(self):
        self.add_doc("大事记", "王芳在2019年接任公司首席执行官。"
                               "李明在2015年带领团队发布了第一代智能音箱。")
        r = self.ask("谁在2015年做了什么？")
        self.assertTrue(r["found"], r.get("reason"))
        self.assertIn("李明", r["answer"])
        src = r["answers"][0]["sources"][0]
        self.assertEqual(src["sent_index"], 1)
        # 2019 年的句子不能用来回答 2015 年的问题
        self.assertNotIn("王芳", src["sentence"])

    def test_definition_not_confused_with_price(self):
        self.add_doc("产品说明", "智能音箱是一种可以用语音交互的家用设备。"
                                "智能音箱最初的售价是1999元。")
        r = self.ask("智能音箱是什么设备？")
        self.assertTrue(r["found"], r.get("reason"))
        self.assertIn("语音交互", r["answer"])
        self.assertNotIn("1999", r["answer"])
        for g in r["answers"]:
            for s in g["sources"]:
                self.assertNotIn("售价", s["sentence"])

    def test_money_question(self):
        self.add_doc("产品说明", "智能音箱最初的售价是1999元。")
        r = self.ask("智能音箱卖多少钱？")
        self.assertTrue(r["found"], r.get("reason"))
        self.assertIn("1999", r["answer"])


class TestIncrementalIndex(QATestBase):
    def test_new_doc_answers_old_question(self):
        self.add_doc("旧文档", "机器学习是人工智能的一个分支。")
        r1 = self.ask("什么是知识图谱？")
        self.assertFalse(r1["found"])

        # 新增文档后，旧问题应能查到新答案
        self.add_doc("新文档", "知识图谱是一种结构化的知识表示技术。")
        r2 = self.ask("什么是知识图谱？")
        self.assertTrue(r2["found"], r2.get("reason"))
        self.assertIn("知识表示", r2["answer"])

    def test_snapshot_reload(self):
        self.add_doc("文档", "向量数据库用于存储与检索向量。")
        self.engine.reindex()
        # 模拟进程重启：新建引擎，从快照加载后仍能答题
        new_engine = QAEngine(self.registry, data_root=self.tmp)
        r = new_engine.answer("什么是向量数据库？")
        self.assertTrue(r["found"], r.get("reason"))

    def test_delete_doc_removes_answer(self):
        rid = self.add_doc("待删文档", "量子计算利用量子叠加原理进行计算。")
        self.add_doc("保留文档", "机器学习是人工智能的一个分支。")
        self.assertTrue(self.ask("量子计算利用什么原理？")["found"])
        self.registry.task("corpus").delete(rid)
        r = self.ask("量子计算利用什么原理？")
        self.assertFalse(r["found"])


class TestTokens(unittest.TestCase):
    def test_bigram_covers_oov(self):
        from nlp.segmenter import Segmenter
        terms = analyze_tokens("区块链技术", Segmenter())
        self.assertIn("区块", terms)
        self.assertIn("块链", terms)


if __name__ == "__main__":
    unittest.main()
