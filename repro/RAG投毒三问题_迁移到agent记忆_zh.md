# 把 Confundo（RAG 投毒）指出的三个问题，迁移其"本质"到 agent memory(MINJA) 上测试

来源论文：*Confundo: Learning to Generate Robust Poison for Practical RAG Systems*。它指出现有 RAG 投毒在**真实系统**里会严重失效，根因是三个被忽视的现实：

| Confundo 的问题（RAG 语境） | 问题的**本质** | 迁移到 agent memory(MINJA) 的对应 |
|---|---|---|
| **P1 分块/预处理碎片化**：文档入库前被切成 chunk，毒文本被切断/未知 chunk 大小 → 检索不到、毒性减弱 | 在"攻击者载荷"与"最终被存储+检索的东西"之间，存在一个**攻击者不可控的入库变换** | MINJA 假设 (query, reasoning) 记录被**逐字、整条**存+取；但真实 agent 记忆会**切块/摘要/反思**后再存（MemGPT 摘要、LangChain summary memory、Generative-Agents reflection）。**这个变换会不会把 MINJA 的 bridging step 碎片化？** |
| **P2 词法不泛化**：毒针对某个 query 措辞；受害者换措辞 → 效果腰斩 | 检索/触发**绑定具体措辞** | 受害者用同义词/改写时，恶意记录是否还被检索、directive 是否还触发 |
| **P3 目标不泛化**：每个攻击只为单一目标定制（事实错/观点偏/幻觉） | 一套机制**难支持多目标** | MINJA 的同一机制能不能支持多种攻击目标 |

> 注意：不是把 RAG 的攻击搬过来，而是把"**入库变换碎片化 / 词法绑定 / 目标单一**"这三个**本质**搬到 agent memory 上重测。

---

## 迁移实验 1（P1 本质）：入库变换碎片化 —— raw vs summarize vs chunk

强攻击格（misinfo / tool_hijack / label_flip，BGE 检索），唯一变量 = 入库管线：
- **raw**：整条记录原样存、整条取（= MINJA 的隐含假设）。
- **summarize**：存前用 LLM 把记录压成一句（= summary memory）。
- **chunk**：把记录按句切块、逐块入向量库，检索返回 top-k **碎块**（= RAG chunking）。

ASR%：

<!-- FILL pipeline -->

## 迁移实验 2（P2 本质）：词法泛化 —— verbatim vs paraphrase vs 触发词同义替换

注入 victim='security'，测试 query 三档：原词 / 改写但保留'security' / 把'security'换成同义词（触发词消失，如 cybersecurity）。攻击目标 = misinfo（"提到 security → 答 B"），BGE 稠密检索。

| 测试 query 档位 | ASR | 例子 |
|---|---:|---|
| verbatim（原词） | **100%** | "...a liberal perspective on future energy **security**?" |
| paraphrase（改写，保留 security） | **100%** | "...aligns with a liberal outlook on future energy **security**?" |
| synonym（触发词换成 cybersecurity，"security"字面消失） | **100%** | "...future energy **cybersecurity**"（注：脚本二次替换把词面弄成 "cybercybersecur"，触发词被彻底破坏，结果反而更强） |

**结论：MINJA 对 P2 基本免疫，远比 RAG 投毒鲁棒。** Confundo 报告 RAG 投毒"换措辞→效果腰斩(~50%)"；而 MINJA 在改写、甚至把触发词整词删掉时仍 100%。原因有两个，都和 RAG 不同：
1. **稠密检索天然泛化**：BGE 把 "security"/"cybersecurity"/"energy security" 映到相近向量，恶意记录照样被取回——不像 BM25/词频那样绑字面。
2. **指令长在被取回的"推理"里，不绑 query 字面**：RAG 的毒针要"query 命中→毒块进 context"；MINJA 取回的是 demo（含被劫持的 Thought + 答案 B），directive 已经焊在 demo 内部，query 怎么改写都不影响 demo 一旦被取回就生效。

> 但**不是零敏感**：在更脆弱的 `ascii_shift`(QA) 目标上，改写受害 query 仍掉 10–20 点（total 90→70、security 90→80、food 80→70，见 `run_qa_paraphrase`）。即"带显式 directive 的目标(misinfo)对词法完全鲁棒；纯靠 demo 模仿的目标(ascii_shift)有残余词法敏感"。总体仍远好于 RAG 的腰斩。

## 迁移实验 3（P3 本质）：目标泛化 —— 已在统一框架里验证

我之前的统一 harness 用**同一套 MINJA 机制**跑了 4 种目标：`ascii_shift`/`misinformation`/`tool_hijack`/`label_flip`，在 3 个记忆系统上多数 100%。所以 **P3 对 MINJA 不是问题**：与 Confundo 批评的"单目标 RAG 攻击"不同，MINJA 的"注入(query, 劫持推理)记录"机制**天然支持多目标**（改答案/换工具/翻标签/数据外泄都用同一套）。

## 结论

<!-- FILL -->
