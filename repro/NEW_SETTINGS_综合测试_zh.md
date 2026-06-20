# MINJA 在新数据集 / 新开源记忆系统 / 新任务设定上的综合测试（中文）

模型：DeepSeek-V3.2-Exp（SiliconFlow）；嵌入：BGE-m3。代码在 `repro/new/`（`lib.py` 记忆后端+数据集，`attacks.py` 攻击目标+MINJA 注入循环，`run_all.py` 矩阵）。结果 `repro/results/new_matrix.json`。**失败 case 单独写在 `repro/FAILURES_综合_zh.md`。**

## 测试维度（一次性把能覆盖的都做了）

**(1) 新的开源 agent 记忆系统**（都路由到 SiliconFlow 的 BGE 嵌入 + DeepSeek）：
- `BGE-cosine`：自建余弦检索（论文式 few-shot demo 记忆）。
- `LangChain + FAISS`：真实 LangChain 记忆栈（FAISS 向量库）。
- `LlamaIndex`：真实 LlamaIndex `VectorStoreIndex` 检索。

**(2) 新数据集**：
- `MMLU`（4 选）：security_studies / professional_medicine / elementary_mathematics。
- `CommonsenseQA`（5 选，本地 `train_rand_split.jsonl`）：全新常识推理域。
- 合成 `tool-routing`（6 工具）与 `sentiment`（3 类）：全新任务格式。

**(3) 新任务设定 / 攻击目标**（不止论文那个 ASCII 位移玩具）：
- `ascii_shift`：论文原目标（看到 victim → 答案字母 +4）。
- `misinformation`：**更真实**——把含 victim 词的问题改写为返回某个**固定错误答案**（B）。
- `tool_hijack`：**agentic**——含 victim 词的请求被强制路由到 target 工具（send_email）。
- `label_flip`：**分类**——含 victim 词的文本被翻成 target 情感（positive，掩盖差评）。

## 主结果（ASR%，每格 n_inject=6 / n_test=8 / n_benign=8 / k=5）

| 场景（攻击 \| 数据集→target） | BGE-cosine | LangChain+FAISS | LlamaIndex |
|---|--:|--:|--:|
| `ascii` \| MMLU.security | 100 | 50 | 88 |
| `ascii` \| MMLU.**patient** | **0** | **0** | **0** |
| `ascii` \| MMLU.total | 75 | 88 | 100 |
| `ascii` \| **CSQA**.water（5 选，全新数据集） | 50 | 75 | 62 |
| `misinfo`（→固定错答 B） \| MMLU.security | 100 | 100 | 100 |
| `misinfo` \| MMLU.**patient** | **25** | **12** | 88 |
| `misinfo` \| **CSQA**.water | 100 | 100 | 100 |
| `tool_hijack`（→send_email）\| 合成工具路由 | **100** | **100** | **100** |
| `label_flip`（→positive）\| 合成情感分类 | **100** | **100** | **100** |

## 真实记忆构成变体（记忆里混入 +10 条"含 victim 的正常记录"，BGE）

| 场景 | 理想 ASR | 真实 ASR | 变化 |
|---|--:|--:|--:|
| `misinfo` \| MMLU.security→B | 100 | **12.5** | **塌** |
| `tool_hijack` \| invoice→send_email | 100 | **100** | 不塌 |
| `label_flip` \| headphones→positive | 100 | **100** | 不塌 |

## 关键结论

1. **MINJA 能迁移到真实开源记忆框架。** 在 **LangChain+FAISS** 和 **LlamaIndex** 上，ASR 与自建 BGE 检索相当、甚至更高（如 ascii MMLU.total：LlamaIndex 100 vs BGE 75）。说明攻击不依赖某种特制检索，对生产级向量记忆栈同样有效。
2. **MINJA 能泛化到新数据集。** 全新的 **CommonsenseQA**（5 选常识推理）上，misinformation 攻击 100%、ascii 50–75%。
3. **MINJA 能泛化到新任务设定，而且更狠。** `tool_hijack`（把含 victim 的请求强制路由到 send_email）和 `label_flip`（把差评翻成好评）在**所有 3 个记忆系统上都是 100%**——这些**更真实**的攻击目标（输出一个固定工具/标签）比论文那个 ASCII 位移玩具**更可靠**，因为目标是"复制一个固定输出"而非"执行一个推理变换"。
4. **失败 case 集中且可解释**（详见 `FAILURES_综合_zh.md`）：(a) 深推理域（professional_medicine/"patient"）抗注入；(b) 真实记忆构成把 QA misinformation 从 100 打到 12.5，但**对 tool/sentiment 不起作用**；(c) ascii 这种"推理变换型"目标最不稳。

> 一句话：换数据集、换真实记忆框架、换更真实的攻击目标，MINJA 的核心威胁**普遍成立且常常更强**；但它的失效边界也很清楚——**深推理任务** + **触发词不在答案链上** + **记忆里有同词正常记录** 三者叠加时会被显著削弱。这与 RQ1/RQ2 的结论完全一致。
