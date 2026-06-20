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

## 主结果（ASR%，每格 n_inject=8 / n_test=10 / n_benign=12 / k=5）

<!-- FILL new_matrix -->
（矩阵进行中，数字稍后补）

## 真实记忆构成变体（记忆里混入含 victim 的正常记录，BGE）

<!-- FILL realistic -->
（稍后补）

## 关键结论

<!-- FILL -->
（稍后补）
