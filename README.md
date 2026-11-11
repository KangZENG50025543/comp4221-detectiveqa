# 从长篇侦探小说里找证据｜COMP 4221

**我们想知道：让模型先多找一些候选段落，再从中挑出最有用的 5 段，能不能帮它找齐破案线索、答得更准？** 比如，一段交代了作案时间，另一段才给出人物行踪。只找到其中一段，模型即使选对了嫌疑人，我们也要问：它拿到的证据真的够吗？

## 当前复核方式：AI先评，组员确认

按新的组内安排，一个新的Agent已逐项复查 **12题、36份回答**，补好判断理由与具体核对提醒。4份初评有评级调整，旧意见保留；**人工确认仍为0**。

本地新页面已预填AI建议。组员核对后可直接确认、修改后确认，或标为待讨论；每次确认记录实际复核人，改名不会转移之前的确认归属。这一流程记录为 **AI辅助人工复核**，不称独立盲评。原答题分数和提交稿不改。

[操作说明](reports/AI初评与人工复核_20261111.txt) · [初评完成与待复核状态](reports/assisted_review_summary_20261111.json) · [页面生成脚本](scripts/build_assisted_review.py)。页面含原文，仅放本地；当前入口为 `runs/answer_diagnostic_dev_20261111_01/answer_review/assisted_20261111_02/verify_ai_review.html`。确认前显示的值均是AI建议，导出记录将它与实际人工判断分开保存。

## 最新进展：已开始真正回答问题

沿用固定的12道开发题，完成 **36次独立作答**：A原检索、C重排检索、N无材料。每次使用全新代理上下文；请求模型为 `gpt-6.1-sol`，实际后端修订号与采样参数未知。先锁答案再核分，原正式DeepSeek配置未改，外部付费API调用为0。

| 原标签命中数／完整分母 | A 原检索 | C 重排检索 | N 无材料 |
|---|---:|---:|---:|
| 小说 DetectiveQA | 3/4 | 3/4 | 0/4 |
| 多材料 MuSiQue（规范化EM） | 2/4 | 2/4 | 1/4 |
| 长文章 QuALITY | 4/4 | 4/4 | 0/4 |

拒答计未答对。MuSiQue的A/C平均token F1均为66.67%；一题答出了学校名却附加所在地，EM未命中，不能简单称为认错学校。数据集分别统计，不混成一个准确率。

**最有价值的例子：同一道小说题，A凭检索到的解释答对，C也答对，却明确承认依靠已有记忆。** 这说明准确率还不能代替证据支持检查。43条引用都通过了段号和原句存在检查；另有AI语义初审，但它不是人评或新的金标。每集仅4题，仍不能据此判断方法优劣或泛化。

当前按上方的新安排，由组员核验AI初评、校准歧义和充分性规则，再固定可重复的答题配置与保留集方案。原样本、分数和提交稿不改。人评页面含原文，仅保存在本地，公开仓库提供生成脚本；144个判断项和36个备注初始全部为空。

- [这轮结果、例子与接下来三步](reports/真实答题诊断与下一步_20261111.txt)
- [当前规划与完成状态](reports/current_plan_20261111.json)
- [输入与评分规则](reports/answer_diagnostic_protocol_20261111.json)、[真实答题分数](reports/answer_diagnostic_summary_20261111.json)、[逐题指标](reports/answer_diagnostic_items_20261111.json)
- [AI证据初审及局限](reports/answer_support_review_20261111.json)、[36个独立代理登记](reports/answer_agent_registry_20261111.json)、[原始回复哈希锁](reports/answer_responses_lock_20261111.json)

答题评分汇总保留核分时的状态快照；其语义审阅 `pending` 已由后续独立的AI支持报告补充。当前共23项单元检查通过，另检查了复核页面显示和导出恢复；没有生成任何人类评分。

## 同日更新：六题复核，及后续实验

**之前把“未收齐金标段落”直接称为“丢了作答关键线索”，解释过强，现更正。**

| 题目 | 逐题读原文后的判断 |
|---|---|
| 124-0 旅行钟 | 参考解析认错了返回拿手套的人；题目问动机，所给链条也未证明唯一动机。 |
| 209-1 太太的反应 | 核心证据已在；好奇、避谈等选项也有支持，需审查唯一性。 |
| 27-0 过去生活 | 已选段落直接支持较富裕的过去；缺的标注不是必需。 |
| 209-3 车站推人 | 已选到直接答案，且位于当前允许前缀内；答案揭示边界需要复核。 |
| 118-0 拜访目的 | 第一名片段停在提问处，遗漏紧随其后的解释；是较明确的上下文缺口。 |
| 118-2 继承秘密 | 已选内容明确支持秘密及其财产影响；缺的是背景段落。 |

这属于 AI 事后审校，不冒充组员人评；没有改原答案、截断线或旧分数。[逐题依据](reports/six_case_semantic_audit_20261111.json)列明原文段落号和不确定性。

下一步也已实跑：在全部 56 道开发题上，以原C为基线，比较“保留前四块，第五槽尝试补首位块的后继”及对称的前驱对照。邻块必须来自同一文档、同一原候选池；不用题号、金标或审阅标签选择。

| 原金标完整段落召回 | 原C | 补后继 | 补前驱 |
|---|---:|---:|---:|
| 小说20题 | 36.94% | 38.72% | 36.80% |
| 其中审阅过的6题 | 44.44% | 52.78% | 44.44% |
| 其余14道开发题 | 33.72% | 32.70% | 33.52% |

118-0补回部分动机解释，但仍未收齐后续解释。另一个涨分案例只是补回非必要线索，且其余14题没有改善，**暂不采用该规则作为已验证的改进**。这是开发诊断，非答案准确率、非泛化验证；同为五块但token不同。MuSiQue因原生文档边界全部不变，不算迁移成功；QuALITY仍无证据金标。详见[中文记录](reports/六题复核与后续实验_20261111.txt)、[实际汇总](reports/context_continuity_summary_20261111.json)、[配对子集与token检查](reports/context_continuity_paired_20261111.json)。

已通过全部16项测试，未调用付费问答API。要复现这一离线对照，在完成下文两套固定开发运行后执行：
```powershell
. ./scripts/environment.ps1
& $dqPython scripts/run_context_continuity_pilot.py
```

## 2026-11-11：已完成一轮跨数据集开发实验

新增 **20 道小说题、24 道 MuSiQue、12 道 QuALITY** 的检索试跑（小说包含之前试过的开发题），共 168 组证据输出。没有生成问答结果，也没有代填人工评分。

| 完整标注段落召回（按题平均） | A 直接选5 | B 最多10选5 | C 最多50选5 |
|---|---:|---:|---:|
| DetectiveQA，20题/5本小说 | 34.65% | 36.91% | 36.94% |
| MuSiQue，24题 | 67.01% | 76.39% | 77.43% |

这不是答案准确率。MuSiQue 的 C 实际只处理原生 20 块；QuALITY 的 C 处理 7—21 块，没有证据金标，所以没有编造召回率。MuSiQue 的 24 题共享组成问题后连成 21 组，不能当作 24 个独立案例。

这 7 题表现为“候选已覆盖全部指定标注段落，最终未收齐”；其中 6 题理论上可装入 5 块。**这不等于 6 次已确认的关键证据丢失。** 后续逐题语义审阅发现，多数存在替代充分证据、歧义或标注问题，只有一题有较明确的解释性上下文缺口；仍未测实际回答是否出错。详见[本轮中文记录](reports/下一步实验_20261111.txt)、[小说分书统计](reports/detectiveqa_dev_analysis_20261111.json)、[容量检查](reports/detectiveqa_capacity_analysis_20261111.json)和[跨数据集统计](reports/cross_dataset_dev_analysis_20261111.json)。

已生成 12 题离线复核包，每种材料 4 题。每题两人先独立判断，第三人处理分歧并轮换；方法匿名为 X/Y/Z，初始评分全部留空。表单含原文，仅在本地保存，仓库提供[生成脚本](scripts/build_cross_dataset_review.py)。

MuSiQue 的官方文件一致性也已[验证成功](reports/musique_provenance_20261111.json)；10 月 13 日报告中的未核实状态是历史记录。

组员可以从这里开始，不需要安装环境：

| 想了解什么 | 从哪里看 |
|---|---|
| 已经在小说上试过什么？ | [3 道开发题的检索结果](reports/retrieval_pilot_summary.json)。只测了找线索，还没有正式问答准确率。 |
| 为什么还要用其他数据集？ | [中文调查与试答](scout/调查与试答.txt)，以及[各数据集的来源和使用限制](scout/dataset_registry.json)。 |
| 我们三个人具体做什么？ | 对一题先由两人分别判断：材料够不够、答案能否唯一确定、解释哪一步缺证据。第三人复核分歧，轮换角色。这是[计划](scout/evaluation_plan.json)，不代表已完成人评。 |

## 小说上的对照实验

三种方法都仅使用**数据标注的揭示位置之前**的文本，也都只能给回答模型 **5 段**。这不保证原文尚未提前明示答案，209-3已发现这类问题，边界还需要语义复核。A 直接取相似度最高的 5 段；B 先取 10 段，重新打分后留 5 段；C 把候选增加到最多 50 段，同样留 5 段。另有一个诊断条件 D：直接给标注的证据段，用来区分「没找到」和「给了线索仍没答对」。

2026-10-10 的 3 道开发题中，按题平均，标注线索段落被完整找回的比例为 A **39.81%**、B **51.85%**、C **28.70%**。这是流程试跑，不能据此选定哪种方法最好，更不能当成答题准确率。当前[配置](configs/protocol.json)仍处于开发阶段，回答 API 未启用。

[DetectiveQA](https://huggingface.co/datasets/Phospheneser/DetectiveQA) 中文人工标注子集的 154 题中，[程序检查](reports/data_audit.json)通过 141 题，涉及 23 本小说。现有划分按小说分组，选取 20 道开发题、100 道测试题。[排除原因](reports/exclusions.jsonl)单独保留。另有 10 道测试题曾被助手探索性试答，题号和书号均已[登记](scout/exposure_registry.json)；正式评估时不能把这些题称为全新盲测。

## 换一种材料，结论还成立吗？

| 补充验证 | 检查的能力 | 2026-10-13 助手开发集盲答 |
|---|---|---:|
| [MuSiQue](https://github.com/StonyBrookNLP/musique) | 跨材料连起多条证据；个别题会混淆同名实体。10月13日使用社区镜像；11月11日已与官方原包核对一致。 | 5/9，3 道留空 |
| [QuALITY](https://github.com/nyu-mll/quality) | 长文章中的人物动机和语气；有些选项都有文本依据。 | 5/5 |
| [DetectBench](https://github.com/MikeGu721/DetectBench) | 短谜题和隐含线索；发现缺失证言、题干泄露答案等问题，暂不纳入正式评分。 | 4/6，2 道留空 |

这 20 题按[预定规则](scout/selftest_protocol.json)抽取；答案先锁定再核分，留空均算错。[核分汇总](scout/score_summary.json)用于检查题目质量，不是正式模型对比。后续分别报告各数据集的结果，保留曝光记录，按文章或小说隔离调试材料和最终验证材料。

## 需要复现时（Windows PowerShell）

只阅读报告不需要下载。要跑检索，可把仓库克隆到空间充足的磁盘，在仓库根目录运行以下命令。本项目使用 Python 3.12；完整检索流程使用 NVIDIA CUDA，模型和数据在本机单独下载。

```powershell
py -3.12 -m venv .venv
. ./scripts/environment.ps1
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
./scripts/setup.ps1
& $dqPython ./scripts/fetch_assets.py dataset
& $dqPython ./scripts/fetch_assets.py models
./run.ps1 -Action audit
./run.ps1 -Action tests
./run.ps1 -Action pilot -Limit 3 -RunName team_dev_pilot_01
```

重新试跑时使用新的运行名。`manual` 可在本机生成开发集核验表；`prepare-api` 只生成待检查的请求，不发送。小说全文、带原文的请求和阅读表、模型、密钥及含组员学号的课程提交件均未收入共享仓库。


## 补充数据开发流程

在本地下载官方 MuSiQue dev 和 QuALITY v1.0.1 dev，将它们分别放到
`dataset_scout_20261013/raw/musique/musique_ans_v1.0_dev.jsonl` 和
`dataset_scout_20261013/raw/quality/QuALITY.v1.0.1.htmlstripped.dev`。
来源与版本见 [登记表](scout/dataset_registry.json)，本轮实际文件哈希见[固定抽样清单](reports/cross_dataset_selection_20261111.json)。将 `scout/exposure_registry.json` 复制到 `dataset_scout_20261013/exposure_registry.json`，以排除此前探索样本。原始文件不上传仓库。

在已有本地环境和模型后运行：
```powershell
. ./scripts/environment.ps1
& $dqPython scripts/prepare_cross_dataset_dev.py --run-name cross_dataset_dev_20261111_01
& $dqPython scripts/run_cross_dataset_pilot.py --run-name cross_dataset_dev_20261111_01
& $dqPython scripts/run_retrieval_pilot.py --limit 20 --run-name dev_retrieval_20261111_01
& $dqPython scripts/build_cross_dataset_review.py
```

这些命令拒绝覆盖既有结果；已有同名运行时应查看其文件，或给两个运行换名并使用复核脚本的参数指定路径。新流程拒绝非开发集标记及含金标字段的普通输入，并校验固定文件哈希。正式测试集方案仍未冻结。本轮开发使用题号、文章/小说及组成问题记录在[曝光登记](reports/development_exposure_20261111.json)。
