# 从长篇侦探小说里找证据｜COMP 4221

**我们想知道：让模型先多找一些候选段落，再从中挑出最有用的 5 段，能不能帮它找齐破案线索、答得更准？** 比如，一段交代了作案时间，另一段才给出人物行踪。只找到其中一段，模型即使选对了嫌疑人，我们也要问：它拿到的证据真的够吗？

组员可以从这里开始，不需要安装环境：

| 想了解什么 | 从哪里看 |
|---|---|
| 已经在小说上试过什么？ | [3 道开发题的检索结果](reports/retrieval_pilot_summary.json)。只测了找线索，还没有正式问答准确率。 |
| 为什么还要用其他数据集？ | [中文调查与试答](scout/调查与试答.txt)，以及[各数据集的来源和使用限制](scout/dataset_registry.json)。 |
| 我们三个人具体做什么？ | 对一题先由两人分别判断：材料够不够、答案能否唯一确定、解释哪一步缺证据。第三人复核分歧，轮换角色。这是[计划](scout/evaluation_plan.json)，不代表已完成人评。 |

## 小说上的对照实验

三种方法都只能读到揭晓答案之前，也都只能给回答模型 **5 段**。A 直接取相似度最高的 5 段；B 先取 10 段，重新打分后留 5 段；C 把候选增加到最多 50 段，同样留 5 段。另有一个诊断条件 D：直接给标注的证据段，用来区分「没找到」和「给了线索仍没答对」。

2026-10-10 的 3 道开发题中，按题平均，标注线索段落被完整找回的比例为 A **39.81%**、B **51.85%**、C **28.70%**。这是流程试跑，不能据此选定哪种方法最好，更不能当成答题准确率。当前[配置](configs/protocol.json)仍处于开发阶段，回答 API 未启用。

[DetectiveQA](https://huggingface.co/datasets/Phospheneser/DetectiveQA) 中文人工标注子集的 154 题中，[程序检查](reports/data_audit.json)通过 141 题，涉及 23 本小说。现有划分按小说分组，选取 20 道开发题、100 道测试题。[排除原因](reports/exclusions.jsonl)单独保留。另有 10 道测试题曾被助手探索性试答，题号和书号均已[登记](scout/exposure_registry.json)；正式评估时不能把这些题称为全新盲测。

## 换一种材料，结论还成立吗？

| 补充验证 | 检查的能力 | 2026-10-13 助手开发集盲答 |
|---|---|---:|
| [MuSiQue](https://github.com/StonyBrookNLP/musique) | 跨材料连起多条证据；个别题会混淆同名实体。本轮为已校验的社区镜像，未与官方原包比对。 | 5/9，3 道留空 |
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
