# GPU 研究与迁移验收

日期：2026-11-23 至 2026-11-24。状态：completed_with_negative_cases。

| 模型 | 方案 | 状态 | NDCG@20 | Recall@20 | 种子 |
|---|---|---|---|---|---|
| SASRec initial negative case | sasrec-b1-retrain | completed | 0.08045030576134957 | 0.15543205449982073 | 2026 |
| SASRec initial negative case | candidate-1 | failed | — | — | 2026 |
| GRURec migration | grurec-baseline | completed | 0.04133806899498413 | 0.08941376837576193 | 2026 |
| GRURec migration | candidate-1 | failed | — | — | 2026 |
| SASRec dropout remaining approved slot | sasrec-b1-retrain | completed | 0.08045030576134957 | 0.15543205449982073 | 2026 |
| SASRec dropout remaining approved slot | candidate-1 | completed | 0.07741930180175842 | 0.1477680172104697 | 2026 |

训练、预测和评分状态分别保留。数值来自预测文件的本地独立验证集评分，不预设候选提升。

控制器中断后原远端训练进程继续运行。首次恢复遇到 SSH 超时及未提交预测的状态处理问题，已经保留失败记录并修复；经身份核验复用同一训练，未将首次恢复记为自动通过。

详细源码 Diff、指标、checkpoint/评分哈希与预算见 [JSON 记录](gpu-acceptance-20261123.json)。原始权重、数据和会话保留在本地，不进入 Git。
