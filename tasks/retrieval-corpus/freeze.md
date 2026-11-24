# Evaluation separation

工具参考调用通过率、诊断任务成功率和模型推荐效果使用不同分母。冻结任务答案不进入模型上下文或检索库，负结果和失败任务不能从分母删除。

Freeze source, task definitions and corpus hashes before evaluation. Reference-policy checks validate the evaluator and never count as end-to-end Agent success.
