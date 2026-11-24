# Task recovery

任务恢复先查询已有作业和产物，不重复提交或重置额度。模型角色在工具操作结果不明时报告需要核验，不盲目重放副作用。

Resume reconciles existing jobs and preserves spent budgets. A controller checkpoint is distinct from a model weight checkpoint; mid-epoch training continuation is not implemented.
