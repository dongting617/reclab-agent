# 独立推荐评分协议

`reclab_agent.grading` 只接受普通数据，不运行候选代码，不读取 pickle 或权重。
调用方从可信数据适配器提供 `targets` 与 `histories`；候选仅提供按分数降序排列的商品 ID。

`RankingProtocol(dataset_id, split, catalog, ks=(10, 20), mask_seen=True)` 固定全目录、每用户一个目标的评价口径。
catalog 和 K 是唯一整数元组，构造时排序；布尔值不当作整数。协议内容使用规范 JSON 计算 SHA-256。
训练、验证、测试以及不同目录、K、过滤规则的结果不能直接比较。

`grade_rankings(predictions, targets, histories, protocol)` 严格要求三个字典有相同的非空字符串用户集合。
历史交互允许重复，但所有商品必须存在于目录。开启历史过滤时目标不能在历史里，预测也不能包含历史商品。
预测必须合法、不重复，至少提供 `min(max(ks), eligible_catalog_size)` 个商品，防止通过省略困难用户或短列表美化结果。

单目标 Recall@K 为目标是否进入前 K；NDCG@K 为命中时的 `1 / log2(rank + 1)`，否则为零。
报告保存整体均值、逐用户 rank 和指标、协议与评价上下文哈希。上下文哈希绑定每用户的真实目标和 seen 集合。
目标未出现在提交的前缀中时 rank 为 null，所有未命中的 K 指标为零。

`compare_reports(baseline, candidate, bootstrap_samples=1000, seed=0)` 只比较相同协议、用户、目标与历史的报告。
它按用户 ID 对齐，给出候选减基线的均值差和配对用户 bootstrap 的 95% percentile 区间。
各指标使用同一组重采样用户。区间衡量用户抽样差异，不包含训练随机种子变化或多次选优偏差；负收益同样如实返回。
评分和比较不会将任务成功等同于指标提升。

调用示例：

```python
from reclab_agent.grading import RankingProtocol, grade_rankings, compare_reports

protocol = RankingProtocol("tiny-v1", "valid", (1, 2, 3), (1, 2))
trusted_targets = {"user-a": 3}
trusted_histories = {"user-a": [1]}
baseline = grade_rankings({"user-a": [2, 3]}, trusted_targets, trusted_histories, protocol)
candidate = grade_rankings({"user-a": [3, 2]}, trusted_targets, trusted_histories, protocol)
comparison = compare_reports(baseline, candidate)
```
