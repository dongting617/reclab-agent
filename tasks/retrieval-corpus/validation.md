# Independent validation scoring

验证集预测由固定评分器重算 NDCG 和 Recall；模型自述的分数不能替代评分。用户配对 bootstrap 只描述固定模型下的用户抽样不确定性。

Checkpoint selection and candidate comparison use validation data. Paired user bootstrap is not repeated training and does not establish cross-seed stability.
