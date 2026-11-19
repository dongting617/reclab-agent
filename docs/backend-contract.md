# 新模型与数据集接入合同

RecLab 将推荐后端与 Agent 编排分开。接入新模型时，保留核心工具调用、实验预算、Trace 和独立评分器，仅增加能在容器中运行的源码后端与经过验证的数据包。

当前落实的是通用接入边界、合成数据和可信的确定性 Popularity 基线。`synthetic-smoke` 用于程序合同验证，其分数不能作为真实推荐质量证据。真实Beauty适配及冻结B1完整验证集推理已复现，并通过独立评分；可被Agent修改、隔离训练的SASRec源码后端仍待接入，当前没有由这条新链路完成的神经模型训练结果。详见reports/week-01-completion-20261020.json。

## 数据包

```text
dataset/
  public/
    manifest.json
    train.json
    valid.json
  private/
    test.json
```

`manifest.json` 包含 `schema_version: 1`、合法的 `dataset_id`、`profile`、`task: "next_item"` 及 `catalog`。目录商品 ID 必须为不重复的正整数，布尔值不能充当整数。

`train.json` 是 `dict[str, list[int]]`，保存每用户的时间有序训练历史。`valid.json` 是 `dict[str, {"history": list[int], "target": int}]`；同一用户的验证 history 必须等于冻结的训练历史，单一验证目标不能已在历史中。`private/test.json` 采用相同的单目标结构，通常测试 history 为 train＋valid 交互。数据适配器必须先完成时间划分、商品映射和过滤，再冻结数据包。

候选容器的 `/data` **只挂载 `public/`**。测试标签留在可信控制器的 private 区域，不进入工作区、镜像、RAG 索引或候选日志。当前 `SourceBackend.prediction_command()` 只开放 valid；最终测试是独立的可信冻结操作，不能通过传入 `--split test` 绕过冻结。

`DatasetBundle` 将 public manifest、train 和 valid 的规范内容组成 `data_hash`。独立评分报告还使用评价上下文哈希绑定该划分的真实目标和 seen 集合。private 标签变动不会改变可公开的训练数据身份，但会改变最终测试报告的评价身份，因此旧、新标签结果不能错误配对。

## 源码后端

```text
source/
  src/train.py
  src/predict.py
  configs/default.json
```

注册 `SourceBackend(backend_id, source_files, image)`，source_files 至少声明训练和预测入口，并将配置和全部必要模块放入自包含的源码快照。后端不能依赖开发机的相邻仓库、绝对源码路径或 sibling imports；需要的公共模块应打包到当前快照或冻结的镜像依赖中。

后端训练逻辑命令为：

```text
python -B /workspace/src/train.py --data /data --output /output --config /workspace/configs/default.json
```

预测逻辑命令为：

```text
python -B /workspace/src/predict.py --data /data --output /output --config /workspace/configs/default.json --model /model --split valid
```

`training_command(config_name="default")` 与 `prediction_command(split="valid", config_name="default")` 采用相同配置命名规则；配置名需通过标识符校验，不能传任意路径。Docker Worker 使用固定 `/usr/local/bin/python -E -s -B` 前缀运行这些入口，入口及后续参数与后端合同一致；前缀加强不会开放任意命令。

训练入口从只读数据和配置读取输入，将权重、训练配置与日志写入自己的 `/output`。推理必须是独立容器：Worker 只接受同源码、数据、配置和种子的已完成训练作业，核对输出回执后将其只读挂为 `/model`；预测入口从 `/model` 恢复权重，将排名写到推理作业自己的 `/output/predictions.json`。不能把原训练输出读写复用为推理输出，也不能从宿主机反序列化候选 pickle。控制器不直接 import 候选源码。

Worker 将种子注入 `RECLAB_SEED`，**入口必须主动读取并配置所用的随机源**。注入环境变量本身不会自动设置 Python、NumPy、PyTorch、CUDA 或 DataLoader 的随机行为。例如入口先读取 `seed = int(os.environ["RECLAB_SEED"])`，再按实际依赖配置对应随机源及数据加载器。确定性算子选择、库/驱动版本和重复 seed 实验另需明确记录，不能仅凭有这个变量就宣称完全可复现。

`predictions.json` 必须是 `dict[str, list[int]]`：键是评价用户 ID，值为分数降序的 top-20 商品 ID。列表不能有重复、未知或已见商品，必须严格覆盖全部评价用户。若过滤后的目录不足 20 个商品，提交完整 eligible 目录。目标不在 top-20 时允许未命中，不能跳过该用户。

候选可以打印调试指标，但正式结果由可信评分器读取预测后独立计算。禁止通过候选自报的数值决定是否改进；评分协议见 `grading-protocol.md`。

## 实验身份与首次接入验收

每个作业的元数据必须锁定源码快照哈希、镜像身份、public 数据哈希、配置与随机种子。正式复现实验使用不可变镜像 digest；`python:3.11-slim` 等标签仅用于本地合同示例。模型产物还要绑定来源作业，避免不同源码或数据的权重混用。

新后端接入按以下次序验证：

1. 创建并验证数据包，确认 public 挂载不包含测试标签。
2. 在明确资源限制的容器中跑通训练入口，确认产物写入约定目录。
3. 跑通预测入口并严格验证预测 JSON。
4. 用可信 targets 和 histories 独立评分，复现人工基线结果。
5. 再开放模型、损失、采样或训练逻辑给 Agent 修改，保留评分器和数据协议只读。

通用合同测试通过仅证明边界与程序行为正确。新模型的真实训练、算力成本、重复种子波动及质量收益都要在接回该后端后独立验证并记录。
