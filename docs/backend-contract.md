# 新模型与数据集接入合同

RecLab 将推荐后端与 Agent 编排分开。接入新模型时，保留核心工具调用、实验预算、Trace 和独立评分器，仅增加遵循训练/预测入口协议的源码后端与经过验证的数据包；执行配置选择 SSH+tmux 或 Docker。

当前落实了通用接入边界、合成数据、可信的确定性 Popularity 基线，以及以 SASRec 为默认模型的配置驱动源码入口。`synthetic-smoke` 用于程序合同验证，其分数不能作为真实推荐质量证据。新链路已通过实际 AutoDL GPU 的三轮训练、独立预测和本机评分，见[实测报告](../reports/autodl-ssh-tmux-20261119.json)；这是 Codex 会话控制的短训验收，独立大模型 Agent 优化仍需配置 API 并正式运行。历史数据与冻结 B1 验收见 reports/week-01-completion-20261020.json。

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

## 任务配置与检查入口

`configs/beauty-sasrec-task.json` 是一次优化任务的配置入口，引用已有的模型/训练参数文件。两者职责分别是：任务配置登记源码目录、数据目录、评分器、优化指标、编辑范围、种子、预算、语言模型环境变量名和输出目录；`source.config` 选择源码目录中的 `configs/<名称>.json`，其中定义模型文件、类、参数和训练超参数。相对路径以任务配置文件所在目录解析，与启动命令的当前目录无关。

```bash
python -B -m reclab_agent.research_cli check --config configs/beauty-sasrec-task.json
```

检查入口读取公开数据、计算身份、检查源码语法和配置指向的模型类是否存在，核对评分器支持的指标。它不导入目标模型、不加载权重、不读取测试标签、不调用在线语言模型、不启动 Worker，也不创建配置指定的输出目录。`source.editable_files` 可只列模型和配置；完整源码清单与可编辑子集分别登记，通用工具冻结所有依赖，Agent 只能修改已批准的子集。

JSON 输出分别给出 `inputs_valid` 和 `ready_to_run`。检查通过时返回退出码 0；输入无效时返回 2。检查命令本身不创建运行绑定，`pending` 列出尚需检查的运行条件。静态检查不会证明模型构造函数、数值训练或独立预测已经运行成功，也不会产生研究计划批准记录；实际运行证据单独记录。

当前内置数据适配器为 `next_item_v1`，要求商品目录按 ID 连续排列为 1…N；评分器描述为 `next_item_ranking_v1`。任务配置只能选择登记名，不能提供要导入的 Python 模块或执行命令。后续新任务可以在可信控制端注册数据加载与评分器描述回调，复用检查逻辑：

```python
from reclab_agent.research.configuration import load_task_configuration
from reclab_agent.research.inspection import TaskAdapterRegistry, inspect_task

registry = TaskAdapterRegistry()
registry.register_dataset("my_dataset_v1", trusted_dataset_loader)
registry.register_evaluator("my_evaluator_v1", trusted_evaluator_description, score=trusted_score)
report = inspect_task(load_task_configuration("my-task.json"), registry=registry)
```

数据加载器接收数据包根路径，返回具有 `public`、`dataset_id`、`data_hash` 和 `info()` 的对象；公开目录沿用现有 Worker 的三个文件合同。评分器描述回调返回 `EvaluatorDescription`，登记协议哈希和支持的指标；`score(dataset, predictions)` 返回独立评分报告。通用工具调用固定评分回调并核对协议哈希、评价上下文哈希、有限指标和评价人数。仅登记描述仍可用于检查，但不能执行实验。不同任务的完整数据协议和评分实现不会仅凭一个登记名自动生成。

`agent.model_env`、`agent.api_key_env` 只填写环境变量名，不填写密钥本身。报告只输出是否配置，不返回变量值。当前 AutoDL 示例使用 `execution.backend=ssh_tmux` 和连接配置路径；检查命令不建立 SSH 连接。Docker 模式另填 `execution.policy`，继续按用户要求暂缓。连接字段和七个远程工具见[执行层说明](execution.md)。

## 配置驱动的模型入口

仓库中的 `backends/sasrec/` 是可直接登记为源码树的自包含目录，仅包含 `src/` 与 `configs/`。训练、预测和公共数据读取不依赖相邻的 `semantic-id-rec` 仓库。默认 `src/sasrec.py` 保持原始模型源码字节不变。

统一配置为 `backends/sasrec/configs/default.json`。模型文件、类名、构造参数和训练参数都由该文件提供，例如：

```json
{
  "model": {
    "file": "src/sasrec.py",
    "class": "SASRec",
    "kwargs": {"hidden_size": 64, "num_layers": 1, "num_heads": 1, "dropout": 0.4}
  },
  "max_length": 50,
  "epochs": 30,
  "patience": 5,
  "batch_size": 256,
  "eval_batch_size": 256,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "grad_clip": 5.0,
  "device": "cuda"
}
```

`model.file` 相对源码快照根目录解析，必须位于 `src/`。运行时将真实目录大小作为 `num_items`、输入长度作为 `max_length` 注入构造函数，`kwargs` 不允许覆盖这两个值。模型加载只发生在实验作业内，控制端不导入模型代码。

同类下一商品推荐模型实现以下接口后，更换模型只需提供对应的源码文件并修改配置，无需改训练、预测入口：

- 继承 `torch.nn.Module`，构造函数接收 `num_items`、`max_length` 和配置中的 `kwargs`。
- `max_length` 属性与配置一致；`encode(histories)` 接收 `[batch, max_length]` 的左填充商品 ID，`0` 为 padding。
- `catalog_logits(states)` 返回 `[batch, num_items]` 分数，第 `i` 列对应商品 ID `i+1`。
- 权重使用标准 `state_dict`，能通过 `torch.load(..., weights_only=True)` 恢复。

当前训练方法固定为训练历史前缀预测下一商品，使用全目录交叉熵和 AdamW；不同任务或不同训练目标需要另外的适配。输入数据参数 `--data` 指向包含三个公开 JSON 的目录本身，不是原始完整 JSON。数据/输出挂载路径与随机种子由 Worker 提供，不在模型配置中重复指定。

训练按公开验证集 NDCG@20 选择最佳轮次，只保存一个 `best.pt`、训练配置与逐轮日志，不读测试标签。独立预测入口核对源码、数据、配置和种子身份后，读取只读权重并输出 `predictions.json`；只开放 `--split valid`。训练过程的选轮分数只作诊断，最终报告仍以宿主独立评分器为准。

默认 B1 配置请求 CUDA；不可用时明确失败，CPU 只能在配置中显式选择。代码和数据的静态检查、GPU 短训、独立 Agent 优化分别记录证据，不能相互替代。Docker 按用户要求暂缓。

## 实验身份与首次接入验收

每个作业的元数据必须锁定源码快照哈希、镜像身份、public 数据哈希、配置与随机种子。正式复现实验使用不可变镜像 digest；`python:3.11-slim` 等标签仅用于本地合同示例。模型产物还要绑定来源作业，避免不同源码或数据的权重混用。

新后端接入按以下次序验证：

1. 创建并验证数据包，确认 public 挂载不包含测试标签。
2. 在明确资源限制的容器中跑通训练入口，确认产物写入约定目录。
3. 跑通预测入口并严格验证预测 JSON。
4. 用可信 targets 和 histories 独立评分，复现人工基线结果。
5. 再开放模型、损失、采样或训练逻辑给 Agent 修改，保留评分器和数据协议只读。

通用合同测试通过仅证明边界与程序行为正确。新模型的真实训练、算力成本、重复种子波动及质量收益都要在接回该后端后独立验证并记录。
