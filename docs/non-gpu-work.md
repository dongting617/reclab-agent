# 本地功能、评测与验收

更新日期：2026-11-24。沿用现有分层与四阶段流程，不新增编排框架。本页区分功能实现、离线测试和真实接口结果。

## 本地工作台与同一份状态

2026-11-24 界面更新：正式页面已按确认后的浅色原型接入真实会话、报告及 SSE。新增主指标提升图、刷新最佳的关键改进、实验详情、实际源码差异和可筛选事件；未核验评分、缺少比较身份、过程要求或审计未通过的候选，不进入最佳曲线。参见[新版工作台与验收](workbench-design.md)。

安装 `pip install -e ".[service]"`，执行：

```bash
python -m reclab_agent.cli.web --root . --port 8000
python -m reclab_agent.cli.research runs list --root .
```

访问 `http://127.0.0.1:8000`。`application/runs.py` 同时为 Web 和 CLI 提供任务、研究会话、报告和事件视图：

- 登记任务保存在 `.reclab/state/runs.sqlite3`。
- 研究会话直接读取 `reclab.json` 指向任务的输出目录，不另复制一套研究状态。
- 工作台通过现有报告模型核验指标和实际源码差异；显示假设、实验、报告和 Trace。
- 相同幂等键和相同请求返回原任务；同键不同内容返回冲突。
- SSE 按持久事件序号继续读取；正在写入的最后一行 Trace 等待完整后再返回。
- 等待执行/配置的任务可取消；运行中的研究会话需执行端确认，不伪造取消成功。

默认 Web 不绑定真实训练执行器，只登记任务、读取会话和执行证据演示。高级调用方可通过 `create_app(root, run_task=...)` 注入执行器，并负责同一个 RunStore 的生命周期更新。CLI 仍是实际研究启动入口。

内置浏览器通道不可用，首次选择 Edge 也因可能显示无关已登录页面被自动审批拒绝。获得用户明确许可后，仅在新建的本地工作台标签页完成了页面截图观察、证据演示、报告切换、任务登记与取消、Trace 交互验收。验收发现取消后事件未更新，已修复为重新读取持久化事件，并在浏览器确认出现 `cancelled` 记录。页面资源、API、幂等、取消、事件续读和共享状态也有 HTTP 集成测试。

## 可恢复的多角色协作

通过任务的 `task.mode = "team"` 或诊断 CLI 的 `--agent team` 选择：

1. Planner：通过数据输出 schema 提交结构化计划，没有业务执行工具。
2. Executor：使用宿主批准且当前阶段允许的工具。
3. Reviewer：只检查证据和结论，通过数据 schema 提交复核，没有业务执行工具；最多要求一次修订。

三者共享模型请求和工具范围，不为每个角色额外分配训练额度。角色状态保存在对话/Trace 旁的 `.roles.json`；原对话和预算保留。恢复时跳过已完成角色；已完成检查点直接返回保存结果，不再调用模型或重复写完成事件。

模型返回的纯 JSON 或单个完整 JSON 代码块都可解析；夹杂解释文字、多代码块或不符合字段约束的响应仍拒绝。该解析也复用于原计划执行与反思策略。

`tools/role_results.py` 定义 `role_plan`、`role_review` 的数据格式，复用既有原生函数调用协议。提交结果转成数据消息，不进入业务工具注册表，也不计为一次实际工具执行。调用记录仍计入同一个模型预算。

若工具执行结果不明，返回 `reconciliation_required`，先核验工具和实验状态。它不是训练中间 epoch 续训，也不承诺任意第三方工具恰好执行一次。

## CPU 文本模型与检索

`integrations/text_models.py` 将 Sentence Transformers 转成现有索引的两个 callable。所有模型固定使用 `device="cpu"`、`trust_remote_code=False`；默认仅查本地缓存。

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[retrieval]"
python -m reclab_agent.cli.evaluate retrieval --modes bm25
python -m reclab_agent.cli.evaluate retrieval --modes bm25 dense hybrid reranked --download-models
```

配置模型及版本：

```bash
python -m reclab_agent.cli.evaluate retrieval \
  --embedding-model BAAI/bge-small-zh-v1.5 \
  --reranking-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --modes bm25 dense hybrid reranked --download-models \
  --output reports/local-retrieval-models.json
```

默认 embedding 面向中文，默认 reranker 面向英文；语种和语料不匹配时可能变差，不将“接了模型”视为收益。可替换为已有的中文或多语模型，并用 `--embedding-revision`、`--reranking-revision` 固定提交。报告记录请求版本、可取得的实际模型提交和库版本。

研究任务可在原 `research` 配置内增加：

```json
{
  "retrieval": {
    "mode": "hybrid",
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "allow_download": false
  }
}
```

以上是 `research` 对象的片段；不替换整个任务文件。先在本地准备模型缓存，再运行研究。默认仍为 BM25，缺少真实模型时不回退并冒称 dense/hybrid。

标注集 `tasks/retrieval.json` 包含 12 篇受版本约束的技术文档、12 个有答案查询和 1 个无答案查询。答案按来源和行号标注，查询答案不进入语料。该集合用于工程验收，不代表一般检索能力。报告 Recall@5、MRR、延迟和无答案空结果比例；dense 未校准拒答阈值，引用能定位也不等于所有结论都受支持。首条查询耗时包含延迟加载模型的成本。

BM25 验收：Recall@5 = 1.0，MRR = 0.9583，无答案查询返回空结果。证据见 [BM25 报告](../reports/local-retrieval-bm25-20261123.json)。

## 30+30 场景及冻结约束

当前默认 `tasks/diagnostic-v2/diagnostic-dev.json` 和 `tasks/diagnostic-v2/diagnostic-frozen.json`，各 30 个**合成诊断决策场景**，不是实际训练记录。

开发组覆盖收益、主指标退化、保护指标退化、训练失败、缺少评分和种子不匹配；冻结组覆盖协议/数据不匹配、预算耗尽、停止未确认、必需诊断缺失和重复候选。两组的场景组不重叠。同组的五个变体相关，不作为独立样本做统计显著性声明。

`tasks/diagnostic-v2/local-evaluation-freeze.json` 固定 manifest 哈希。生成脚本仅用于建立首个版本，已有冻结文件时拒绝覆盖。新版本应另行冻结，不为改善测试成绩改标签。

`.gitattributes` 为 `tasks/**` 和 `reports/*.json` 保留字节，避免 Windows Git 自动换行破坏冻结哈希。[跨平台检出复核](../reports/frozen-checkout-20261124.json)在独立临时仓库验证 `core.autocrlf=true`：修复前 24 个文件中 21 个字节改变，修复后为 0，没有改写标签或重冻。

```bash
python -m reclab_agent.cli.evaluate diagnostics \
  --manifest tasks/diagnostic-v2/diagnostic-dev.json \
  --output reports/diagnostic-dev-reference.json

python -m reclab_agent.cli.evaluate diagnostics \
  --manifest tasks/diagnostic-v2/diagnostic-frozen.json \
  --output reports/diagnostic-frozen-reference.json
```

默认 `reference` 运行确定性参考策略，只校验任务/工具/评分链路；两组 30/30 不能当作真实 Agent 成功率。

初稿中 dataset ID 含场景族名称，可能构成额外提示。已保留原始 v1 冻结文件和真实试跑记录，另建 v2 匿名标识版本并作为默认；旧分数不迁移到 v2。v2 的两组参考检查均为 30/30。2026-11-24 在 GPU 阶段首次推送后另行运行匿名 v2 正式模型对照，结果见最新验收汇总。

v1 真实接口工程试跑：ReAct 开发集 27/30、冻结集 29/30；结构化多角色冻结集 27/30，其中引用错误、格式错误和一次接口超时均保留。开发阶段格式兼容及角色交接修复记录未删除。不同模式的调用量和校准过程不同，不宣布模式胜者，也不将这些结果当作匿名 v2 或真实推荐优化的成绩。

真实模型评测必须显式选择策略和模型，使用已有的 API 环境变量：

```bash
python -m reclab_agent.cli.evaluate diagnostics \
  --manifest tasks/diagnostic-v2/diagnostic-frozen.json --mode team --env-file .reclab/agent.env \
  --max-steps 8 --max-requests 240 \
  --output reports/diagnostic-frozen-team.json
```

评测为每个任务建立只含本任务证据的临时工具目录。gold label、参考决策及其他任务文件不能由模型工具读取。正确决策与可定位引用分别评分；失败任务留在分母。模型完成状态、响应、失败类型和 Trace 均保留，不以工具调用正确代替端到端研究成功。

连续三次模型服务错误后停止该批后续请求，剩余任务标为未运行并保留在分母；模型输出格式错误与服务不可用分别记录。

相同输出文件可恢复；manifest、prompt、模型、策略、价格或预算变化时拒绝复用。模型请求在发送前预留，失败也扣次数；已记录任务不重跑。原始 Trace 和角色状态保存在被 Git 忽略的 `reports/*-traces/` 下。

## 用量与费用

研究报告和评测报告记录请求次数、输入/缓存/输出 token，以及缺少 usage 的请求数。只有显式提供价格表时才估算费用；缺少 usage 不按免费处理，估算也不等于服务商账单。

```bash
python -m reclab_agent.cli.evaluate costs --trace traces/latest.jsonl
```

11 月 23 日的接口试跑共预留 335 次请求，硬限额为 400 次；该时间窗口结束后没有恢复其额度。11 月 24 日的正式匿名诊断和编码对照独立设置合计最多 640 次请求、一个小时截止时间，诊断分配 512 次、编码分配 128 次。失败与超时同样占请求额度。多个校准报告和恢复报告可能重叠，不能简单相加来推算账单。当前评测 CLI 关闭 SDK 内部重试，并设置单请求 40 秒超时；Agent 重试仍经过评测计数。

本轮每组诊断使用 `max_steps=8`、`max_requests=240`，四组另受宿主总额度约束。CLI 的 `--max-requests` 只约束当前输出批次，不自动为多个进程共享额度；本轮宿主的合计上限和实际预留数单独归档在汇总报告中。

## 验收证据与边界

- [最新验收汇总](../reports/non-gpu-acceptance-20261124.md)：匿名模型对照、编码、用量和交付边界；[第一批本地验收](../reports/non-gpu-acceptance-20261123.json)保留历史结果。
- [本地 Web HTTP 检查](../reports/local-web-http-20261123.json)：页面与静态资源、共享 CLI 状态、幂等提交、取消和事件流；没有启动训练。
- [BM25 检索结果](../reports/local-retrieval-bm25-20261123.json)：固定语料 Recall@5 为 1.0，MRR 为 0.9583；不代表开放领域答案质量。
- [匿名 v2 开发集参考检查](../reports/diagnostic-dev-v2-reference-20261123.json)与[冻结集参考检查](../reports/diagnostic-frozen-v2-reference-20261123.json)：各 30/30，均未调用模型。

最新回归为主测试 521 项（513 通过、8 项因环境条件跳过），策略测试 37 项通过；109 个 Python 文件与三个 Web 资源均已验证打包。前端事件刷新修复也通过 JavaScript 语法检查和浏览器实际交互复核。

[第一天失败审计与修复](../reports/day1-failure-audit-20261124.md)增加原始模型文本、答案结构校验和类型化错误。错误报告策略作为评测身份的一部分，旧输出文件不能直接覆盖。严格 JSON 规则未放宽，解释文字加 JSON 仍为格式失败；没有送出的本地预算拒绝不计入 API 使用量，已预留角色额度不回滚。旧冻结成绩保持不变，后续验收需要独立输出路径。

后续状态已更新：真实 SASRec 训练/评分、同一作业恢复核验与 GRU 迁移基线已有 [GPU 验收记录](../reports/gpu-acceptance-20261123.md)，包含失败与负结果。第二批已完成 Tavily 真实搜索/提取、CPU 编码及匿名诊断对照；开发集单/多角色为 28/30、27/30，冻结集为 26/30、23/30，多角色冻结成绩未达 80% 目标。本阶段预留 478/640 次请求，4 次缺少 usage，未核实账单。原生 SSH/tmux 不等同于云端 Docker/GPU 隔离，首次恢复仍需人工核验。

## 四类编码任务对照

沿用 `tools/`、`execution/` 和 `evaluation/` 的现有分层，不增加研究阶段。冻结文件为 `tasks/coding.json` 和 `tasks/coding-freeze.json`；生成器 `scripts/build_coding_evaluation.py` 拒绝覆盖既有冻结记录。

四类任务分别修复已看物品过滤、零填充均值、最近历史截取与左填充、负样本目录。每类两个独立 trial，各有 3 个公开检查和 24 个隐藏检查。模型只读取说明和 `src/solution.py`，可调用源码读取、补丁、语法检查及 `run_checks`；隐藏答案和参考实现不挂载进候选容器。评分由宿主确定性比较容器返回值，宿主不执行候选源码。

实际评测使用本地已缓存的不可变 Python 镜像 ID `sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1`。Linux/WSL 中须先具备 Docker 访问权限；评测不会下载镜像、扩大 socket 权限或转为宿主执行。容器使用 UID 65532、无网络、只读根和源码、128 MB/1 CPU/64 进程上限、10 秒超时、有限输出及独立清理 Watchdog；不转发宿主 API Key。真实边界与控制器强制退出检查见 [容器验收](../reports/coding-container-acceptance-20261124.json)。

[运行环境记录](../reports/coding-runtime-20261124.json)保存 Docker 29.1.3、容器 Python 3.12.15、linux/amd64 和完整仓库摘要。需要准备同一公开镜像时，可先手动执行 `docker pull public.ecr.aws/docker/library/python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1`，再核验本地 ID；本轮复用了缓存，没有下载。

```bash
python -m reclab_agent.cli.evaluate coding --image sha256:YOUR_VERIFIED_LOCAL_IMAGE_ID \
  --mode reference --repetitions 2 --output reports/my-coding-reference.json

python -m reclab_agent.cli.evaluate coding --image sha256:YOUR_VERIFIED_LOCAL_IMAGE_ID \
  --mode react --env-file .reclab/agent.env --repetitions 2 \
  --max-steps 8 --max-requests 64 --output reports/my-coding-react.json
```

`YOUR_VERIFIED_LOCAL_IMAGE_ID` 必须替换为本地核验的 64 位镜像 ID。`--mode team` 使用相同每 trial 最多 8 次模型请求、4 次补丁、3 次公开检查；整批最多 64 次模型请求，失败也计数。相同输出文件恢复时不重跑已记录 trial；中断中的 trial 标为失败，已预留次数继续占用。参考模式不使用模型或 API Key。

2026-11-24 实测固定参考配方 8/8、单 Agent 3/8、多角色 6/8。多角色包含一次接口超时和一次错误字段名造成的失败；单 Agent 包含未修改源码、用尽步数及隐藏检查失败。失败保留在分母；两种模式的实际请求、Token 和时延并不相等。这个小规模合成结果不代表真实推荐收益或一般仓库修复能力，不据此切换默认模式。

成功规则还要求 Agent 正常完成：单 Agent 另有一次源码已通过全部检查，但会话达到步数上限，仍计失败。部分任务说明未完整列出输入字段，错误键名也可能反映任务歧义；后续改进须另行冻结清晰的输入契约，不能回改本轮说明并覆盖成绩。

[工作台截图式演示](../reports/workbench-demo-20261124/README.md)展示已完成真实 GPU 会话，视频不发起训练；它是实际截图合成，未声称连续录屏。

## 真实 CPU 检索结果

已实际运行 `BAAI/bge-small-zh-v1.5` 与较小的 `cross-encoder/ms-marco-TinyBERT-L2-v2`。后者是本轮验收的轻量模型，与 CLI 默认 MiniLM 模型不同。CPU 模型均已成功加载，没有远程 GPU 调用。

| 模式 | Recall@5 | MRR | 无答案查询返回空结果比例 |
|---|---|---|---|
| BM25 | 1.0000 | 0.9583 | 1.0 |
| dense | 1.0000 | 0.9028 | 0.0 |
| hybrid | 1.0000 | 0.9583 | 0.0 |
| reranked | 0.6667 | 0.6111 | 0.0 |

语料只有 12 篇文档、12 个有答案查询与 1 个无答案查询。这次英文重排模型在中英混合语料上表现下降，向量模式也没有校准拒答阈值；不据此宣称检索普遍提升，默认继续采用 BM25。

[首次运行报告](../reports/local-retrieval-cpu-20261123.json)的首次查询耗时包含下载和模型初始化，不能当作稳定推理时延。另保存[固定模型提交、仅用本地缓存的复核](../reports/local-retrieval-cpu-cached-20261123.json)，用于区分下载与后续查询。
