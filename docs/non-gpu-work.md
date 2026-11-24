# 本地功能、评测与验收

更新日期：2026-11-24。沿用现有分层与四阶段流程，不新增编排框架。本页区分功能实现、离线测试和真实接口结果。

## 本地工作台与同一份状态

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

## 主研究者与独立审计

配置 `task.mode = "research_audit"`，或使用诊断 CLI `--agent research_audit`。
主研究者使用任务授权工具；审计者只读宿主冻结的证据包，先独立检查，再审查结论。
具体权限、预算、恢复和验收见 [独立审计说明](research-audit.md)。
新版本尚未完成真实 API 对照，不复用旧架构的评测成绩。

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

初稿中 dataset ID 含场景族名称，可能构成额外提示。v2 使用匿名标识并作为默认，两个版本的诊断标签均未因本次架构变更而修改。新协作模式使用新报告，不复用其他架构的成绩。


真实模型评测必须显式选择策略和模型，使用已有的 API 环境变量：

```bash
python -m reclab_agent.cli.evaluate diagnostics \
  --manifest tasks/diagnostic-v2/diagnostic-frozen.json --mode research_audit --env-file .reclab/agent.env \
  --max-steps 8 --max-requests 240 --output reports/research-audit-diagnostics.json
```

评测为每个任务建立只含本任务证据的临时工具目录。gold label、参考决策及其他任务文件不能由模型工具读取。正确决策与可定位引用分别评分；失败任务留在分母。模型完成状态、响应、失败类型和 Trace 均保留，不以工具调用正确代替端到端研究成功。

连续三次模型服务错误后停止该批后续请求，剩余任务标为未运行并保留在分母；模型输出格式错误与服务不可用分别记录。

相同输出文件可恢复；manifest、prompt、模型、策略、价格或预算变化时拒绝复用。模型请求在发送前预留，失败也扣次数；已记录任务不重跑。原始 Trace 和角色状态保存在被 Git 忽略的 `reports/*-traces/` 下。

## 用量与费用

研究报告和评测报告记录请求次数、输入/缓存/输出 token，以及缺少 usage 的请求数。只有显式提供价格表时才估算费用；缺少 usage 不按免费处理，估算也不等于服务商账单。

```bash
python -m reclab_agent.cli.evaluate costs --trace traces/latest.jsonl
```

当前评测 CLI 关闭 SDK 内部重试，并设置单请求 40 秒超时；Agent 重试仍经过评测计数。`--max-requests` 约束当前输出批次，不自动为多个评测进程共享额度。本次重构没有调用真实 API 或重置历史运行额度。

## 验收证据与边界

- [本地 Web HTTP 检查](../reports/local-web-http-20261123.json)：页面与静态资源、共享 CLI 状态、幂等提交、取消和事件流；没有启动训练。
- [BM25 检索结果](../reports/local-retrieval-bm25-20261123.json)：固定语料 Recall@5 为 1.0，MRR 为 0.9583；不代表开放领域答案质量。
- [匿名 v2 开发集参考检查](../reports/diagnostic-dev-v2-reference-20261123.json)与[冻结集参考检查](../reports/diagnostic-frozen-v2-reference-20261123.json)：各 30/30，均未调用模型。

当前回归结果见 [仓库优化记录](optimization-log.md)。独立审计的实际模型效果仍需单独验收。



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

`YOUR_VERIFIED_LOCAL_IMAGE_ID` 必须替换为本地核验的 64 位镜像 ID。`--mode research_audit` 使用相同每 trial 最多 8 次模型请求、4 次补丁、3 次公开检查；整批最多 64 次模型请求，失败也计数。相同输出文件恢复时不重跑已记录 trial；中断中的 trial 标为失败，已预留次数继续占用。参考模式不使用模型或 API Key。


成功规则还要求 Agent 正常完成：单 Agent 另有一次源码已通过全部检查，但会话达到步数上限，仍计失败。部分任务说明未完整列出输入字段，错误键名也可能反映任务歧义；后续改进须另行冻结清晰的输入契约，不能回改本轮说明并覆盖成绩。

[工作台截图式演示](../reports/workbench-demo-20261124/README.md)展示已完成真实 GPU 会话，视频不发起训练；它是实际截图合成，未声称连续录屏。

## 真实 CPU 检索结果

以下是旧版检索语料上的历史结果。本次将角色说明替换为研究与审计说明，`retrieval.json` 标记为 `research_audit_v1` 并更新语料哈希；诊断任务未改。下表不代表新语料成绩，也未重新调用文本模型。

已实际运行 `BAAI/bge-small-zh-v1.5` 与较小的 `cross-encoder/ms-marco-TinyBERT-L2-v2`。后者是历史验收的轻量模型，与 CLI 默认 MiniLM 模型不同。CPU 模型均已成功加载，没有远程 GPU 调用。

| 模式 | Recall@5 | MRR | 无答案查询返回空结果比例 |
|---|---|---|---|
| BM25 | 1.0000 | 0.9583 | 1.0 |
| dense | 1.0000 | 0.9028 | 0.0 |
| hybrid | 1.0000 | 0.9583 | 0.0 |
| reranked | 0.6667 | 0.6111 | 0.0 |

语料只有 12 篇文档、12 个有答案查询与 1 个无答案查询。这次英文重排模型在中英混合语料上表现下降，向量模式也没有校准拒答阈值；不据此宣称检索普遍提升，默认继续采用 BM25。

[首次运行报告](../reports/local-retrieval-cpu-20261123.json)的首次查询耗时包含下载和模型初始化，不能当作稳定推理时延。另保存[固定模型提交、仅用本地缓存的复核](../reports/local-retrieval-cpu-cached-20261123.json)，用于区分下载与后续查询。
