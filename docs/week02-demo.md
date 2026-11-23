# 第2周受控 Docker 示例：从准备到独立评分

本示例补齐 `reclab-research` 命令入口。它演示固定 Popularity 算法的训练、预测、产物收集和独立评分，数据为 `fixture_only` 合成样本；没有接入 LLM，不代表真实推荐模型优化效果。

2026-10-20 已在本机 WSL2 Ubuntu 的真实 Docker 实跑：train、predict 两个作业均 completed/exit0，seed42，对12个合成验证用户完成外部独立评分；Recall/NDCG@10/20四项均为0，符合fixture设计。镜像、源码、数据、预测与评分哈希和作业回执见 [CPU 验收报告](../reports/week-02-cpu-acceptance-20261020.json)，原始CLI记录为 `.reclab/week02-demo-run-20261020.json`。此结论限 CPU：真实云端/GPU/Beauty新训练仍待验，学习掌握待自测。

## 1. 先理解两个执行层

控制器运行可信的 `reclab_agent` 代码，负责登记身份、提交作业和评分。`src/train.py`、`src/predict.py` 是候选代码，只能在 Docker 中执行。Python 虚拟环境解决包依赖，Docker 的挂载、网络、身份与资源限制决定候选程序能访问什么。没有 Docker 时命令返回结构化错误，绝不退回宿主运行训练。

`prepare` 只生成固定源码并进行 AST/JSON 语法检查。它通过 `Workspace.snapshot` 建立独立副本，再通过 `InputTree.freeze` 记录源码目录和公共数据目录的身份。工作副本修改不会改变快照，修改快照会导致 `run` 拒绝执行。

## 2. 在同一个 Linux Docker 主机中运行

从仓库根目录运行下面命令；本周 stdlib 演示无需安装项目依赖。后续接入 LLM SDK 时另建 Linux 虚拟环境，Windows 的 `.venv` 不能直接给 Linux 使用。WSL2 Ubuntu 可以作为本机 CPU Docker 主机；不要在 Windows Python 中准备后直接把 Windows 路径交给远程 Docker。Ubuntu 命令使用 `python3`。

本机已安装 Ubuntu 官方 `docker.io`，真实验收使用 Docker Engine 29.1.3 与 cgroup v2。Docker/containerd 的开机自动启动已关闭，按需运行；本次临时监督和 WSL 保活进程已结束。WSL 空闲退出后，下次需先在 WSL 启动 daemon，再启动项目监督，不应把 Docker 未运行当作需要重装依赖。可在 PowerShell 输入第一行，后续命令在该 Linux 终端执行并保持终端开启：

```powershell
wsl --distribution Ubuntu --user root
```

```bash
systemctl start docker.service
cd /mnt/e/DongTing/简历/reclab-agent
docker info
```

这里的 root 仅用于可信控制器管理本机 daemon；候选容器仍为 UID/GID 65532。已有实跑目录为 `/var/lib/reclab-agent-week02-demo-20261020`，镜像为验收报告中记录的 ECR RepoDigest；重新准备演示时使用新的目录名，不覆盖证据。

```bash
python3 -m reclab_agent.cli.research demo prepare --root "$PWD/.reclab/week02-demo-001"
```

目标目录必须不存在，避免覆盖已有实验。准备结果写入 `demo.json`，包括数据哈希、源码内容哈希、源码目录哈希、公共数据目录哈希、固定种子 42 和每个作业 30 秒限制。

镜像由管理员在可信准备步骤预先拉取，解析真实 digest；此步骤可以联网，后续候选容器禁网：

```bash
docker pull python:3.12-slim-bookworm
IMAGE=$(docker image inspect python:3.12-slim-bookworm --format '{{index .RepoDigests 0}}')
python3 -m reclab_agent.cli.research demo doctor --root "$PWD/.reclab/week02-demo-001" --image "$IMAGE"
```

Worker 使用 digest 并设置 `--pull never`；示例不会自动拉取镜像。`doctor` 仅检查能力，真正运行的容器才提供执行证据。若普通 Linux 用户无 Docker 权限，管理员可以在既有 WSL 发行版中以 root 运行可信控制器；不需要把用户加入 docker 组，不需要开放 TCP daemon。候选容器本身使用 UID 65532。

`doctor` 会建立 Worker 数据库并生成 `worker-policy.json`。在执行 run 前，管理员先在另一个终端显式启动独立监督（Ctrl+C 停止监督进程，不会删除作业）：

```bash
python3 -m reclab_agent.execution.watchdog --state "$PWD/.reclab/week02-demo-001/worker" --policy "$PWD/.reclab/week02-demo-001/worker-policy.json" serve --interval 0.2
```

保持该进程运行，在原终端中继续：

```bash
python3 -m reclab_agent.cli.research demo run --root "$PWD/.reclab/week02-demo-001" --image "$IMAGE"
python3 -m reclab_agent.cli.research demo status --root "$PWD/.reclab/week02-demo-001"
```

这不是自动后台服务；CLI 不检查常驻 Watchdog 存在与否。为验收中断/挂起场景，应提前保持该独立进程运行。本次实际使用官方 ECR Python 镜像，保留其真实 `public.ecr.aws/docker/library/python@sha256:...` 身份；上面的 Docker Hub 准备命令是可选官方来源，不能拿另一个 registry 的名字替换已记录 digest。

## 3. 顺序与预算

只允许两个固定作业：train 成功后才提交 predict；并发上限为 1。两者显式读取 `RECLAB_SEED=42` 并设置 Python 随机种子。训练仅统计 `train.json` 的商品计数；预测读取前一训练作业只读模型目录，按计数降序、商品 ID 升序排序，遮蔽历史商品，为每个验证用户输出 Top20。

同一根目录绑定同一个镜像和固定配置；重试复用 Worker 的确定性作业 ID。更换镜像必须准备新的根目录。公共挂载只有 `dataset/public`，测试标签位于独立 `dataset/private`；不得把数据集父目录注册成公共输入。

公开 `demo run` 由父控制器监督可信子进程：总执行等待最多 120 秒，失败或中断后另给停止确认进程最多 30 秒。超时时终止的是控制器进程；Docker 作业需要通过持久化作业 ID 发起取消，并用独立 stop-only watchdog 检查退出。若 Docker 不可达、停止未确认或停止确认本身超时，返回 `stop_confirmed: false`，保留状态，要求管理员继续监督，不能声称作业已经停止。独立 watchdog 的部署和故障验收见第二周执行环境文档。

停止确认对所有作业都重新检查容器，包含数据库里已经 `completed`/`failed` 的作业；终态记录不能排除容器被重新启动。检查按所有权标签识别容器，发信号与退出复核使用不可变容器 ID。返回的 `observed_at_unix` 和 `confirmation_scope: observation_only` 说明它只确认该观察时点；即使已经确认退出，也需要继续独立 watchdog 以捕捉迟到的 create/start。`demo status` 读取缓存文件和数据库，明确返回 `live_verified: false`，不能作为当前容器已经停止的证据。

## 4. 如何核对结果

- `run.json`：两次作业 ID、状态、产物 receipt、镜像和输入身份。
- `predictions.json`：Worker 校验并收集的预测。
- `report.json`：宿主可信 `DatasetBundle.grade` 对验证集独立评分。
- `run.json` 中 `predictions_sha256`、`report_sha256`：对应规范化 JSON 内容哈希。
- `last-error.json`：失败原因与停止确认结果。
- `cancel-observation.json`：每个作业的退出观察时点、容器状态和是否需要持续监督。

合成样本设计用于接口验证，Popularity 得分为零也不能解释为真实模型质量差。真实 Beauty/B1 的结果仍应引用第一周冻结基线证据。

## 5. 练习与答案

1. 为什么 `prepare` 通过不代表训练能运行？答案：它只检查语法和输入身份，没有执行候选程序，也没有验证 Docker 能力。
2. 为什么同一源码存在两个哈希？答案：`Workspace` 哈希绑定路径与文本，`InputTree` 哈希绑定路径与文件字节哈希；应分别比较相同类型的身份。
3. 为什么不能直接相信 `run.json` 中 `cancelled`？答案：数据库状态不等于所有异步 Docker 命令已经结束；独立监督需要用新的容器检查确认退出，处理迟到的 create/start。
4. 如何保证测试标签没有暴露？答案：只注册公共子目录，检查 Worker 实际挂载，并用越权读取用例实测；文件夹命名本身不能提供隔离证据。
5. 怎么把这个 Demo 扩展成真实 Agent？答案：后续用受控工具接入计划、补丁和模型决策；保留相同的快照、执行、产物和独立评分边界，另做 Agent 任务评测。

本页兼作使用教程与本次实跑记录；已完成的 CPU 范围以验收报告为准，不能外推到云端/GPU或真实模型优化。模拟 Worker 的单元测试只证明编排行为。原理、推演题和代码阅读路径见 [Worker 运行时实战课](weeks/learning/week-02-worker-runtime.md)。
