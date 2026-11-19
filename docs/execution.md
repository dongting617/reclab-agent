# Docker Worker：实现、故障恢复与验收边界

2026-10-20。第二周执行层的实现说明。本次已在本机 WSL2 Ubuntu 的真实 Linux Docker 完成受控 CPU 验收和 CLI 训练→预测→独立评分；见 [CPU 验收报告](../reports/week-02-cpu-acceptance-20261020.json)。Linux 全量188/188通过、0跳过，Windows178通过/10平台条件跳过/0失败。真实云端 Docker、GPU/PyTorch 与 Beauty 新训练仍待验；AutoDL 容器实例不支持 Docker，禁止改用宿主进程运行候选代码。假 Docker 测试只证明控制逻辑，本次 CPU fixture 也不证明 GPU 或推荐质量提升。

## 信任边界与使用位置

`DockerWorker` 运行在专用 Linux Docker daemon 所在主机；本次使用既有 WSL2 Ubuntu 实测 CPU。Windows 控制平面后续通过受控远程接口调用它；Worker 代码不自动 SSH、不安装 Docker、不改动 AutoDL，也不尝试嵌套容器或把宿主 Docker Socket 挂给候选。WSL Docker 由管理员显式准备，未开放 TCP daemon、未给普通用户添加 docker 组。

管理员预先登记 `WorkerPolicy`：一个不可变镜像摘要、来源 ID → 冻结源码目录、数据 ID → 冻结 public 目录、资源上限和是否开放 GPU。Agent 的 `JobSpec` 只有登记 ID、train/predict 阶段、配置名、种子、限额以及已完成训练作业 ID。它没有 command、image、mount、Docker 参数、SSH 或任意路径字段。

管理员、Docker daemon、预审镜像和 Worker 状态目录均属于可信计算基；状态目录应使用支持 SQLite 与 OS 文件锁语义的本地文件系统。控制器本身不导入候选模块、不反序列化候选权重，不用 `eval` 或 pickle 收集预测。Docker 权限应仅授予 Worker；这不是把不受信任代码直接交给有 Docker 权限的通用终端。

## 已实现接口

```python
from reclab_agent.execution import DockerWorker, InputTree, JobSpec, WorkerPolicy

source = InputTree.freeze("/srv/reclab/snapshots/approved-v1", "source")
public = InputTree.freeze("/srv/reclab/datasets/beauty/public", "public_data")
policy = WorkerPolicy(
    image="registry.example/reclab@sha256:<管理员核实的真实64位摘要>",
    sources={"b1-v1": source}, datasets={"beauty-v1": public},
)
worker = DockerWorker("/srv/reclab/worker-state", policy)
worker.capabilities()  # 缺 daemon、seccomp、cgroup 能力或镜像时明确抛 CapabilityError。
job = worker.submit(JobSpec("b1-v1", "beauty-v1"))
worker.poll(job["job_id"])
worker.cancel(job["job_id"])
worker.recover()
```

上面的镜像字符串是说明占位符，必须替换真实 digest，不能直接运行。`InputTree.freeze` 对实际普通文件逐一计算 SHA256，再对文件清单计算内容身份；它与 `Workspace.snapshot` 的文本内容哈希是两个不同契约，不能互换填写。登记后再次提交会重验文件身份，变动后需重新审查登记。

源码只允许 `src/`、`configs/` 下 Python/JSON/YAML 文本，必须有 `src/train.py`、`src/predict.py`；配置文件为 `configs/<config>.json`。数据目录只能包含 `manifest.json`、`train.json`、`valid.json`，必须直接登记 public 子目录。传整个 DatasetBundle、附加 test.json、链接或额外文件会失败。输入注册和输出收集都拒绝符号链接、Windows reparse point、硬链接和特殊文件。

## 实际容器命令与训练/推理协议

容器固定使用镜像中的 `/usr/local/bin/python -E -s -B`，不执行 shell。训练脚本接收：

```text
/workspace/src/train.py --data /data --output /output --config /workspace/configs/default.json
```

脚本从 `RECLAB_SEED` 读取种子。候选可修改模型逻辑，但无法通过 JobSpec 改写执行器命令。训练退出码为 0 且存在合法普通输出文件时生成输出清单/哈希回执；它只表示执行产物，不表示模型质量合格。

推理是另一个作业和另一个容器，必须引用相同登记源码、数据、配置和种子的成功训练作业。训练产物重验回执后只读挂载 `/model`，推理自己的 `/output` 为新目录：

```text
/workspace/src/predict.py --data /data --output /output --config /workspace/configs/default.json --model /model --split valid
```

`SourceBackend.prediction_command(split="valid", config_name="default")` 与此入口参数保持一致，并通过合同测试核对；实际模型适配器需要接受 `--config` 与只读 `--model`。最终 test 尚未开放；需另做冻结后的可信推理协议。

`collect_predictions(job_id)` 只读取 `predictions.json`，接受 `{用户ID: [正整数itemID, ...]}`，拒绝重复 JSON key、NaN/Infinity、布尔 ID、异常结构、过大文件及完成后篡改。JSON 最大 32 MB，单用户最多 1000 项；用户覆盖、重复排名、合法目录、屏蔽 seen 与指标由外部固定评分器继续核验。只收集排名，不采纳候选自报指标。

每个容器显式设置：

- UID/GID `65532:65532`，非 root；`--cap-drop ALL`、`no-new-privileges`，要求 daemon 报告 seccomp。
- `--network none`、只读根目录、私有 IPC；不挂凭据、评分器、测试标签或 Docker Socket。
- 源码 `/workspace` 和 public 数据 `/data` 只读；仅该作业 `/output` 可写。推理额外只读 `/model`。
- CPU、内存、相同大小 memory-swap、PID 上限、nofile 上限、64 MB 的 noexec/nosuid/nodev 临时目录。
- 日志使用 local driver，`max-size=1m`、`max-file=1`、显式 `compress=false`；读取最多 200 行/64 KB。压缩默认值与单文件配置的冲突由本次真实失败定位并回归。
- 镜像只接受 `repository@sha256:...`，要求本机已具备该 RepoDigest，`--pull never`。镜像声明额外 VOLUME 会被拒绝。
- 默认并发 1。GPU 仅能由管理员 Policy 开启，并要求 NVIDIA runtime，固定开放 device 0；不允许 Agent 自选设备。

Docker CPU/内存/PID 支持任一缺失就失败，不把 daemon 警告后的降级当作限额成功。GPU 设备开放不等于显存配额；当前没有宣称提供 GPU 多租户隔离或显存硬限额。

## 幂等、恢复与停止为什么要这样设计

作业键绑定 `JobSpec + 源码身份 + public 数据身份 + WorkerPolicy身份`；稳定 `job_id`、容器名和三个精确 ownership label 都由键产生。SQLite 唯一键和 `BEGIN IMMEDIATE` 负责抢占，60 秒 lease 保存恢复意图；每个作业另有贯穿 launch/observe/stop 的 OS advisory lock（Linux flock、Windows 文件范围锁），不能只靠会过期的 lease 保护 Docker 副作用。活进程即使暂停超过 lease，也仍持有 OS 锁，其他 Worker 不会同时接管；进程崩溃后 OS 自动释放锁。Docker 的确定容器名提供跨数据库/daemon 故障窗口中的另一重去重。锁文件不自动删除。

状态包括 `queued → launching → running → completed/failed/timed_out/cancelled`。连接结果不确定时保留 `reconciling`；停止尚未确认时保持 `stopping`，不能提前宣布取消成功。已启动容器意外消失标记 `lost`，不会自动再训练一遍。终态不可改写。

例子：Docker 已完成 create，控制器还没记账便断线。恢复时先 inspect 确定名称及 ownership，找到 created 容器再 start；不会再 create 第二个。若 start 后断线，inspect 得到 running，直接接管观察，不再 start。未知同名容器不会被启动、停止或删除；本实现完全没有自动 `docker rm`。

取消与创建并发时，先持久化停止意图；活跃 launcher 在创建后、启动前以及启动后检查它。已创建但未启动的容器可以安全记为取消；已运行容器使用 KILL，并 inspect 确认退出。kill 失败时保持 stopping。重启会继续核验和停止，不能把 SSH 断开误当成训练已停止。

**时限的实际边界。** `deadline` 是在 launch 前写入 SQLite 的墙钟截止时间，包含创建耗时；`poll/recover` 会停止过期容器。另提供独立的 `DockerDeadlineWatchdog`，由管理员在 daemon 主机上启动为第二个进程，不能与 Worker 共用一个易被暂停的线程。客户端或 SSH 断开不会自动停止已独立运行的监督进程。只执行一次 CLI submit、一次 watchdog `once` 后退出，都不会留下常驻监督。

## 独立 watchdog：Worker 暂停持锁时仍能停

`python -m reclab_agent.watchdog` 仅开放 `once` 和 `serve`。它读取既有私有 Worker SQLite 与同一份已审核 Policy，不接受候选命令、任意容器名或 Docker 参数；不会创建、启动、删除容器，也不会在宿主执行候选。代码没有自动安装 systemd、计划任务或后台服务，进程守护由管理员显式配置。

```bash
# 两个独立终端/服务；使用同一真实私有状态目录与同一 Policy。
python -m reclab_agent.worker --state /srv/reclab/worker-state --policy /srv/reclab/policy.json serve --interval 1
python -m reclab_agent.watchdog --state /srv/reclab/worker-state --policy /srv/reclab/policy.json serve --interval 1

# 只做一轮检查，不创建或启动任何作业。
python -m reclab_agent.watchdog --state /srv/reclab/worker-state --policy /srv/reclab/policy.json once
```

首次过期观察先 inspect 确定名称，检查 worker/job/spec 三个精确 label，并区分“现在已过期”和“进程确实运行超过截止时间”。若容器已退出且 `FinishedAt<=deadline`，不写超时意图，交 Worker 核验产物后决定 completed/failed；缺失完成时间时同样交 Worker 核验，不凭本轮观察时间覆盖已完成结果。首次 inspect 断连时不提前判定超时。

对确实仍运行的过期作业，先在私有 `stop-journal/` 写独立、fsync 的不可变停止标记，再尽力写 `docker_stop_intents`。Watchdog 对 SQLite 写操作只等 50 ms；旧 Worker 暂停在 SQLite 写事务中时，独立 journal 不受该写锁阻塞，watchdog 仍向已核实的不可变容器 ID 发 KILL，SQLite 记账标为 pending 并于后续重试。Worker 恢复后的停止检查也读取 journal，不能因 SQLite 尚未补账而启动候选。Watchdog 启动只读取既有 schema/owner，不执行建表/INSERT；旧版本数据库需先由 Worker 显式迁移。

只有 ownership 和不可变容器 ID 都匹配，才向该 ID 发 KILL；kill 返回 0 也不算确认成功，必须重新 inspect 观察到 `exited/dead` 及整数退出码。Worker 自己的停止也在 fresh inspect 后固定按 ID 发 KILL，避免确定名称被其他容器替换后误停外部作业。未知 labels、身份不符、无法 inspect、kill 后仍 running 都不会宣布停止成功。

这里有两种锁语义。Worker 的 OS job lock 继续保护 create/start，禁止同一作业被两个 launcher 重复启动。Watchdog 的 KILL 刻意不等待这个锁：如果旧 Worker 被 SIGSTOP，它会一直持锁，但不应阻止另一个可信监督进程停止已运行容器。只有落作业终态时才尝试非阻塞取锁；锁被占用时可返回 `stop_confirmed=true, launcher_busy=true, status=stopping`，表示本次实际观察到退出，但 launcher 仍可能恢复执行，终态记账暂缓。

不能把 `created` 或 `missing` 当作永久安全。旧 launcher 可能正停在最后一次停止检查与 `docker start` 之间，或 Docker create 的响应丢失后才完成创建。Watchdog 会保留意图并在每一轮继续检查，包括已经写入终态的作业；迟到的 start 被发现后仍会 KILL。合规 Worker 在 create 前、create 后及 start 后检查同一持久意图，恢复时不再启动新工作。意图先到先保留，先取消后过期不会把取消原因改写成超时。

**没有不可绕过的全局硬时限保证。** 实际停止延迟包括扫描间隔、前面作业的 Docker RPC 耗时以及 daemon 响应时间。SQLite 既有提交记录可读、仅写事务被占用时可以紧急停止；数据库排他锁导致既有作业身份不可读、文件系统无法访问、watchdog 自身宕机、Docker daemon 不响应、宿主暂停或墙钟大幅回拨时，仍无法保证准点停止；管理员必须监督 Worker 与 watchdog 并对异常报警。Watchdog 和 Worker 不能互相代替存活性监控。本实现不将这些情况描述为内核级 wall-time 配额。后台监管只授予管理员，不开放给候选工作区。

**存储边界。** Worker 状态根目录在 Linux 设为 0700，单个输出目录允许容器 UID 写入。收集预算默认 512 MB、最多 1024 文件，完成后冻结普通文件清单。它是收集限额，不是写入时的磁盘 quota；管理员还需对专用输出文件系统配置容量/配额。不要把事后文件大小检查说成实时磁盘隔离。

## 管理员 CLI 与镜像准备

`python -m reclab_agent.worker --state <私有状态目录> --policy <已审核policy.json>` 后接：

| 命令 | 行为 |
|---|---|
| `doctor` | 查询本地 daemon/cgroup/seccomp/摘要镜像；不拉镜像、不跑候选 |
| `serve --interval 1` | 常驻恢复和超时监督；应由管理员服务保持运行 |
| `submit --spec job.json` | 提交固定 JobSpec，返回作业身份和当前状态 |
| `status JOB_ID` | poll/协调已有作业；queued 作业有资源时可被调度，DB终态仍是历史记录，不单独证明容器当前未运行 |
| `cancel JOB_ID` | 持久化停止意图并确认进程退出 |
| `recover` | 单次扫描已有作业，包括断线/超时处理 |
| `collect JOB_ID` | 返回普通 JSON 排名，供独立 grader 使用 |
| `logs JOB_ID` | 有界读取已核验 ownership 的容器日志 |

CLI 输出 `{ok, result}` 或 `{ok:false,error,detail}`；能力错误退出码 2。policy.json 仅由管理员编辑，含 image、sources、datasets、可选 max_limits/gpu/max_output_bytes/max_concurrent_jobs。sources/datasets 每项为 `{directory, sha256}`；不要把这个文件开放给 Agent 工作区。

Watchdog 的 `once/serve` 输出每个停止意图的 `stop_confirmed`、`container_state`、`launcher_busy` 与诊断。`stop_confirmed` 是该轮实际观察值，不是“以后再也不会运行”的承诺；created/missing/连接未知返回 false，持久意图继续受监督。Policy/输入身份改变时拒绝操作，需要管理员恢复原登记身份再接管。

`docker/Dockerfile` 是可选的最小 CPU 合同镜像说明，不包含 PyTorch；自定义构建时须由管理员 pin `BASE_IMAGE`、保存 RepoDigest 并预装。本次没有构建或发布自有镜像，而是直接使用 Docker 官方 ECR Python 发布镜像的真实 digest 完成 CPU 验收，实际 repository/digest 保存在验收报告，不能改写成另一个 registry 的身份。GPU/PyTorch 镜像、依赖锁定与真实模型训练还需另行准备，不能把这份 Dockerfile 或 CPU 镜像当成已具备 GPU 训练环境。

## 本次测试与尚未验收

`tests/test_execution.py` 使用假 Docker runner 注入 create/start 后断线、并发提交、lease 恢复、未知 ownership、取消创建竞态、未确认停止、超时、OOM、已启动容器丢失、输出篡改/硬链接与异常 JSON。runner 只保存参数、模拟 daemon 状态，绝不运行候选；这些是控制逻辑和安全合同测试。

`tests/test_watchdog.py` 进一步测试独立停止进程合同：过期与取消的持久意图、kill 后断线、未知 labels/缺失不可变 ID、created/missing 后迟到 start、终态后重新运行、旧 launcher 暂停在 start 前的竞态、实际跨进程 OS job lock，以及 Linux SIGSTOP 注入。还覆盖首次晚观察按时完成的容器、真实 SQLite `BEGIN IMMEDIATE` 写锁期间新 watchdog 启动及紧急停止、畸形 State 不影响其他作业，以及 Worker 按 ID 停止时同名 foreign 容器保持运行。跨进程测试使用真文件锁，但 Docker 行为仍为 FakeDocker；它们证明停止控制逻辑不受 launcher job lock 阻断，不是实际容器隔离或 daemon 可用性的证明。

真实测试 `RealDockerSmokeTests` 仅在 Linux、本地 Docker CLI 和明确配置 `RECLAB_DOCKER_TEST_IMAGE=<已预装的真实摘要>` 时启用，条件不满足自动 skip；启用后能力或执行失败就是真失败，不再降级 skip。它运行固定审查过的 stdlib 训练/预测脚本，检查非 root、只读输入/根目录、无 Socket/SSH 私钥/测试标签路径、禁网及模型只读，再交外部评分器。它依然是 synthetic smoke，不是 Beauty/B1 复现或真实推荐优化证据。

新增 `tests/test_real_execution.py` 在明确 opt-in 后运行七类真实 CPU 验收，覆盖实际输入/输出边界、重复提交与 Worker 重建、取消、监督超时、CPU/PID/cgroup、持锁时独立停止及有界 OOM。本次七类均在 WSL2 Linux daemon 通过；完整 Linux 回归188/188通过、0跳过。Windows 同一套发现188项，178通过、10项因 Linux Docker/平台限制跳过、0失败。不能把188项都称为真实容器用例；其中 FakeDocker 仍只验证控制逻辑。

受控 `reclab-research` CLI 也已实跑两个固定作业，completed/exit 0、seed42、12个合成用户，由外部 grader 重算四项零指标，保持 `fixture_only` 与 `llm_connected=false`。入口、独立 Watchdog 操作顺序和产物见 [CLI 教程](week02-demo.md)，全部身份与测试范围见 [CPU 验收报告](../reports/week-02-cpu-acceptance-20261020.json)。

第二周 CPU 工程已通过；原完成标准中的真实云端容器仍未勾选，GPU 与 Beauty/B1 容器内新训练闭环未验收，个人学习待自测。第一周真实 Beauty 冻结基线是否复现另见其独立报告，不能由本次合成 CPU 结果推导。输出仅有收集预算、没有实时磁盘 quota，监督进程/daemon 故障仍需如实报告。

学习练习：create 返回超时后立刻创建一个随机名新容器，会有什么后果？原 create 可能已成功，随机名绕过去重会重复训练。应查询确定名称和 ownership，并保留不确定状态直到核验。另一个问题：文件清单 SHA256 一致，能否证明模型有效？不能；它只绑定产物，质量还要由冻结数据协议和外部 grader 评价。
