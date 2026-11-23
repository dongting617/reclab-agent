# 第2周实战课：让实验能恢复，也能确实停止

董听，你已经熟悉在 PyTorch 中写训练循环。这一周新增的问题是：当训练由 Agent 自动提交，谁能证明它只运行了一次、超时后真的停止、输出确实来自这次实验？这不是再加一个 `train.py` 就能解决的问题，需要把实验做成有身份、有状态、有边界的作业。

本课对应当前 RecLab 的 Worker、独立 Watchdog 和受控 CLI。代码阅读与推演练习可以先完成；真实验收以保存的报告为准。CPU 容器通过不能推导 GPU 容器通过，FakeDocker 测试通过也不能推导真实 Docker 已可用。

## 1. 连接断开，与训练停止，是两件不同的事

先区分四个参与者：CLI 发出请求，Worker 负责作业状态与启动，Docker daemon 管理容器，容器内的 Python 执行训练。CLI 或 SSH 连接断开，并不自动撤销 daemon 已接受的操作。

假设 Worker 已发送 `start`，daemon 已启动容器，但成功响应尚未返回时网络中断。此时客户端只能知道“没有收到确认”，不能判断“没有启动”。如果恢复时直接再跑一遍训练，可能重复消耗资源；如果直接记为失败，也可能留着一个仍在训练的容器。

RecLab 为每个实验建立确定性身份：输入源码目录身份、公共数据目录身份、镜像 digest、配置、seed 和资源策略共同决定 Job ID。同一请求重试时先找原作业和原容器，再根据真实状态恢复。这里的幂等不是“相同请求只来一次”，而是“相同请求来了多次，仍指向同一个实验”。

你可以把它与 checkpoint 做对照：模型 checkpoint 保存训练参数；Worker 数据库保存哪个作业在运行、对应哪个容器、deadline 和产物身份。只有 checkpoint，没有作业身份，仍可能重复启动两条训练；只有作业身份，也不自动获得模型训练步数的恢复能力。

阅读 `reclab_agent/execution/worker.py` 的 `JobSpec`、`DockerWorker._key`、`submit`、`poll`、`recover`。注意 `recover` 的职责是恢复管理关系，不是保证从上一次 optimizer step 接着训练。

## 2. 租约不能证明旧进程已经死亡

考虑一段时间线：A 领取作业，租约到 60 秒；A 在检查容器后被暂停；第 61 秒 B 发现租约过期。B 能否推断 A 已死亡？不能。进程暂停、调度延迟和慢 I/O 都可能让一个活进程超过租约。

如果 B 启动并完成实验，A 随后拿着旧的 `created` 状态恢复，再调用 `start`，同一个已经退出的容器可能重新执行。确定性容器名可以防止多建一个同名容器，却不能阻止旧容器被再次启动。

现在同一主机上的同一 Job 使用 OS 文件锁，覆盖检查与 Docker 副作用。Linux 分支使用 `fcntl.flock`，Windows 分支使用相应文件锁。进程暂停时锁仍在，另一 Worker 不能因为租约过期就绕过锁；进程崩溃、文件描述符关闭后，锁由 OS 释放。锁文件不随意删除，否则两个进程可能锁住不同文件，却都以为自己独占同一个作业。

这解决了重复执行，但产生另一个问题：持锁的 A 被暂停，B 无法进入锁内，谁来及时停止已经运行的容器？如果所有取消逻辑都必须先取得同一把锁，互斥做对了，停止能力却被卡住。

阅读 `DockerWorker._job_lock`。用纸写出“进程仍活着”“租约有效”“持有 OS 锁”三个布尔量，理解它们之间没有简单的等价关系。[Python 文件锁资料](https://docs.python.org/3/library/fcntl.html#fcntl.flock)

## 3. 独立 Watchdog 为什么只能停止，不能启动

独立 Watchdog 是另一个可信进程，专门监督到期和停止意图。它不能提交、创建或启动候选容器，避免监督者变成第二个不受控的调度器。

停止分成三步：先持久化停止意图，再识别属于本 Worker 和 Job 的容器并发出停止信号，最后重新检查退出。停止意图像一张保留的“禁止继续启动”通知，恢复的 launcher 在启动前后都要检查它。即使 SQLite 暂时被写锁占住，私有文件系统中的 `stop-journal` 也要保留这条意图；数据库稍后再补记状态。

识别容器不能只看名字。名字可能被删除后重新使用，所以先核对 Worker、Job、规格哈希等所有权标签，再使用不可变容器 ID 发信号和复核，避免两次检查之间名称已经指向另一对象。

Watchdog 的 KILL 不等待 launcher 的 OS 作业锁，因此 A 即使仍持锁，监督者也可以停止已经运行的、确认归属本作业的容器。数据库终态登记可以稍后完成。这时可能出现：容器已退出、数据库仍是 `stopping`、报告标记 `launcher_busy`。这不是应当抹平的矛盾，而是两个事实：停止已经观测到，状态登记仍待完成。

同样，数据库 `completed` 只是一条历史记录。取消时也要重新检查容器，不能因为这条记录就声称现在一定没有进程。`stop_confirmed: true` 绑定某个观察时点；为处理迟到的 create/start，停止意图与独立监督仍需保留。只运行一次 `tick()`，不能提供未来持续监督。

阅读顺序：`execution.py` 的 `_request_stop`、`_journal_stop`，再看 `watchdog.py` 的 `_owned_container`、`_stop`、`tick`。关注 `stop_confirmed`、`accounting_pending` 和 `launcher_busy` 各自表达的事实。

## 4. 有超时参数，不等于有停止保证

每个候选作业有资源策略，例如 CPU 1、内存 256 MiB、PID 32、运行时间 30 秒。CPU、内存和进程数由 Docker/cgroup 执行；运行时间需要有可信监督者在 deadline 后采取动作。Worker 和 Watchdog 都停摆时，仅在 JSON 里写一个 deadline 不会自行产生停止信号。

受控 `demo run` 另有外层等待预算：可信执行控制器最多等待 120 秒，失败后取消确认进程另有 30 秒预算。这个外层超时结束的是可信子进程的等待，不等于 Docker 内候选自动死亡，所以还要用持久化 Job ID 取消、确认退出，并保留独立 Watchdog。进程创建、OS 调度和状态文件 I/O 也有耗时；这些数字不是实时系统的调度承诺。[Python 子进程超时语义](https://docs.python.org/3/library/subprocess.html#subprocess.run)

内存例子也需要分清：退出码 137 经常对应 SIGKILL，但取消和 OOM 都可能出现这个码。判断内存超限时，应结合 Docker 的 `State.OOMKilled`、实际内存限制和受控测试行为，不能只看 137。`--cpus=.25` 表示 CPU 时间配额，不表示把进程固定到某个四分之一核；真实验收还应观察节流计数，而不只检查参数字符串。

## 5. 为什么 FakeDocker 全绿，真实容器仍可能失败

FakeDocker 能在可控时刻模拟“daemon 已接受 create，但响应丢失”“launcher 持锁暂停”“KILL 返回错误”等路径。它适合重复推演控制逻辑，不会真的创建 namespace、施加 cgroup 或加载日志驱动，因此不能证明这些机制生效。

本轮真实 CPU 容器验收暴露了一个具体问题：Worker 为限制日志使用 `local` 驱动和 `max-file=1`，但该驱动默认启用压缩，当前 daemon 拒绝这个组合。结果容器未正常运行、日志收集也失败；FakeDocker 没实现这一驱动的参数组合校验，所以此前没有发现。

修复是在每个候选容器的参数中显式增加 `--log-opt compress=false`，保留单文件轮转配置，不需要修改全局 daemon 设置。Docker 官方文档列出压缩默认值和可配置项；Moby 源码说明压缩与单文件配置的校验约束。[local 日志驱动](https://docs.docker.com/engine/logging/drivers/local/)、[Moby 配置校验源码](https://github.com/moby/moby/blob/master/daemon/logger/local/config.go)

这个案例的排查顺序值得记住：训练无输出时，先检查容器是否真正运行过，再区分启动错误与训练错误；不要直接改模型代码。保存 create/start 结果、容器 inspect 和日志失败本身，修复后重新运行受影响的真实案例。日志收集失败不能被吞掉后记成验收成功。

`tests/test_real_execution.py` 的七类真实用例分别验证训练/预测边界与独立评分、重复提交及 Worker 重建、取消、运行中监督超时、CPU/PID/cgroup、持锁时独立停止、有界 OOM。请分别阅读每个用例怎样取得证据。真实测试本身也只证明这些受控条件下的行为；它不等于 CLI 演示已经完成，更不等于 GPU 隔离通过。

## 6. 跟着代码阅读与演示，不急着勾“已掌握”

建议先读本课的前三节并画一遍时间线，再按以下路径阅读：

1. `reclab_agent/execution/worker.py`：JobSpec → WorkerPolicy → InputTree → 作业身份 → 锁与状态 → 挂载参数 → 产物收集。
2. `reclab_agent/execution/watchdog.py`：只停止的能力边界 → 所有权和不可变 ID → 停止意图 → 确认与记账。
3. `tests/test_execution.py`、`tests/test_watchdog.py`：挑选一项断线、一项持锁、一项陌生 ownership，预测断言再看实现。
4. `tests/test_real_execution.py`：找到与模拟用例对应的真实容器证据，区分 inspect 参数与容器实际行为。
5. `docs/week02-demo.md`、`reclab_agent/examples/demo.py`、`research_cli.py`：理解 prepare → doctor → 独立监督 → run → 独立评分。

演示开始前，先在同一 Linux Docker 主机上完成 prepare 和 doctor。doctor 建立 Worker DB 与 `worker-policy.json`；随后在独立终端显式启动：

```bash
python -m reclab_agent.execution.watchdog --state <demo根目录>/worker --policy <demo根目录>/worker-policy.json serve --interval 0.2
```

保持这个进程运行，再从另一个终端执行 `demo run --root <demo根目录> --image <管理员预拉取的真实RepoDigest>`。这是操作示意，路径和 digest 必须替换为已核对值；本课没有替你执行该命令。CLI 不会自动安装后台服务，也不会把缺少 Docker 的情况退回宿主训练。

## 7. 三道推演题与参考答案

**题一：** Worker 发送 start 后连接断开。数据库显示 `launching`，不知道容器是否运行。你会重新提交一个新 Job ID、记为失败，还是按原身份恢复？需要什么证据？

**答案：** 按原身份恢复，核对原输入和策略，检查归属该作业的原容器。响应丢失不能推出命令没执行，新 Job ID 会绕过幂等边界。只有实际退出与产物校验完成，才有依据登记成功或失败；结果不确定时保持可恢复状态。

**题二：** A 持有作业锁并被暂停；deadline 已过，B 拿不到锁，SQLite 又暂时被写事务占住。B 应等待所有锁释放再停止，还是让独立监督采取动作？什么时候可以写“停止已确认”？

**答案：** 不启动第二条训练，也不让停止信号依赖 launcher 锁。独立监督持久化私有停止日志，核对容器归属和不可变 ID，发出 KILL 后重新检查退出。SQLite 记账暂不可用可以标记待补，不能阻止已经确认归属容器的停止；只有退出观测成功才能写停止已确认。数据库仍 `stopping` 或 `accounting_pending` 时保留差异并继续监督。

**题三：** Mock 测试全部通过，真实测试容器没有训练输出；数据库后来显示 `completed`，管理员又重启了同一个容器。你分别怎样判断首次无输出和当前是否停止？

**答案：** 首次先看 daemon 的创建/启动错误和 inspect；本课案例是日志配置组合被拒绝，不能先归因模型。当前重新核对所有权并 inspect 不可变 ID；历史 `completed` 不保证当前退出。需要时写持久停止意图、KILL 并再次确认；报告注明观察时点并继续独立监督，而不是用 `demo status` 的缓存记录充当实时证明。

自测通过的标准是：你能不看代码讲清楚这三段时间线，指出哪个组件拥有启动权、哪个拥有停止权、哪些观察还不足以得出结论。练习完成前，学习掌握状态保持待确认。
