# 第二周：真实 Docker Worker 验收

2026-10-20：本页对应的七类真实 CPU Docker 验收已在本机 WSL2 Ubuntu 通过，受控 CLI 亦完成固定训练→预测→外部评分；汇总见 [CPU 验收报告](../reports/week-02-cpu-acceptance-20261020.json)。Linux 全量188/188通过、0跳过，Windows178通过/10平台条件跳过/0失败。AutoDL 容器实例不支持 Docker，不能在该实例上用宿主 Python 代替候选容器；真实云端/GPU/Beauty新训练仍待验。下面固定程序仅用 Python 标准库、合成数据与 CPU popularity 拟合，不下载模型、不调用 API，不构成推荐质量提升证据。

## 本次实测范围

真实验收在 WSL2 的本地 Linux daemon 执行，使用管理员预拉取并保存原始 ECR repository/digest 的官方 Python 镜像。七类真实用例与五项准备/证据合同测试共12项通过；完整188项回归还包含 FakeDocker/协议测试，不能把全量计数当成188个容器案例。真实失败曾定位到 `local` 日志驱动默认压缩与 `max-file=1` 冲突，修复每容器 `compress=false` 后重新运行并保留失败记录。

CLI 实跑两个作业均 completed/exit0，seed42；12个合成验证用户的 Recall/NDCG@10/20均为0，外部评分与fixture设计一致。`fixture_only`、`llm_connected=false`，无LLM优化声明。第二周前三完成项只在这些受控CPU条件下勾选，第四项“真实云端容器运行”仍未勾选；学习掌握待自测。

## 为什么需要另一套测试

`tests/test_execution.py` 的假 Docker runner 可以稳定注入断线、竞态和未知 ownership，证明 Worker 的控制流程。真正的非 root、只读挂载、禁网、cgroup OOM 和进程停止必须让本地 Linux daemon 创建真实容器。新增 `tests/test_real_execution.py` 保留这两类证据的区别。

未设置 `RECLAB_REAL_DOCKER` 时，7 项真实测试显式跳过，5 项准备/证据保存合同测试仍可运行。显式设置 `RECLAB_REAL_DOCKER=1` 后，缺 Linux、本地 Docker、已预装真实摘要、seccomp 或所需 cgroup 能力均导致测试失败，绝不退回假 Docker 或宿主执行。无效 opt-in 值也失败。

## 在专用 Linux Docker 主机运行

前置：管理员审核 Python 官方镜像、Docker 权限和测试源码，并有可用的本地 `/var/run/docker.sock`。测试只接受 Docker Hub 的 `python@sha256:...`（含 `library/`、`docker.io/library/` 明确别名）或 Docker 官方 ECR 发布地址 `public.ecr.aws/docker/library/python@sha256:...`，不接受可变 tag 或任意其他仓库。准备步骤允许管理员从这两个明确官方仓库拉取一次、解析实际 RepoDigest，随后 Worker 仅使用精确摘要与 `--pull never`。如需跨次严格复现，应复用首次记录的真实摘要，而不是重新解析 tag。

当 Docker Hub 网络不可达时，可由管理员先拉取 `public.ecr.aws/docker/library/python:3.12-slim`。Docker 官方说明与 AWS 合作发布 Docker Official Images，并在 Hub 更新后自动推送到 ECR；AWS 的认证 Docker 发布者目录列有 Python。[Docker 官方同步说明](https://www.docker.com/blog/news-from-aws-reinvent-docker-official-images-on-amazon-ecr-public/)、[AWS 官方发布说明](https://aws.amazon.com/blogs/containers/docker-official-images-now-available-on-amazon-elastic-container-registry-public/)、[ECR Python 仓库](https://gallery.ecr.aws/docker/library/python)

ECR 不是任意第三方镜像白名单。必须使用其 inspect 中实得的 `public.ecr.aws/docker/library/python@sha256:<真实摘要>`，不得把这个 URI 改写成 `python@sha256:...` 来绕过身份检查；相同 tag 也不能推定两个 registry 的 manifest/index 摘要相同。provenance 分别保存真实 repository、registry、来源页面与官方分发依据。

```bash
# 在仓库根目录；下列 IMAGE 必须是管理员已审核、预装的真实摘要。
export IMAGE='python@sha256:<替换真实64位十六进制摘要>'
python3 scripts/prepare_worker_acceptance.py \
  --image "$IMAGE" --evidence .reclab/worker-acceptance/preflight
export RECLAB_DOCKER_TEST_IMAGE="$IMAGE"
export RECLAB_REAL_DOCKER=1
export RECLAB_ACCEPTANCE_EVIDENCE=.reclab/worker-acceptance
python3 -m unittest discover -s tests -p test_real_execution.py -v
```

命令中的 `<...>` 是不可执行占位符。准备脚本只 inspect 已存在镜像和 daemon，保存来源与代码 SHA256；它不会拉镜像、构建镜像或执行候选。测试也不要求安装项目依赖，从仓库根目录导入本地标准库包即可。

## 每项通过意味着什么

| 真实测试 | 实际核查 |
|---|---|
| fit → predict → grader | UID/GID 65532、NoNewPrivs、CapEff=0、seccomp=2、nofile=256；根/源码/数据写入失败，网络仅 loopback 且外连失败；测试标签和宿主哨兵不可读；只读 `/model`；外部评分结果与可信 popularity 排名一致，忽略候选伪造指标 |
| 重复提交与 Worker 重建 | 同一源码、数据、配置、种子得到相同 job ID、容器 ID 和 StartedAt；owner 下只有一个容器；重建 Worker 从 SQLite 恢复 |
| 取消 | 发出取消后检查真实容器退出、Running=false、exit=137；终态必须是 cancelled |
| 超时 | 独立 `serve` 线程监督到期；调用方只读 SQLite，不能靠调用方 poll 偷做停止；确认 timed_out 和 Running=false |
| 独立 watchdog | 一个线程一直持有 Worker 作业锁，Worker 不 poll；独立 watchdog 到期后确认真实容器已停止，但锁忙时保留 stopping；释放锁后 Worker 恢复到 timed_out |
| CPU/PID | inspect 的真实 HostConfig 与容器 cgroup 值一致；0.25 CPU 的固定短计算出现 throttling；最多尝试 10 个直接子进程，PID=8 时 EAGAIN 后全部回收，无递归创建 |
| 内存 | cgroup=64 MiB 下至多分配 256 MiB，必须实际 OOMKilled=true、exit=137、Worker=failed/oom_killed |

每个已终止容器还检查 NetworkMode、ReadonlyRootfs、Capabilities、CPU、Memory/MemorySwap、PidsLimit、Init、IPC 与精确挂载列表。预测只挂载前一成功训练的输出并验证模型未被改写；完成后人为修改预测文件时，Worker 必须因回执不符拒绝收集。

模型拟合仅统计训练交互频次，预测输出普通 JSON，由可信 `DatasetBundle.grade` 调固定独立评分器。fixture 的 NDCG@20=0 是合成设计的结果，用来证明不会采纳候选自报的 1.0，不能据此评价真实模型。

## 证据在哪里

每次运行新建 `.reclab/worker-acceptance/run-<随机运行ID>/`，防止历史容器与状态污染。随机 ID 只区分独立验收运行；同一次重复提交测试仍使用同一确定作业身份。`latest-run.txt` 指向当前运行。

- `provenance.json`：真实 registry/repository、官方来源与分发依据、实得 RepoDigest、image ID、架构、daemon/cgroup 能力、执行/评分/测试源码哈希及允许的 CI 运行身份。
- 每个测试目录：固定训练/预测源码与配置、合成 public/private 分区、Worker SQLite、实际输出、完整 inspect（含 HostConfig/State/Mounts）、有界容器日志和可信 grader 报告。
- 成功或失败都先保存证据。清理只针对本次 Worker 已知且三个 ownership label 精确匹配的容器；先取消并确认停止，再按容器 ID 删除，不使用 `--force`。清理失败也使测试失败。

这些文件只含合成数据和非秘密主机哨兵，不读取、导出或挂载真实 SSH/API 凭据。不要把真实业务数据或凭据放进验收目录。证据根位于 `.reclab/`，默认不进入 Git。

## 手动 GitHub Actions 方案

`.github/workflows/worker-acceptance.yml` 只提供 `workflow_dispatch`，不会因 push 或 PR 自动启动。候选运行位置为 GitHub 托管 Ubuntu 24.04 CPU 主机；Action 权限只有 `contents: read`，checkout 不保存认证信息，无 secrets、无镜像发布、无部署。checkout/upload-artifact 均固定到官方仓库的完整 commit SHA。

workflow 先从 `docker.io/library/python:3.12-slim` 拉官方镜像、保存 image inspect、解析实际 RepoDigest，再只按精确摘要运行以上套件。官方 tag 在首次解析前可变，因此证据必须记录具体 digest；不允许把不同 digest 的两次结果直接视作完全相同环境。无 pip install、远程安装脚本或模型下载。证据即使失败也保留 7 天。

本次仅写入 workflow，尚未 push 或触发远端 Action。CPU 真容器结果来自本机 WSL2，不来自 GitHub Actions；不能把本地通过说成远端 Action 或云端容器已通过。所有 fake 单元测试通过也不能代替真实容器。

## 仍需另行完成

本套件不是 GPU 验收。NVIDIA Container Toolkit、GPU 可见性/实际训练、真实 Beauty 适配器在容器中的训练—预测闭环仍需 Docker GPU 主机。独立 watchdog 仍需单独被系统服务监督，Worker 与 watchdog 同时离线时不保证按时停止；输出只有收集预算而非写入时磁盘 quota，共享 GPU 显存没有硬配额。这些限制须如实记录，不能由当前 CPU 通过结果推导为已具备生产多租户隔离。

学习自测：为什么 `docker create --memory 64m` 参数正确，仍要执行有界 OOM 测试？因为参数构造只证明意图，daemon/cgroup 可能不支持或配置不生效；真实 HostConfig、容器 cgroup 文件和 OOMKilled 状态构成不同层面的证据。为什么取消后需要 inspect？因为请求送达不代表进程已退出，不能在 Running 仍为 true 时记录 cancelled。

官方资料：[Docker 资源限制](https://docs.docker.com/engine/containers/resource_constraints/)、[Python 官方镜像](https://hub.docker.com/_/python)、[固定版本 checkout](https://github.com/actions/checkout/tree/11bd71901bbe5b1630ceea73d27597364c9af683)、[固定版本 upload-artifact](https://github.com/actions/upload-artifact/tree/ea165f8d65b6e75b540449e92b4886f43607fa02)。
