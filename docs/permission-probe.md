# 第一周：固定可信文件权限探针

`scripts/probe_dataset_permissions.py` 只证明指定非特权进程对指定文件的访问结果。它不执行候选代码，不接受 shell 或代码参数，不能作为第二周 Docker/Worker 验收。

前提：可信 Linux root 进程、`fork`、`prctl`、可读 `/proc/self/status`；`/root` 已为 root-owned `0700`。public 直属仅 `manifest.json`、`train.json`、`valid.json`，不允许 symlink、其他文件/目录或已知测试标签 JSON 键。私有 test 文件与原始含 test 的 JSON 必须是 `/root` 下真实存在的不同普通文件，所有路径不含 symlink。脚本不会修改现有文件或权限；若平台能力不足，退出码非零。

可信导入器应先从原数据导出 public，不能仅靠关键字检查判断数据来源。脚本检查有限的已知测试键，不能辨识人为伪装在任意字段中的标签；public 的来源、schema 和内容仍须独立审计。

```bash
/root/miniconda3/bin/python scripts/probe_dataset_permissions.py \
  --public-dir /root/reclab-trusted/beauty/public \
  --private-test-path /root/reclab-trusted/beauty/private/test.json \
  --original-source-path /root/autodl-tmp/semantic-id-rec-current/data/processed/beauty-sasrec.json \
  --output /root/reclab-trusted/permission-report-unique.json
```

执行时新建 `/tmp/reclab-permission-probe-*`，public 拷贝目录为 `0555`、文件为 `0444`，并保留该唯一 stage 供检查。父进程先证明私有文件存在并记录哈希；固定子进程完成所有加载后清空环境、关闭非标准 FD、清除补充组、清空 capability bounding set、切换 real/effective/saved UID/GID 为 65534，启用 `no_new_privs`，再核对所有 capability sets 为零。

子进程实际读 public 必须成功；打开 public 写权限必须 `EACCES`；直接访问原始 JSON、私有 test、父目录越界路径和专用 symlink 均必须 `EACCES`。`ENOENT` 不算权限验证成功。写权限探针不使用 `O_TRUNC`，即使意外可写也不破坏源文件。结束后父进程复核 public 与私有原件哈希。

报告不输出文件内容，保存源码/数据哈希、UID/GID、caps、`NoNewPrivs`、各次访问 errno。`status=passed` 与 `sandbox_established=false` 同时出现是正确语义：仅有限文件权限实测通过。

没有建立 namespace/chroot、网络、进程、CPU/内存/磁盘/GPU 资源隔离，没有验证任务超时、取消、恢复。该脚本内部的 15 秒看门狗只约束固定探针，不是通用 Worker 的生命周期能力。Docker 验收仍未完成，禁止以此开放 Agent 生成代码执行。

Windows 单测只验证输入、拒绝条件与报告合同；真实 Linux 权限证据必须另外实际运行并保存报告。
