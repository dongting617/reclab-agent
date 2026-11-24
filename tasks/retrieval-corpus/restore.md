# Snapshot restoration

源码快照按通用换行符归一化计算内容身份。恢复前校验完整内容及只读依赖，损坏快照不会先写入部分候选文件。

Snapshot restore normalizes CRLF and CR to LF before hash verification. A dependency mismatch is rejected instead of silently repairing read-only code.
