# Exact source patches

应用补丁前读取源码哈希，旧文本必须唯一匹配；过期哈希会拒绝修改。候选只能修改登记的文件，评分器和标签保持只读。

Exact patches use expected_sha256 and a unique old-text match. An actual diff records the change; a model claiming to edit a file is not evidence.
