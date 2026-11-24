# Workbench event recovery

Web 与 CLI 查看同一份持久化任务和研究会话。任务幂等键绑定请求内容，SSE 用事件序号续读；刷新网页不会重新提交训练。

Last-Event-ID resumes event delivery. A duplicate idempotency key returns the original task; a changed payload with the same key is a conflict.
