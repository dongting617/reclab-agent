# Researcher and independent auditor

主研究者调用任务允许的工具；独立审计者使用隔离会话，只读冻结证据，先独立分析再核查研究结论。研究和审计共用不可重置的请求预算。

The researcher can execute authorized research tools. The independent auditor can only read frozen evidence using read_audit_evidence; it cannot edit source, train, change budgets or access arbitrary files. Audit checkpoints preserve separate contexts and request reservations across restart.
