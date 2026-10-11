# RecLab Agent

RecLab Agent 是一个面向推荐算法实验的可评测诊断智能体项目。它把真实推荐实验的结构化证据封装为受限工具，让 Agent 负责规划、选择证据与组织结论，数值计算、文件身份和判定逻辑由确定性 Python 代码完成。

当前里程碑提供：

- 限定在仓库根目录内的只读文件沙箱；
- 证据清单、文件 SHA256 与 JSON 元信息核验；
- 基于 JSON Pointer 的指标读取和数值比较；
- Beam Search 漏召与目标频次差距诊断；
- 15 个带可执行断言的开发集任务；
- 三份来自 Semantic ID 推荐实验的脱敏证据样本。

## 快速开始

项目仅依赖 Python 3.10+ 标准库：

```bash
python -m unittest discover -s tests -v
python -m reclab_agent
python -m reclab_agent --list-tools
```

基准命令会执行 `tasks/dev.json` 中的 15 个参考调用，并输出逐任务结果与通过率。

## 项目结构

```text
reclab_agent/   只读工具、任务加载与基准执行
evidence/       可复现的脱敏实验事实
tasks/          Agent 任务与确定性断言
tests/          工具边界和基准测试
```

## 安全边界

当前工具不能启动训练、改写实验产物或读取仓库以外的文件。后续实验执行能力会采用已注册命令、预算限制和显式审批，不向 Agent 暴露任意 shell。

## 下一步

第二个里程碑将加入 Agent runtime：根据自然语言问题选择工具、记录完整轨迹，并分别评估工具选择、结论正确性、引用完整性和成本。
