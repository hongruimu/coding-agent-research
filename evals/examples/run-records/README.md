# Run Record Examples

这些静态样例使用 `coding-agent/eval-run-record-v1`：

- `passed.json`：功能、安全、记录和已配置预算全部通过；未配置成本预算为 `not_applicable`。
- `budget-exceeded.json`：功能仍为 `pass`，步骤预算为 `exceeded`，总体为 `fail`。
- `safety-failed.json`：安全轴为 `fail`，总体为 `fail`。
- `record-incomplete.json`：记录轴为 `incomplete`，总体为 `fail`。

样例只用于合同和结果一致性测试，不代表真实 Agent 已经运行。消息、工具参数和工具输出均携带脱敏、字节数、截断和脱敏后摘要元数据；模型配置与环境信息禁止使用明显的凭据键名。
