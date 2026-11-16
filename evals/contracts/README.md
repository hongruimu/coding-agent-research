# Evaluation Contract v1

M0.2 使用两个相互关联的 UTF-8 JSON 文档定义评测边界：

- `task.json`：编排侧任务身份、固定环境、Agent 可见输入、grader 引用、约束、预算和必需运行产物。
- `grader.json`：独立 grader 身份、目标任务身份、隐藏输入和必需运行产物。

## Task Contract

`schema` 必须为 `coding-agent/eval-task-v1`。必需字段：

- `task.id`、`task.version`、`task.kind`；
- `environment.fixture_revision`、`environment.network`；
- `agent_input.instruction_path`、`agent_input.workspace_path`；
- `grader.id`、`grader.version`；
- `constraints.allowed_paths`、`constraints.forbidden_paths`；
- `budgets.max_steps`、`max_time_seconds`、`max_tool_calls`、`max_output_bytes`、`max_cost_usd`；
- `artifacts.required`。

Agent 可见路径必须使用 POSIX 相对路径，解析后必须仍位于当前任务目录。绝对路径、`..`、反斜杠路径、越界路径和指向隐藏 grader 目录的路径均无效。

## Grader Contract

`schema` 必须为 `coding-agent/eval-grader-v1`。必需字段：

- `grader.id`、`grader.version`；
- `task.id`、`task.version`；
- `hidden_inputs`；
- `artifacts.required`。

隐藏输入必须是 grader 目录内已存在的文件。任务引用的 grader 身份必须与 grader 合同一致，grader 指向的任务身份也必须与任务合同一致。

## 当前边界

`coding_agent.evals.contracts.load_contract_pair` 只加载和校验合同，不执行 Agent、不运行 grader、不创建沙箱，也不判定预算或总体结果。运行记录和结果语义属于 `M0.2-T2`。

最小合法样例位于 `evals/contracts/examples/minimal/`。
