# Evaluations

本目录用于验证 Coding Agent 是否真正完成端到端编程任务，不用于替代组件测试。

## 目录职责

- `tasks/`：任务指令、初始环境、约束和时间预算。
- `graders/`：测试、文件差异和约束检查等独立验收逻辑。
- `runs/`：本地评测运行结果，由程序生成且不提交 Git。

每个任务至少记录：

- `task_id`
- `instruction`
- `environment`
- `grader`
- `timeout`
- `max_steps`
- `expected_constraints`

