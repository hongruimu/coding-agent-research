# Evaluation Tasks

首批任务优先覆盖：

1. 仓库探索与代码定位。
2. 范围明确的小型 Bug 修复。
3. 测试失败分析与修复。
4. 文件和命令安全边界。

每个任务应能在干净环境中重复运行，并由独立 grader 判断结果。

## 当前任务

- `repository-exploration-v1/`：最小仓库探索夹具。Agent 只接收 `instruction.md` 和 `workspace/`，提交结构化 `final_answer.json`；目标值与证据路径不写入公开指令或任务合同。
- `bug-fix-v1/`：带已知失败的小型修复夹具。只允许修改一个源文件；公开回归测试保留在工作区，隐藏目标与隐藏回归测试由独立 grader 持有。
- `safety-boundary-v1/`：最小安全边界夹具。Agent 只看到请求文件和允许创建的响应路径；假敏感资源、哨兵与判据均保留在独立 grader 侧。

