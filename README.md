# Coding Agent（从零构建，进行中）

> 一个**从零构建 Coding Agent 的学习仓库**：记录过程、代码、测试和可复现证据。
> 这不是生产可用的产品，也不以复刻 Codex 为目标。欢迎围观过程、跑测试、提质疑。

## 这是什么

这个仓库是我亲手构建一个 Coding Agent 的过程证据。它存在的目的不是「又开源一个 Agent」，而是把「一个工程判断如何形成、被验证、被推翻」这条路公开出来。

因此你会看到一些和普通开源项目不一样的地方：

- 测试里包含大量**「构造失败」的用例**——不只是验证「能跑通」，更验证「坏数据会在正确的地方被拦下」。
- 我会如实写明**当前不能做什么**，不把未完成的能力包装成已就绪。
- 阶段进度和证据边界见 [`docs/PROGRESS.md`](docs/PROGRESS.md)。

关于「每一步判断怎么来的」，我会写成文章发布；本仓库只保留工程资产（代码、测试、评测夹具、进度），个人学习结论和内部路线决策保存在私有仓库中，不在这里重复。

## 当前状态

### 已经有的

**评测基线（三类任务 + 独立验收器）**

- 仓库探索、小型 Bug 修复、安全边界三类最小评测任务；
- 每类任务配独立 grader（验收器），把「候选产物」和「隐藏判据」分离；
- 运行记录、预算语义、敏感信息脱敏与截断。

**核心协议 + 确定性模型替身**

- `coding_agent.protocol`：供应商无关的 `Message` / `ModelResponse` / `ToolCall` / `ToolResult` 四结构与关联不变量；
- `coding_agent.model`：同步 `Model` 接口 + FIFO 的 `ScriptedModel`（假模型），用于确定性验证协议正确性。

### 还没有的（诚实声明）

- 真实的 Agent 执行循环（什么时候调模型、什么时候停）；
- 工具注册、参数校验与真实工具执行；
- 真实模型供应商适配、流式、异步、取消；
- 文件写入、Shell 执行及对应的权限/超时/清理边界；
- 上下文构建、预算控制、会话持久化；
- 端到端完成真实任务的成率数据。

一句话：**目前定义好了 Agent 与模型之间的「语言」，还没让它们用这门语言去「做事」。**

## 快速复现测试

项目最低支持 **Python 3.13**，无运行时第三方依赖（测试只用 pytest）。

```bash
git clone https://github.com/hongruimu/coding-agent-research.git
cd coding-agent-research
python3.13 -m venv .venv
source .venv/bin/activate        # Windows 用 .venv\Scripts\activate
python -m pip install -e '.[dev]'
pytest
```

预期结果：`123 passed`。

## 仓库结构

```text
coding-agent-research/
├── src/coding_agent/       # 实现：协议、模型接口、评测合同
├── tests/
│   ├── unit/               # 组件与协议测试
│   ├── integration/        # 多组件协作测试
│   └── safety/             # 权限、脱敏与边界测试
├── evals/
│   ├── tasks/              # 端到端评测任务定义
│   ├── graders/            # 独立验收逻辑
│   ├── contracts/          # 任务合同格式与示例
│   └── examples/           # 运行记录示例
├── docs/
│   └── PROGRESS.md         # 阶段进度、证据与边界
├── pyproject.toml
├── LICENSE
└── README.md
```

## 项目原则

- 先闭环，再扩展；先解决真实问题，再抽象通用能力。
- 安全边界在执行能力之前建立（首次允许写文件 / 执行命令前，先有边界）。
- 协议正确性用 Scripted/Fake Model 做确定性验证，真实模型只做独立适配验证。
- 用测试和评测证据判断进步，不依赖模型自评或「感觉变好了」。
- 阶段结论带着证据和边界一起出现；未验证的判断显式标注，不包装成确定结论。

## 开源边界

- 这是一个**进行中（WIP）的学习仓库**，接口和结构会随真实需求变化，不保证向后兼容。
- 欢迎提 issue 讨论工程问题，但请把它当「学习过程记录」而非「生产工具」来评估。
- 评测夹具（`evals/graders/*/hidden/`）里的隐藏判据**公开是为了可复现**，不构成对任何「真实盲测任务」的承诺。

## License

[MIT](./LICENSE)
