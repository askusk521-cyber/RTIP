# RTIP 文档

本仓库包含 RTIP/JAX 迁移、历史迁移记录，以及 `ic5c02384` 反应验证研究数据。

## AI Agent 阅读规则

AI agent 只阅读并使用英文版文档。

中文 Markdown 文件面向人工读者保留；除非用户明确要求中文编辑或翻译，否则 agent 的上下文、任务规划、实现决策和验证记录都应以每组文档中的英文版为准。

## 双语文档约定

所有 Markdown 文档保留英文/中文两份。
另外优先生成英文的记录，再将其翻译为中文版留存。
- 英文主文件尽量使用普通文件名，例如 `README.md`、`USAGE.md`、`summary.md`。
- 中文伴随文件使用 `_ZH.md`，例如 `README_ZH.md`。
- 已存在的中文工作日志不改名，英文伴随版使用 `_EN.md`，例如 `history/daily/20260526_EN.md`。
- 归档迁移文档按目录成对保存：`history/migration/en/` 和 `history/migration/zh/`。

## 主要入口

- `rtipmd/jax/README.md`：RTIP/JAX 包概览。
- `rtipmd/jax/USAGE.md`：环境配置、CLI、Slurm 和输出格式说明。
- `rtipmd/jax/DEEPMD_INTO.md`：DeePMD provider 集成说明。
- `research/ic5c02384/README.md`：`ic5c02384` 论文整理后的输入数据。
- `research/ic5c02384/task_EN.md`：研究任务摘要英文版。
- `research/ic5c02384/references/reaction_reference_EN.md`：文献反应和验证参考英文版。
- `history/README_EN.md`：历史工作日志索引英文版。

AI agent 应优先阅读以上英文文件，以及 `history/daily/*_EN.md` 下的英文日志伴随版。
