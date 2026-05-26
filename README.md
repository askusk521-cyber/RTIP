# RTIP Documentation

This repository contains the RTIP/JAX migration, historical migration notes,
and the `ic5c02384` reaction-validation research data.

## AI Agent Reading Rule

AI agents must read and use the English documentation only.

Chinese Markdown files are maintained for human readers, but agent context,
task planning, implementation decisions, and validation notes should come from
the English side of each documentation pair unless the user explicitly asks for
Chinese-language editing or translation.

## Bilingual Documentation Convention

Every Markdown document is kept as an English/Chinese pair.
generate the English documentation then translate it in Chinese.
- English primary files use the normal filename when possible, for example
  `README.md`, `USAGE.md`, or `summary.md`.
- Chinese companion files use `_ZH.md`, for example `README_ZH.md`.
- Legacy Chinese work logs keep their existing filenames and use `_EN.md` for
  the English companion, for example `history/daily/20260526_EN.md`.
- Archived migration documents are paired by directory:
  `history/migration/en/` and `history/migration/zh/`.

## Main Entry Points

- `rtipmd/jax/README.md`: RTIP/JAX package overview.
- `rtipmd/jax/USAGE.md`: environment setup, CLI usage, Slurm usage, and output
  format notes.
- `rtipmd/jax/DEEPMD_INTO.md`: DeePMD provider integration notes.
- `research/ic5c02384/README.md`: prepared input data for the `ic5c02384`
  paper.
- `research/ic5c02384/task_EN.md`: research task summary.
- `research/ic5c02384/references/reaction_reference_EN.md`: literature
  reaction and validation reference.
- `history/README_EN.md`: archived work-log index.

For agents, prefer the files listed above and the English companions under
`history/daily/*_EN.md`.
