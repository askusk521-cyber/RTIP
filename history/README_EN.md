# HISTORY

This directory archives completed or frozen migration records, migration plans,
module mappings, test strategies, numerical-difference notes, and work logs.

Archive criteria:

- The remote tree `n5:/home/lhshen/RTIP/rtipmd/jax` passed the full test suite
  on 2026-05-24 with `87 passed`.
- The RTIP/JAX engineering migration, workflow entry points, DeePMD provider
  boundary, and currently testable scope are complete.
- Scientific benchmarks, dedicated model validation, and Stage 7 JAX-native
  performance optimization are follow-up research or optimization work, so they
  are archived away from the main instruction documents.

Manual-style documents kept in their original locations:

- `rtipmd/jax/README.md`
- `rtipmd/jax/USAGE.md`
- `rtipmd/jax/DEEPMD_INTO.md`
- `research/ic5c02384/README.md`
- `research/ic5c02384/reactions/**/README.md`
- `research/ic5c02384/task_EN.md`

Directory guide:

- `daily/`: work logs (`20260519.md`, `20260524.md`, `20260525.md`,
  `20260526.md`) with English companions named `*_EN.md`.
- `week1conclusion.md`: first-week summary, paired with
  `week1conclusion_EN.md`.
- `migration/en/`: English migration-history documents.
- `migration/zh/`: Chinese migration-history documents.
