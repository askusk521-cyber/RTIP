# 教科书验证会话记录（2026-08-19）

## 目标

用高中教科书反应验证 DeePMD + RTIP-MD（JAX 移植）的正确性。

## 做了什么（按时间）

1. 选体系: CH2O+H2→CH3OH（正1）、C2H4+H2→C2H6（正2）、CH2O+CH4（负）。
   几何取实验键长; 用仓库 `synthesize` CLI 生成 5 Å 分离的反应对;
   sidecar `atom_add_pot` 覆盖全部原子。
2. 首次提交 431/432/433（旧代码）: 431 在第 600 步 NaN。
   诊断: `rti_pot_force` 在 RTI 距离=0（两碎片质心精确重合）时
   `1/d^7` 与除法 0/0（rtip.out 第 600 行 pot_rtip/f_rtip 先 NaN，
   真实势 pot_real/f_real 仍有限——定位到偏置势而非 DeePMD）。
3. 修复: rtip.py 三个函数加 `_RTI_DIST_EPS = 1e-6` Bohr 平移;
   新增回归测试; 全套测试 91 passed。
4. 用修复后代码重跑三个 case（440/441/442）:
   - ch2o_h2 通过（甲醇）
   - c2h4_h2 通过（乙烷）
   - ch2o_ch4 未通过（瞬时 C–C + CH2O 分解为 CO+H2）
5. 单点/基准（DPA-3.2-5M, OMol25 默认头）:
   - 反应能偏差: +22.9 / −158.4 / +157.4 kcal/mol
   - H2 曲线: 键长 0.74 Å 正确, 解离能偏高 0.6 eV
   - cell=None 与 20 Å 盒子结果一致（调用方式无问题）
6. 发现并修复分析脚本中的两个伪影:
   - 原子半径是 Bohr 而距离用 Å（单位 bug）
   - 1.25 倍邻接把乙烯 1,2-C–H 近距接触(~1.2 Å)误判为键
   最终方案: 按 case 追踪特定原子对 + 教科书物理阈值。
7. 事故记录: 温和负对照复用时用了相同 CASE 名，作业覆盖了主负对照的
   rtip.out（轨迹 PDB 在本地有完整副本）。规则已写入 AGENTS.md:
   case 目录必须唯一。
8. 重跑中: validation_ch2o_ch4（主负对照）与 validation_ch2o_ch4_gentle
   （温和参数负对照）。

## 关键数字（DeePMD 单点，实验几何）

| 量 | 值 |
|---|---|
| CH2O+H2→CH3OH ΔE | +0.97 kcal/mol（参考 −21.9） |
| C2H4+H2→C2H6 ΔE | −191.0 kcal/mol（参考 −32.6） |
| CH4+2O2→CO2+2H2O ΔE | −34.4 kcal/mol（参考 −191.8） |
| H2 平衡键长 | 0.74 Å（实验 0.741） |
| H2 解离能 | 5.10 eV（实验 ~4.48） |

## 教训

* slurm 脚本里 `#SBATCH -o logs/...` 相对路径依赖提交时 cwd；
  已改为绝对路径。
* 结果目录名必须唯一（见 AGENTS.md）。
* 分析键事件时不要盲目复用 formose 的 1.25 邻接约定。
