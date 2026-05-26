# Bias 探索摘要

- PDB: `bias_1tBu_CO2_synthesis-md_a0002_T300_200/bias_1tBu_CO2_synthesis-md_a0002_T300_200.pdb`
- OUT: `bias_1tBu_CO2_synthesis-md_a0002_T300_200/bias_1tBu_CO2_synthesis-md_a0002_T300_200.out`
- 是否达到 promising 成键阈值: `False`

| 帧 | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS 核心 RMSD / A | product 核心 RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 200 | -7.583572 | -0.242601 | 3.057290 | 3.207533 | 2.103872 | 1.928139 |
| 最接近 TS | 145 | -7.592906 | -0.175868 | 3.449408 | 3.379697 | 2.042464 | 2.102356 |
| 最接近 product | 200 | -7.583572 | -0.242601 | 3.057290 | 3.207533 | 2.103872 | 1.928139 |
| 最佳成键 | 200 | -7.583572 | -0.242601 | 3.057290 | 3.207533 | 2.103872 | 1.928139 |

参考距离：

| 参考 | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

帧文件：

- final: `bias_1tBu_CO2_synthesis-md_a0002_T300_200/bias_1tBu_CO2_synthesis-md_a0002_T300_200_final.xyz`
- best_ts_like: `bias_1tBu_CO2_synthesis-md_a0002_T300_200/bias_1tBu_CO2_synthesis-md_a0002_T300_200_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_synthesis-md_a0002_T300_200/bias_1tBu_CO2_synthesis-md_a0002_T300_200_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_synthesis-md_a0002_T300_200/bias_1tBu_CO2_synthesis-md_a0002_T300_200_best_bond_forming.xyz`
