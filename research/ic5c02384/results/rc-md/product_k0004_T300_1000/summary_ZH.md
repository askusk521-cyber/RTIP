# Bias 探索摘要

- PDB: `bias_1tBu_CO2_rc-md_product_k0004_T300_1000/bias_1tBu_CO2_rc-md_product_k0004_T300_1000.pdb`
- OUT: `bias_1tBu_CO2_rc-md_product_k0004_T300_1000/bias_1tBu_CO2_rc-md_product_k0004_T300_1000.out`
- 是否达到 promising 成键阈值: `True`

| 帧 | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS 核心 RMSD / A | product 核心 RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -7.580605 | 0.018800 | 1.277481 | 3.099909 | 3.376138 | 3.244724 |
| 最接近 TS | 152 | -7.549489 | 0.010255 | 2.261556 | 2.406695 | 2.006715 | 2.120549 |
| 最接近 product | 317 | -7.572176 | 0.000599 | 1.752371 | 1.609503 | 2.363573 | 1.989881 |
| 最佳成键 | 293 | -7.563243 | 0.000396 | 1.581997 | 1.275408 | 2.221182 | 2.043352 |

参考距离：

| 参考 | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

帧文件：

- final: `bias_1tBu_CO2_rc-md_product_k0004_T300_1000/bias_1tBu_CO2_rc-md_product_k0004_T300_1000_final.xyz`
- best_ts_like: `bias_1tBu_CO2_rc-md_product_k0004_T300_1000/bias_1tBu_CO2_rc-md_product_k0004_T300_1000_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_rc-md_product_k0004_T300_1000/bias_1tBu_CO2_rc-md_product_k0004_T300_1000_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_rc-md_product_k0004_T300_1000/bias_1tBu_CO2_rc-md_product_k0004_T300_1000_best_bond_forming.xyz`
