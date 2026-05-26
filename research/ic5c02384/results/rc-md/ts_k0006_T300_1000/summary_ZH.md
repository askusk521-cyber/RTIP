# Bias 探索摘要

- PDB: `bias_1tBu_CO2_rc-md_ts_k0006_T300_1000/bias_1tBu_CO2_rc-md_ts_k0006_T300_1000.pdb`
- OUT: `bias_1tBu_CO2_rc-md_ts_k0006_T300_1000/bias_1tBu_CO2_rc-md_ts_k0006_T300_1000.out`
- 是否达到 promising 成键阈值: `True`

| 帧 | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS 核心 RMSD / A | product 核心 RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -7.540607 | 0.002110 | 2.740785 | 2.138549 | 2.291689 | 2.224380 |
| 最接近 TS | 140 | -7.572156 | 0.006399 | 2.743379 | 2.468511 | 2.198193 | 2.283372 |
| 最接近 product | 362 | -7.539482 | 0.018271 | 3.477786 | 2.699369 | 2.751171 | 2.036670 |
| 最佳成键 | 186 | -7.529515 | 0.002495 | 2.326562 | 1.992452 | 2.318408 | 2.332585 |

参考距离：

| 参考 | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

帧文件：

- final: `bias_1tBu_CO2_rc-md_ts_k0006_T300_1000/bias_1tBu_CO2_rc-md_ts_k0006_T300_1000_final.xyz`
- best_ts_like: `bias_1tBu_CO2_rc-md_ts_k0006_T300_1000/bias_1tBu_CO2_rc-md_ts_k0006_T300_1000_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_rc-md_ts_k0006_T300_1000/bias_1tBu_CO2_rc-md_ts_k0006_T300_1000_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_rc-md_ts_k0006_T300_1000/bias_1tBu_CO2_rc-md_ts_k0006_T300_1000_best_bond_forming.xyz`
