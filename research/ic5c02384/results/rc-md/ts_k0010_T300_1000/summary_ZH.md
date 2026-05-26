# Bias 探索摘要

- PDB: `bias_1tBu_CO2_rc-md_ts_k0010_T300_1000/bias_1tBu_CO2_rc-md_ts_k0010_T300_1000.pdb`
- OUT: `bias_1tBu_CO2_rc-md_ts_k0010_T300_1000/bias_1tBu_CO2_rc-md_ts_k0010_T300_1000.out`
- 是否达到 promising 成键阈值: `True`

| 帧 | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS 核心 RMSD / A | product 核心 RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -7.545382 | 0.014382 | 3.529612 | 2.056105 | 0.991170 | 0.910266 |
| 最接近 TS | 1000 | -7.545382 | 0.014382 | 3.529612 | 2.056105 | 0.991170 | 0.910266 |
| 最接近 product | 1000 | -7.545382 | 0.014382 | 3.529612 | 2.056105 | 0.991170 | 0.910266 |
| 最佳成键 | 157 | -7.513482 | 0.006391 | 2.249734 | 2.081722 | 2.225033 | 2.277555 |

参考距离：

| 参考 | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

帧文件：

- final: `bias_1tBu_CO2_rc-md_ts_k0010_T300_1000/bias_1tBu_CO2_rc-md_ts_k0010_T300_1000_final.xyz`
- best_ts_like: `bias_1tBu_CO2_rc-md_ts_k0010_T300_1000/bias_1tBu_CO2_rc-md_ts_k0010_T300_1000_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_rc-md_ts_k0010_T300_1000/bias_1tBu_CO2_rc-md_ts_k0010_T300_1000_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_rc-md_ts_k0010_T300_1000/bias_1tBu_CO2_rc-md_ts_k0010_T300_1000_best_bond_forming.xyz`
