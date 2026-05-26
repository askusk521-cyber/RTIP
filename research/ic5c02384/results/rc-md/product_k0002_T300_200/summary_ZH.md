# Bias 探索摘要

- PDB: `bias_1tBu_CO2_rc-md_product_k0002_T300_200/bias_1tBu_CO2_rc-md_product_k0002_T300_200.pdb`
- OUT: `bias_1tBu_CO2_rc-md_product_k0002_T300_200/bias_1tBu_CO2_rc-md_product_k0002_T300_200.out`
- 是否达到 promising 成键阈值: `True`

| 帧 | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS 核心 RMSD / A | product 核心 RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 200 | -7.560509 | 0.006963 | 2.498790 | 2.455138 | 2.194256 | 2.288123 |
| 最接近 TS | 167 | -7.581671 | 0.013218 | 2.893594 | 2.805219 | 2.150007 | 2.258479 |
| 最接近 product | 163 | -7.582702 | 0.014293 | 2.957253 | 2.849017 | 2.150591 | 2.258124 |
| 最佳成键 | 200 | -7.560509 | 0.006963 | 2.498790 | 2.455138 | 2.194256 | 2.288123 |

参考距离：

| 参考 | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

帧文件：

- final: `bias_1tBu_CO2_rc-md_product_k0002_T300_200/bias_1tBu_CO2_rc-md_product_k0002_T300_200_final.xyz`
- best_ts_like: `bias_1tBu_CO2_rc-md_product_k0002_T300_200/bias_1tBu_CO2_rc-md_product_k0002_T300_200_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_rc-md_product_k0002_T300_200/bias_1tBu_CO2_rc-md_product_k0002_T300_200_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_rc-md_product_k0002_T300_200/bias_1tBu_CO2_rc-md_product_k0002_T300_200_best_bond_forming.xyz`
