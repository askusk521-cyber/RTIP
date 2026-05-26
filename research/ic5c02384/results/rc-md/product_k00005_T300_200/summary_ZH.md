# Bias 探索摘要

- PDB: `bias_1tBu_CO2_rc-md_product_k00005_T300_200/bias_1tBu_CO2_rc-md_product_k00005_T300_200.pdb`
- OUT: `bias_1tBu_CO2_rc-md_product_k00005_T300_200/bias_1tBu_CO2_rc-md_product_k00005_T300_200.out`
- 是否达到 promising 成键阈值: `False`

| 帧 | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS 核心 RMSD / A | product 核心 RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 200 | -7.581677 | 0.007850 | 3.752085 | 3.407329 | 2.427672 | 2.517993 |
| 最接近 TS | 104 | -7.590496 | 0.012034 | 4.098191 | 3.483791 | 2.353018 | 2.434864 |
| 最接近 product | 97 | -7.586783 | 0.012348 | 4.083044 | 3.483744 | 2.353531 | 2.434237 |
| 最佳成键 | 200 | -7.581677 | 0.007850 | 3.752085 | 3.407329 | 2.427672 | 2.517993 |

参考距离：

| 参考 | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

帧文件：

- final: `bias_1tBu_CO2_rc-md_product_k00005_T300_200/bias_1tBu_CO2_rc-md_product_k00005_T300_200_final.xyz`
- best_ts_like: `bias_1tBu_CO2_rc-md_product_k00005_T300_200/bias_1tBu_CO2_rc-md_product_k00005_T300_200_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_rc-md_product_k00005_T300_200/bias_1tBu_CO2_rc-md_product_k00005_T300_200_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_rc-md_product_k00005_T300_200/bias_1tBu_CO2_rc-md_product_k00005_T300_200_best_bond_forming.xyz`
