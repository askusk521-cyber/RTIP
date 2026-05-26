# Bias 探索摘要

- PDB: `bias_1tBu_CO2_rc-md_product_k0010_T300_1000/bias_1tBu_CO2_rc-md_product_k0010_T300_1000.pdb`
- OUT: `bias_1tBu_CO2_rc-md_product_k0010_T300_1000/bias_1tBu_CO2_rc-md_product_k0010_T300_1000.out`
- 是否达到 promising 成键阈值: `True`

| 帧 | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS 核心 RMSD / A | product 核心 RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -7.551050 | 0.052768 | 1.227012 | 2.873286 | 3.926805 | 4.749382 |
| 最接近 TS | 105 | -7.464710 | 0.018218 | 2.052200 | 1.981416 | 1.821244 | 1.939102 |
| 最接近 product | 103 | -7.476335 | 0.019678 | 2.096422 | 2.003034 | 1.822242 | 1.938580 |
| 最佳成键 | 202 | -7.453269 | 0.007082 | 1.247784 | 2.073354 | 2.100451 | 2.041416 |

参考距离：

| 参考 | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

帧文件：

- final: `bias_1tBu_CO2_rc-md_product_k0010_T300_1000/bias_1tBu_CO2_rc-md_product_k0010_T300_1000_final.xyz`
- best_ts_like: `bias_1tBu_CO2_rc-md_product_k0010_T300_1000/bias_1tBu_CO2_rc-md_product_k0010_T300_1000_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_rc-md_product_k0010_T300_1000/bias_1tBu_CO2_rc-md_product_k0010_T300_1000_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_rc-md_product_k0010_T300_1000/bias_1tBu_CO2_rc-md_product_k0010_T300_1000_best_bond_forming.xyz`
