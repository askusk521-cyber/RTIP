# Bias 探索摘要

- PDB: `bias_1tBu_CO2_rc-md_ts_k00005_T300_200/bias_1tBu_CO2_rc-md_ts_k00005_T300_200.pdb`
- OUT: `bias_1tBu_CO2_rc-md_ts_k00005_T300_200/bias_1tBu_CO2_rc-md_ts_k00005_T300_200.out`
- 是否达到 promising 成键阈值: `False`

| 帧 | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS 核心 RMSD / A | product 核心 RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 200 | -7.586533 | 0.004807 | 4.208026 | 3.464244 | 2.636805 | 2.683935 |
| 最接近 TS | 1 | -7.598669 | 0.008299 | 4.094638 | 3.316628 | 2.363878 | 2.469147 |
| 最接近 product | 1 | -7.598669 | 0.008299 | 4.094638 | 3.316628 | 2.363878 | 2.469147 |
| 最佳成键 | 19 | -7.587883 | 0.007983 | 4.059646 | 3.315938 | 2.379019 | 2.484066 |

参考距离：

| 参考 | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

帧文件：

- final: `bias_1tBu_CO2_rc-md_ts_k00005_T300_200/bias_1tBu_CO2_rc-md_ts_k00005_T300_200_final.xyz`
- best_ts_like: `bias_1tBu_CO2_rc-md_ts_k00005_T300_200/bias_1tBu_CO2_rc-md_ts_k00005_T300_200_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_rc-md_ts_k00005_T300_200/bias_1tBu_CO2_rc-md_ts_k00005_T300_200_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_rc-md_ts_k00005_T300_200/bias_1tBu_CO2_rc-md_ts_k00005_T300_200_best_bond_forming.xyz`
