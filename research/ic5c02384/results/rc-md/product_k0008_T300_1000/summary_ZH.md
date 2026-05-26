# Bias 探索摘要

- PDB: `bias_1tBu_CO2_rc-md_product_k0008_T300_1000/bias_1tBu_CO2_rc-md_product_k0008_T300_1000.pdb`
- OUT: `bias_1tBu_CO2_rc-md_product_k0008_T300_1000/bias_1tBu_CO2_rc-md_product_k0008_T300_1000.out`
- 是否达到 promising 成键阈值: `True`

| 帧 | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS 核心 RMSD / A | product 核心 RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -7.548854 | 0.022677 | 1.222219 | 2.722452 | 5.253721 | 4.566438 |
| 最接近 TS | 112 | -7.504382 | 0.019417 | 2.201655 | 2.082736 | 1.874763 | 1.988040 |
| 最接近 product | 345 | -7.460593 | 0.002033 | 1.387291 | 1.854419 | 2.482858 | 1.817830 |
| 最佳成键 | 288 | -7.516190 | 0.000135 | 1.444029 | 1.404181 | 2.385875 | 1.936788 |

参考距离：

| 参考 | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

帧文件：

- final: `bias_1tBu_CO2_rc-md_product_k0008_T300_1000/bias_1tBu_CO2_rc-md_product_k0008_T300_1000_final.xyz`
- best_ts_like: `bias_1tBu_CO2_rc-md_product_k0008_T300_1000/bias_1tBu_CO2_rc-md_product_k0008_T300_1000_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_rc-md_product_k0008_T300_1000/bias_1tBu_CO2_rc-md_product_k0008_T300_1000_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_rc-md_product_k0008_T300_1000/bias_1tBu_CO2_rc-md_product_k0008_T300_1000_best_bond_forming.xyz`
