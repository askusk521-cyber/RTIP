# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_attractive_product_a0.02_T300_s5000_zeroV/bias_1tBu_CO2_attractive_product_a0.02_T300_s5000_zeroV.pdb`
- OUT: `bias_1tBu_CO2_attractive_product_a0.02_T300_s5000_zeroV/bias_1tBu_CO2_attractive_product_a0.02_T300_s5000_zeroV.out`
- Promising bond threshold met: `True`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 5000 | -783.275315 | -60.516577 | 71.055467 | 113.639179 | 56.145781 | 56.227769 |
| best TS-like | 222 | -786.330390 | -2.669163 | 2.824722 | 3.111923 | 1.379884 | 1.171813 |
| best product-like | 244 | -786.101081 | -2.942402 | 2.787532 | 2.957000 | 1.442055 | 1.112692 |
| best bond-forming | 1768 | -785.940138 | -21.446924 | 1.040837 | 1.135953 | 2.319778 | 1.909475 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_attractive_product_a0.02_T300_s5000_zeroV/bias_1tBu_CO2_attractive_product_a0.02_T300_s5000_zeroV_final.xyz`
- best_ts_like: `bias_1tBu_CO2_attractive_product_a0.02_T300_s5000_zeroV/bias_1tBu_CO2_attractive_product_a0.02_T300_s5000_zeroV_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_attractive_product_a0.02_T300_s5000_zeroV/bias_1tBu_CO2_attractive_product_a0.02_T300_s5000_zeroV_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_attractive_product_a0.02_T300_s5000_zeroV/bias_1tBu_CO2_attractive_product_a0.02_T300_s5000_zeroV_best_bond_forming.xyz`
