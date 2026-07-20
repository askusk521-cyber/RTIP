# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_attractive_product_a0.02_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.02_T300_s1000_zeroV.pdb`
- OUT: `bias_1tBu_CO2_attractive_product_a0.02_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.02_T300_s1000_zeroV.out`
- Promising bond threshold met: `True`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -785.390591 | -12.130613 | 1.819525 | 1.747723 | 2.558047 | 2.152655 |
| best TS-like | 222 | -786.330390 | -2.669163 | 2.824722 | 3.111923 | 1.379884 | 1.171813 |
| best product-like | 244 | -786.101081 | -2.942402 | 2.787532 | 2.957000 | 1.442055 | 1.112692 |
| best bond-forming | 607 | -784.837093 | -7.363281 | 1.188988 | 1.113508 | 2.195094 | 1.561758 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_attractive_product_a0.02_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.02_T300_s1000_zeroV_final.xyz`
- best_ts_like: `bias_1tBu_CO2_attractive_product_a0.02_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.02_T300_s1000_zeroV_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_attractive_product_a0.02_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.02_T300_s1000_zeroV_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_attractive_product_a0.02_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.02_T300_s1000_zeroV_best_bond_forming.xyz`
