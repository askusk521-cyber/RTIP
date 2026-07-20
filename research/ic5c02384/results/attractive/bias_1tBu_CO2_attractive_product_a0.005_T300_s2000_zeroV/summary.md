# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_attractive_product_a0.005_T300_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.005_T300_s2000_zeroV.pdb`
- OUT: `bias_1tBu_CO2_attractive_product_a0.005_T300_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.005_T300_s2000_zeroV.out`
- Promising bond threshold met: `True`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 2000 | -786.147665 | -6.065307 | 2.366959 | 1.690230 | 1.709576 | 0.402217 |
| best TS-like | 1602 | -785.837404 | -4.858310 | 1.722768 | 1.510849 | 1.244358 | 0.455757 |
| best product-like | 1724 | -785.998486 | -5.228294 | 1.442426 | 1.574277 | 1.470962 | 0.192095 |
| best bond-forming | 1882 | -785.974476 | -5.707453 | 1.255753 | 1.051915 | 1.401949 | 0.469003 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_attractive_product_a0.005_T300_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.005_T300_s2000_zeroV_final.xyz`
- best_ts_like: `bias_1tBu_CO2_attractive_product_a0.005_T300_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.005_T300_s2000_zeroV_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_attractive_product_a0.005_T300_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.005_T300_s2000_zeroV_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_attractive_product_a0.005_T300_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.005_T300_s2000_zeroV_best_bond_forming.xyz`
