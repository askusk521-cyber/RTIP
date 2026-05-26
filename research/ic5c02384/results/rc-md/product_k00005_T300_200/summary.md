# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_rc-md_product_k00005_T300_200/bias_1tBu_CO2_rc-md_product_k00005_T300_200.pdb`
- OUT: `bias_1tBu_CO2_rc-md_product_k00005_T300_200/bias_1tBu_CO2_rc-md_product_k00005_T300_200.out`
- Promising bond threshold met: `False`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 200 | -7.581677 | 0.007850 | 3.752085 | 3.407329 | 2.427672 | 2.517993 |
| best TS-like | 104 | -7.590496 | 0.012034 | 4.098191 | 3.483791 | 2.353018 | 2.434864 |
| best product-like | 97 | -7.586783 | 0.012348 | 4.083044 | 3.483744 | 2.353531 | 2.434237 |
| best bond-forming | 200 | -7.581677 | 0.007850 | 3.752085 | 3.407329 | 2.427672 | 2.517993 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_rc-md_product_k00005_T300_200/bias_1tBu_CO2_rc-md_product_k00005_T300_200_final.xyz`
- best_ts_like: `bias_1tBu_CO2_rc-md_product_k00005_T300_200/bias_1tBu_CO2_rc-md_product_k00005_T300_200_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_rc-md_product_k00005_T300_200/bias_1tBu_CO2_rc-md_product_k00005_T300_200_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_rc-md_product_k00005_T300_200/bias_1tBu_CO2_rc-md_product_k00005_T300_200_best_bond_forming.xyz`
