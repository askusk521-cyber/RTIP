# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_rc-md_product_k0001_T300_1000/bias_1tBu_CO2_rc-md_product_k0001_T300_1000.pdb`
- OUT: `bias_1tBu_CO2_rc-md_product_k0001_T300_1000/bias_1tBu_CO2_rc-md_product_k0001_T300_1000.out`
- Promising bond threshold met: `False`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -7.592860 | 0.013700 | 3.738998 | 3.088263 | 2.914257 | 2.157687 |
| best TS-like | 162 | -7.586378 | 0.013953 | 3.626739 | 3.292173 | 2.286072 | 2.385773 |
| best product-like | 892 | -7.563992 | 0.009434 | 3.153819 | 1.969827 | 2.632684 | 2.046712 |
| best bond-forming | 888 | -7.564365 | 0.009591 | 3.156669 | 1.965370 | 2.635938 | 2.047547 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_rc-md_product_k0001_T300_1000/bias_1tBu_CO2_rc-md_product_k0001_T300_1000_final.xyz`
- best_ts_like: `bias_1tBu_CO2_rc-md_product_k0001_T300_1000/bias_1tBu_CO2_rc-md_product_k0001_T300_1000_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_rc-md_product_k0001_T300_1000/bias_1tBu_CO2_rc-md_product_k0001_T300_1000_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_rc-md_product_k0001_T300_1000/bias_1tBu_CO2_rc-md_product_k0001_T300_1000_best_bond_forming.xyz`
