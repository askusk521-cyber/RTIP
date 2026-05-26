# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_attractive-md_product_a00005_T300_200/bias_1tBu_CO2_attractive-md_product_a00005_T300_200.pdb`
- OUT: `bias_1tBu_CO2_attractive-md_product_a00005_T300_200/bias_1tBu_CO2_attractive-md_product_a00005_T300_200.out`
- Promising bond threshold met: `False`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 200 | -7.581495 | -0.056962 | 4.278823 | 3.787126 | 2.531642 | 2.617006 |
| best TS-like | 7 | -7.594915 | -0.001995 | 4.086706 | 3.305639 | 2.361180 | 2.465497 |
| best product-like | 12 | -7.592573 | -0.003419 | 4.079156 | 3.305086 | 2.362203 | 2.465166 |
| best bond-forming | 14 | -7.591501 | -0.003988 | 4.074831 | 3.308112 | 2.363567 | 2.465993 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_attractive-md_product_a00005_T300_200/bias_1tBu_CO2_attractive-md_product_a00005_T300_200_final.xyz`
- best_ts_like: `bias_1tBu_CO2_attractive-md_product_a00005_T300_200/bias_1tBu_CO2_attractive-md_product_a00005_T300_200_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_attractive-md_product_a00005_T300_200/bias_1tBu_CO2_attractive-md_product_a00005_T300_200_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_attractive-md_product_a00005_T300_200/bias_1tBu_CO2_attractive-md_product_a00005_T300_200_best_bond_forming.xyz`
