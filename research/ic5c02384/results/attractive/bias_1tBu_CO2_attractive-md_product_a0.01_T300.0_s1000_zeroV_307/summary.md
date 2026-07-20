# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_307/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_307.pdb`
- OUT: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_307/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_307.out`
- Promising bond threshold met: `False`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -6.928864 | -1.628987 | 2.779247 | 3.292657 | 2.812572 | 2.050227 |
| best TS-like | 282 | -7.268576 | -1.688898 | 3.163163 | 3.422350 | 1.506347 | 1.388155 |
| best product-like | 331 | -7.253512 | -1.606401 | 2.702462 | 3.163164 | 1.604562 | 1.199179 |
| best bond-forming | 849 | -7.022023 | -1.509152 | 2.709820 | 2.389584 | 2.335697 | 1.523451 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_307/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_307_final.xyz`
- best_ts_like: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_307/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_307_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_307/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_307_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_307/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_307_best_bond_forming.xyz`
