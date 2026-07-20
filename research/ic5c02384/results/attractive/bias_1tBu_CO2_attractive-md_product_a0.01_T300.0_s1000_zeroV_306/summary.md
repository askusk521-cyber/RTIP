# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_306/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_306.pdb`
- OUT: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_306/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_306.out`
- Promising bond threshold met: `False`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -6.614532 | -1.946963 | 1.982470 | 2.235607 | 2.236585 | 1.532916 |
| best TS-like | 282 | -7.268576 | -1.688898 | 3.163163 | 3.422350 | 1.506347 | 1.388155 |
| best product-like | 333 | -7.256668 | -1.931885 | 2.672691 | 3.182677 | 1.609916 | 1.198161 |
| best bond-forming | 867 | -6.842133 | -1.946960 | 1.077295 | 2.933687 | 2.106502 | 1.525343 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_306/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_306_final.xyz`
- best_ts_like: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_306/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_306_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_306/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_306_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_306/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_306_best_bond_forming.xyz`
