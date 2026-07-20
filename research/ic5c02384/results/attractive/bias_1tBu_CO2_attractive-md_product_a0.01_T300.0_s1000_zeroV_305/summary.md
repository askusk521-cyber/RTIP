# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_305/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_305.pdb`
- OUT: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_305/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_305.out`
- Promising bond threshold met: `True`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -6.552365 | -6.065306 | 1.738769 | 2.956482 | 2.226601 | 1.684646 |
| best TS-like | 282 | -7.268576 | -1.688898 | 3.163163 | 3.422350 | 1.506347 | 1.388155 |
| best product-like | 333 | -7.256152 | -2.004137 | 2.672691 | 3.182677 | 1.609916 | 1.198161 |
| best bond-forming | 647 | -5.998341 | -3.924246 | 0.892496 | 1.502122 | 2.149460 | 1.822162 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_305/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_305_final.xyz`
- best_ts_like: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_305/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_305_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_305/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_305_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_305/bias_1tBu_CO2_attractive-md_product_a0.01_T300.0_s1000_zeroV_305_best_bond_forming.xyz`
