# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_attractive-md_product_a0.002_T300.0_s1000_zeroV_297/bias_1tBu_CO2_attractive-md_product_a0.002_T300.0_s1000_zeroV_297.pdb`
- OUT: `bias_1tBu_CO2_attractive-md_product_a0.002_T300.0_s1000_zeroV_297/bias_1tBu_CO2_attractive-md_product_a0.002_T300.0_s1000_zeroV_297.out`
- Promising bond threshold met: `False`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -7.529965 | -1.209830 | 3.144148 | 4.006956 | 3.368231 | 1.914540 |
| best TS-like | 452 | -7.542444 | -0.532476 | 4.397183 | 4.863531 | 1.982017 | 2.017888 |
| best product-like | 953 | -7.511476 | -1.153567 | 3.921033 | 3.755909 | 3.150680 | 1.839824 |
| best bond-forming | 916 | -7.477166 | -1.109167 | 4.128976 | 2.963709 | 2.763236 | 2.128426 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_attractive-md_product_a0.002_T300.0_s1000_zeroV_297/bias_1tBu_CO2_attractive-md_product_a0.002_T300.0_s1000_zeroV_297_final.xyz`
- best_ts_like: `bias_1tBu_CO2_attractive-md_product_a0.002_T300.0_s1000_zeroV_297/bias_1tBu_CO2_attractive-md_product_a0.002_T300.0_s1000_zeroV_297_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_attractive-md_product_a0.002_T300.0_s1000_zeroV_297/bias_1tBu_CO2_attractive-md_product_a0.002_T300.0_s1000_zeroV_297_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_attractive-md_product_a0.002_T300.0_s1000_zeroV_297/bias_1tBu_CO2_attractive-md_product_a0.002_T300.0_s1000_zeroV_297_best_bond_forming.xyz`
