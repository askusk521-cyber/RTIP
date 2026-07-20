# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_attractive_product_a0.005_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.005_T300_s1000_zeroV.pdb`
- OUT: `bias_1tBu_CO2_attractive_product_a0.005_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.005_T300_s1000_zeroV.out`
- Promising bond threshold met: `False`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -786.025266 | -3.032644 | 2.878870 | 2.836232 | 1.734148 | 1.163376 |
| best TS-like | 963 | -786.126271 | -2.920411 | 2.553008 | 3.977130 | 1.528287 | 1.029918 |
| best product-like | 895 | -786.046441 | -2.714222 | 1.931800 | 2.085803 | 1.705395 | 0.486266 |
| best bond-forming | 883 | -785.829329 | -2.677825 | 1.687579 | 2.129583 | 2.010613 | 0.787219 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_attractive_product_a0.005_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.005_T300_s1000_zeroV_final.xyz`
- best_ts_like: `bias_1tBu_CO2_attractive_product_a0.005_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.005_T300_s1000_zeroV_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_attractive_product_a0.005_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.005_T300_s1000_zeroV_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_attractive_product_a0.005_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.005_T300_s1000_zeroV_best_bond_forming.xyz`
