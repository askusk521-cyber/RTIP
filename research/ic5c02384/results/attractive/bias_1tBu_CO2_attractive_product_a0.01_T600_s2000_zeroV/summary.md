# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_attractive_product_a0.01_T600_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T600_s2000_zeroV.pdb`
- OUT: `bias_1tBu_CO2_attractive_product_a0.01_T600_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T600_s2000_zeroV.out`
- Promising bond threshold met: `True`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 2000 | -786.474633 | -12.130613 | 1.547767 | 1.338466 | 2.369675 | 1.954469 |
| best TS-like | 282 | -786.430174 | -1.687939 | 3.264142 | 3.585033 | 1.552573 | 1.394822 |
| best product-like | 631 | -785.478891 | -3.827201 | 1.222482 | 1.267311 | 1.858899 | 1.151589 |
| best bond-forming | 592 | -785.555031 | -3.590655 | 1.194255 | 1.056684 | 2.472164 | 2.044100 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_attractive_product_a0.01_T600_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T600_s2000_zeroV_final.xyz`
- best_ts_like: `bias_1tBu_CO2_attractive_product_a0.01_T600_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T600_s2000_zeroV_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_attractive_product_a0.01_T600_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T600_s2000_zeroV_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_attractive_product_a0.01_T600_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T600_s2000_zeroV_best_bond_forming.xyz`
