# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_rc-md_product_k0006_T300_1000/bias_1tBu_CO2_rc-md_product_k0006_T300_1000.pdb`
- OUT: `bias_1tBu_CO2_rc-md_product_k0006_T300_1000/bias_1tBu_CO2_rc-md_product_k0006_T300_1000.out`
- Promising bond threshold met: `True`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -7.531779 | 0.000349 | 1.435839 | 1.663334 | 2.543384 | 2.218055 |
| best TS-like | 130 | -7.516136 | 0.013686 | 2.166881 | 2.208643 | 1.936491 | 2.049620 |
| best product-like | 346 | -7.494305 | 0.000609 | 1.459431 | 1.727748 | 2.455671 | 1.886351 |
| best bond-forming | 589 | -7.536496 | 0.000520 | 1.390067 | 1.293571 | 2.273022 | 2.289434 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_rc-md_product_k0006_T300_1000/bias_1tBu_CO2_rc-md_product_k0006_T300_1000_final.xyz`
- best_ts_like: `bias_1tBu_CO2_rc-md_product_k0006_T300_1000/bias_1tBu_CO2_rc-md_product_k0006_T300_1000_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_rc-md_product_k0006_T300_1000/bias_1tBu_CO2_rc-md_product_k0006_T300_1000_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_rc-md_product_k0006_T300_1000/bias_1tBu_CO2_rc-md_product_k0006_T300_1000_best_bond_forming.xyz`
