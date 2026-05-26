# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_rc-md_product_k0010_T300_1000/bias_1tBu_CO2_rc-md_product_k0010_T300_1000.pdb`
- OUT: `bias_1tBu_CO2_rc-md_product_k0010_T300_1000/bias_1tBu_CO2_rc-md_product_k0010_T300_1000.out`
- Promising bond threshold met: `True`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -7.551050 | 0.052768 | 1.227012 | 2.873286 | 3.926805 | 4.749382 |
| best TS-like | 105 | -7.464710 | 0.018218 | 2.052200 | 1.981416 | 1.821244 | 1.939102 |
| best product-like | 103 | -7.476335 | 0.019678 | 2.096422 | 2.003034 | 1.822242 | 1.938580 |
| best bond-forming | 202 | -7.453269 | 0.007082 | 1.247784 | 2.073354 | 2.100451 | 2.041416 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_rc-md_product_k0010_T300_1000/bias_1tBu_CO2_rc-md_product_k0010_T300_1000_final.xyz`
- best_ts_like: `bias_1tBu_CO2_rc-md_product_k0010_T300_1000/bias_1tBu_CO2_rc-md_product_k0010_T300_1000_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_rc-md_product_k0010_T300_1000/bias_1tBu_CO2_rc-md_product_k0010_T300_1000_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_rc-md_product_k0010_T300_1000/bias_1tBu_CO2_rc-md_product_k0010_T300_1000_best_bond_forming.xyz`
