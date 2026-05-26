# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_rc-md_ts_k0002_T300_1000/bias_1tBu_CO2_rc-md_ts_k0002_T300_1000.pdb`
- OUT: `bias_1tBu_CO2_rc-md_ts_k0002_T300_1000/bias_1tBu_CO2_rc-md_ts_k0002_T300_1000.out`
- Promising bond threshold met: `True`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -7.585192 | 0.008027 | 3.446690 | 2.999827 | 2.531404 | 1.662455 |
| best TS-like | 1 | -7.598669 | 0.033194 | 4.094638 | 3.316628 | 2.363878 | 2.469147 |
| best product-like | 833 | -7.573699 | 0.008830 | 3.456717 | 2.866438 | 2.758667 | 1.606234 |
| best bond-forming | 259 | -7.575695 | 0.002102 | 2.737750 | 2.462071 | 2.630899 | 2.466953 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_rc-md_ts_k0002_T300_1000/bias_1tBu_CO2_rc-md_ts_k0002_T300_1000_final.xyz`
- best_ts_like: `bias_1tBu_CO2_rc-md_ts_k0002_T300_1000/bias_1tBu_CO2_rc-md_ts_k0002_T300_1000_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_rc-md_ts_k0002_T300_1000/bias_1tBu_CO2_rc-md_ts_k0002_T300_1000_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_rc-md_ts_k0002_T300_1000/bias_1tBu_CO2_rc-md_ts_k0002_T300_1000_best_bond_forming.xyz`
