# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_rc-md_ts_k00005_T300_200/bias_1tBu_CO2_rc-md_ts_k00005_T300_200.pdb`
- OUT: `bias_1tBu_CO2_rc-md_ts_k00005_T300_200/bias_1tBu_CO2_rc-md_ts_k00005_T300_200.out`
- Promising bond threshold met: `False`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 200 | -7.586533 | 0.004807 | 4.208026 | 3.464244 | 2.636805 | 2.683935 |
| best TS-like | 1 | -7.598669 | 0.008299 | 4.094638 | 3.316628 | 2.363878 | 2.469147 |
| best product-like | 1 | -7.598669 | 0.008299 | 4.094638 | 3.316628 | 2.363878 | 2.469147 |
| best bond-forming | 19 | -7.587883 | 0.007983 | 4.059646 | 3.315938 | 2.379019 | 2.484066 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_rc-md_ts_k00005_T300_200/bias_1tBu_CO2_rc-md_ts_k00005_T300_200_final.xyz`
- best_ts_like: `bias_1tBu_CO2_rc-md_ts_k00005_T300_200/bias_1tBu_CO2_rc-md_ts_k00005_T300_200_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_rc-md_ts_k00005_T300_200/bias_1tBu_CO2_rc-md_ts_k00005_T300_200_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_rc-md_ts_k00005_T300_200/bias_1tBu_CO2_rc-md_ts_k00005_T300_200_best_bond_forming.xyz`
