# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_attractive_product_a0.01_T300_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T300_s2000_zeroV.pdb`
- OUT: `bias_1tBu_CO2_attractive_product_a0.01_T300_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T300_s2000_zeroV.out`
- Promising bond threshold met: `True`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 2000 | -786.398431 | -12.130613 | 1.744566 | 1.323361 | 2.266449 | 1.797608 |
| best TS-like | 279 | -786.465090 | -1.669424 | 3.136261 | 3.469968 | 1.509068 | 1.376189 |
| best product-like | 345 | -786.481769 | -2.074237 | 2.625782 | 3.275652 | 1.671793 | 1.203005 |
| best bond-forming | 916 | -785.757064 | -5.555821 | 1.215359 | 1.135005 | 2.150132 | 1.655608 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_attractive_product_a0.01_T300_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T300_s2000_zeroV_final.xyz`
- best_ts_like: `bias_1tBu_CO2_attractive_product_a0.01_T300_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T300_s2000_zeroV_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_attractive_product_a0.01_T300_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T300_s2000_zeroV_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_attractive_product_a0.01_T300_s2000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T300_s2000_zeroV_best_bond_forming.xyz`
