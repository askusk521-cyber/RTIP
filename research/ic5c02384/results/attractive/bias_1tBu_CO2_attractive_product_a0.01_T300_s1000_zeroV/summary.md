# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_attractive_product_a0.01_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T300_s1000_zeroV.pdb`
- OUT: `bias_1tBu_CO2_attractive_product_a0.01_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T300_s1000_zeroV.out`
- Promising bond threshold met: `True`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -785.854361 | -6.065306 | 1.548074 | 1.544349 | 2.103544 | 1.585717 |
| best TS-like | 279 | -786.465090 | -1.669424 | 3.136261 | 3.469968 | 1.509068 | 1.376189 |
| best product-like | 345 | -786.481769 | -2.074237 | 2.625782 | 3.275652 | 1.671793 | 1.203005 |
| best bond-forming | 916 | -785.755847 | -5.555821 | 1.215404 | 1.133859 | 2.150839 | 1.656525 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_attractive_product_a0.01_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T300_s1000_zeroV_final.xyz`
- best_ts_like: `bias_1tBu_CO2_attractive_product_a0.01_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T300_s1000_zeroV_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_attractive_product_a0.01_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T300_s1000_zeroV_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_attractive_product_a0.01_T300_s1000_zeroV/bias_1tBu_CO2_attractive_product_a0.01_T300_s1000_zeroV_best_bond_forming.xyz`
