# Bias Exploration Summary

- PDB: `bias_1tBu_CO2_rc-md_product_k0002_T300_1000/bias_1tBu_CO2_rc-md_product_k0002_T300_1000.pdb`
- OUT: `bias_1tBu_CO2_rc-md_product_k0002_T300_1000/bias_1tBu_CO2_rc-md_product_k0002_T300_1000.out`
- Promising bond threshold met: `True`

| frame | step | pot_real / Ha | pot_bias / Ha | min(B-O) / A | min(N-C) / A | TS core RMSD / A | product core RMSD / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 1000 | -7.574629 | 0.025849 | 3.438813 | 3.301384 | 2.972965 | 2.349237 |
| best TS-like | 167 | -7.581671 | 0.013218 | 2.893594 | 2.805219 | 2.150007 | 2.258479 |
| best product-like | 163 | -7.582702 | 0.014293 | 2.957253 | 2.849017 | 2.150591 | 2.258124 |
| best bond-forming | 229 | -7.545899 | 0.005314 | 2.418310 | 2.280482 | 2.288861 | 2.339670 |

Reference distances:

| reference | min(B-O) / A | min(N-C) / A |
|---|---:|---:|
| TS | 2.707228 | 1.695627 |
| product | 1.488795 | 1.491083 |

Frame files:

- final: `bias_1tBu_CO2_rc-md_product_k0002_T300_1000/bias_1tBu_CO2_rc-md_product_k0002_T300_1000_final.xyz`
- best_ts_like: `bias_1tBu_CO2_rc-md_product_k0002_T300_1000/bias_1tBu_CO2_rc-md_product_k0002_T300_1000_best_ts_like.xyz`
- best_product_like: `bias_1tBu_CO2_rc-md_product_k0002_T300_1000/bias_1tBu_CO2_rc-md_product_k0002_T300_1000_best_product_like.xyz`
- best_bond_forming: `bias_1tBu_CO2_rc-md_product_k0002_T300_1000/bias_1tBu_CO2_rc-md_product_k0002_T300_1000_best_bond_forming.xyz`
