# ic5c02384 Reaction and Validation Reference

This document summarizes the main reactions, computational boundary
conditions, energy definitions, and directly verifiable local data from the
paper **Predicting Dinitrogen Activation and Coupling with Carbon Dioxide and
Other Small Molecules by Methyleneborane: A Combined DFT and Machine Learning
Study**.

Chinese companion: [reaction_reference.md](reaction_reference.md).

## Sources

- Main PDF:
  `predicting-dinitrogen-activation-and-coupling-with-carbon-dioxide-and-other-small-molecules-by-methyleneborane-a.pdf`
- SI coordinates: `ic5c02384_si_002.xyz`
- Prepared data: `research/ic5c02384/`
- Structure manifest: `research/ic5c02384/manifest.csv`
- Reaction manifest: `research/ic5c02384/reaction_manifest.csv`

The abstract and a few text locations mention sulfur dioxide, but Figure 1/7,
Table 4, the SI coordinates, and local data files use `CS2`. Use `CS2` from the
SI filenames and `reaction_manifest.csv` for validation.

## Literature Computational Conditions

- Geometry optimization: gas phase, `omegaB97X-D/cc-pVDZ`, same level for all
  atoms.
- Frequency calculation: same level; minima have no imaginary frequency and
  TS structures have one imaginary frequency.
- IRC: TS structures were checked by IRC to connect two minima.
- Single-point energies: `omegaB97X-D/cc-pVTZ`.
- Reported Gibbs free energies: `cc-pVTZ` single-point energies plus
  `cc-pVDZ` Gibbs corrections.
- Temperature: `298 K`.
- Constraints: none.
- Grid: Gaussian 16 ultrafine grid.
- Additional analysis: Multiwfn FBO, NBO plus PIO, and EDA through
  distortion/interaction or activation-strain analysis.

## Reaction Overview

The core reaction is coupling of an already activated `N2` complex with a small
molecule:

```text
methyleneborane-N2 complex + X=C-R -> coupling product
```

Common product features:

- The terminal N of `N2` forms a new `N-C` bond to the small-molecule carbon.
- The small-molecule `X` atom donates to the boron center and forms a `B-X`
  interaction or bond.
- For `CO2`, the key distances are `B-O` and `N-C(CO2)`.
- For `CS2`, `H2CO`, `MeCN`, and `MeCH=NMe`, track analogous `B-X` and `N-C`
  distances with `X = S/O/N`.

## Main Paper Table 1: `1-R + CO2` Products

These values are paper Gibbs free energies and key product bond lengths, not SI
title electronic energies.

| Reaction | B-N / A | N-N / A | B-O / A | C-O / A | C-N2 / A | Delta G / kcal mol^-1 |
|---|---:|---:|---:|---:|---:|---:|
| `1-CN + CO2 -> 1-CN-CO2.P` | 1.564 | 1.238 | 1.460 | 1.341 | 1.494 | 2.0 |
| `1-H + CO2 -> 1-H-CO2.P` | 1.552 | 1.242 | 1.473 | 1.335 | 1.491 | 0.3 |
| `1-Me + CO2 -> 1-Me-CO2.P` | 1.557 | 1.241 | 1.486 | 1.332 | 1.492 | -6.5 |
| `1-Ph + CO2 -> 1-Ph-CO2.P` | 1.554 | 1.240 | 1.482 | 1.333 | 1.493 | -3.1 |
| `1-PMe2 + CO2 -> 1-PMe2-CO2.P` | 1.540 | 1.245 | 1.475 | 1.338 | 1.485 | -6.3 |
| `1-SiMe3 + CO2 -> 1-SiMe3-CO2.P` | 1.527 | 1.249 | 1.482 | 1.339 | 1.478 | -1.3 |
| `1-tBu + CO2 -> 1-tBu-CO2.P` | 1.553 | 1.242 | 1.489 | 1.331 | 1.491 | -10.1 |

Conclusion: `1-tBu` is most favorable by reported Gibbs free energy; `1-Me` and
`1-PMe2` are also clearly exergonic. `1-CN` and `1-H` are nearly thermoneutral
or slightly endergonic.

## Main Paper Figure 6: `1-R + CO2` Gibbs Profiles

Values are relative to separated `1-R + CO2`, in `kcal mol^-1`.

| Reaction | INT Delta G | TS Delta G | P Delta G | Validation priority |
|---|---:|---:|---:|---|
| `1-tBu + CO2` | 3.9 | 25.1 | -10.1 | high; representative lowest-barrier and most exergonic case |
| `1-Me + CO2` | 4.1 | 26.9 | -6.5 | high; substituent trend |
| `1-PMe2 + CO2` | -0.2 | 28.9 | -6.3 | high; substituent trend |
| `1-Ph + CO2` | 4.4 | 29.6 | -3.1 | medium |
| `1-SiMe3 + CO2` | 4.2 | 32.9 | -1.3 | medium; highest barrier |

EDA electronic-energy decomposition for `1-R-CO2.TS`:

| TS | Delta Edeform(1-R) | Delta Edeform(CO2) | Delta Eb | Delta E |
|---|---:|---:|---:|---:|
| `1-Me-CO2.TS` | 21.4 | 33.4 | -38.5 | 16.2 |
| `1-Ph-CO2.TS` | 22.3 | 33.9 | -37.6 | 18.6 |
| `1-PMe2-CO2.TS` | 21.0 | 35.1 | -38.5 | 17.6 |
| `1-SiMe3-CO2.TS` | 29.8 | 39.4 | -47.5 | 21.8 |
| `1-tBu-CO2.TS` | 24.0 | 36.8 | -47.7 | 13.1 |

## Main Paper Figure 3: NHC Backbone Effects

Reported Gibbs reaction energies for `n-tBu + CO2`:

| Reaction | Delta G / kcal mol^-1 | Note |
|---|---:|---|
| `2-tBu + CO2 -> 2-tBu-CO2.P` | -2.7 | CAAC type, less favorable |
| `3-tBu + CO2 -> 3-tBu-CO2.P` | -0.7 | CAAC type, less favorable |
| `4-tBu + CO2 -> 4-tBu-CO2.P` | -6.7 | CDAC type, more favorable |
| `5-tBu + CO2 -> 5-tBu-CO2.P` | 2.8 | CAAC type, endergonic |
| `6-tBu + CO2 -> 6-tBu-CO2.P` | 1.5 | CAAC type, endergonic |

The paper concludes that CDAC is generally more favorable than CAAC. CAAC acts
as both sigma donor and pi acceptor, competing with the donor/acceptor role of
`CO2`.

## Main Paper Figure 7: `1-tBu` and Other Small Molecules

Values are relative to separated reactants, in `kcal mol^-1`.

| Reaction | INT Delta G | TS Delta G | P Delta G | Key validation metrics |
|---|---:|---:|---:|---|
| `1-tBu + H2CO -> 1-tBu-H2CO.P` | 5.2 | 11.9 | -32.7 | `B-O`, `N-C(H2CO)` |
| `1-tBu + CS2 -> 1-tBu-CS2.P` | 5.0 | 19.2 | -8.1 | `B-S`, `N-C(CS2)` |
| `1-tBu + CO2 -> 1-tBu-CO2.P` | 3.9 | 25.1 | -10.1 | `B-O`, `N-C(CO2)` |
| `1-tBu + MeCN -> 1-tBu-MeCN.P` | 0.7 | 25.7 | -15.7 | `B-N`, `N-C(MeCN)` |
| `1-tBu + MeCH=NMe -> 1-tBu-MeCH=NMe.P` | 3.4 | 27.5 | -14.0 | `B-N`, `N-C(imine)` |

Barrier order:

```text
H2CO < CS2 < CO2 < MeCN < MeCH=NMe
```

EDA electronic-energy decomposition:

| TS | Delta Edeform(1-tBu) | Delta Edeform(small molecule) | Delta Eb | Delta E |
|---|---:|---:|---:|---:|
| `1-tBu-CO2.TS` | 24.0 | 36.8 | -47.7 | 13.1 |
| `1-tBu-CS2.TS` | 9.4 | 15.5 | -16.8 | 8.1 |
| `1-tBu-H2CO.TS` | 13.6 | 4.3 | -19.6 | -1.8 |
| `1-tBu-MeCN.TS` | 19.7 | 9.4 | -15.7 | 13.3 |
| `1-tBu-MeCH=NMe.TS` | 22.0 | 9.0 | -18.8 | 12.1 |

## Mechanistic and Machine-Learning Notes

- PIO analysis says the main interaction in `1-tBu-CO2.P` is donation from the
  terminal N2 nitrogen p orbital into the CO2 pi* orbital; PBI `0.97`,
  contribution about `49.4%`.
- The second interaction is CO2 oxygen lone-pair sigma donation to the empty B
  p orbital; PBI `0.65`, contribution about `33.1%`.
- The third is N2 pi to C-O pi* back-donation; PBI `0.10`, contribution about
  `5.3%`.
- Machine-learning descriptors include `HOMO-LUMO gap`, `NPA(B)`, `NPA(N1)`,
  `NPA(N2)`, and `Delta FBO`.
- Symbolic regression RMSE is `1.9`, lower than linear regression at `2.2`,
  supporting nonlinear relationships.
- Increasing `HOMO-LUMO gap`, B charge, or `Delta FBO` lowers the reaction
  energy; increasing N charge raises it.
- `1-tBu` has the lowest CO2 coupling barrier, attributed to strongest binding
  between deformed reactants in the TS and linked to higher fragment HOMO
  energy.
- For small-molecule activation, the H2CO vs MeCH=NMe barrier difference mainly
  comes from reactant deformation energy rather than binding energy.

## Local SI Electronic-Energy Reaction Manifest

Values in `reaction_manifest.csv` come from SI XYZ title electronic energies
(`E = ... a.u.`), not paper Gibbs free energies. They are suitable for DeePMD
single-point trend checks but should not be directly mixed with paper
`Delta G`.

### Selected RTIP/JAX Reaction Directories

| Directory | References | SI product Delta E / kcal mol^-1 | SI TS Delta E / kcal mol^-1 | Suggested use |
|---|---|---:|---:|---|
| `1-tBu__CO2` | product + TS | -31.05 | 10.11 | first priority; already validated on 20260524 |
| `1-Me__CO2` | product + TS | -26.10 | 13.86 | substituent trend |
| `1-PMe2__CO2` | product + TS | -26.76 | 14.21 | substituent trend |
| `1-Ph__CO2` | product + TS | -21.99 | 16.24 | medium CO2 system |
| `1-SiMe3__CO2` | product + TS | -21.35 | 18.54 | high-barrier CO2 system |
| `1-H__CO2` | product only | -18.83 | n/a | endpoint validation |
| `1-CN__CO2` | product only | -17.20 | n/a | endpoint validation |
| `1-tBu__H2CO` | product + TS | -19.60 | -4.68 | easiest small-molecule channel |
| `1-tBu__CS2` | product + TS | -16.80 | 5.09 | small-molecule comparison |
| `1-tBu__MeCN` | product + TS | -15.70 | 10.19 | small-molecule comparison |
| `1-tBu__MeCH=NMe` | product + TS | -18.80 | 8.34 | small-molecule comparison |

### CO2 Product Electronic-Energy Matrix

Units are `kcal mol^-1` relative to separated `n-R + CO2`.

| family | CN | H | Me | Ph | PMe2 | SiMe3 | tBu |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1-R` | -17.20 | -18.83 | -26.10 | -21.99 | -26.76 | -21.35 | -31.05 |
| `2-R` | -10.28 | -10.65 | -16.85 | -15.61 | -14.06 | -13.97 | -22.87 |
| `3-R` | -6.66 | -7.81 | -13.40 | -13.20 | -11.81 | -11.96 | -20.00 |
| `4-R` | -16.78 | -20.26 | -24.28 | -21.01 | -20.92 | -19.50 | -28.09 |
| `5-R` | -8.83 | -9.95 | -14.46 | -13.18 | -11.34 | -11.37 | -16.41 |
| `6-R` | -9.57 | -11.28 | -14.68 | -12.68 | -13.00 | -12.35 | -19.33 |

## RTIP/JAX Validation Boundary Conditions

### Input Boundary

- Reactants: `research/ic5c02384/reactions/<reaction>/1.xyz` and `2.xyz`.
- Initial separated structure: generated by `rtip-jax synthesize --dist 5.0
  --seed 0`.
- References: use `product.xyz` and `ts.xyz` where available; product-only
  systems should use endpoint energy and geometry validation.
- Environment: gas-phase cluster, non-periodic, no solvent, no pressure/cell.
- Elements: `B, C, H, N, O, P, S, Si`.
- DeePMD model: must cover those elements; 20260524 used
  `/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt`.

### Energy Terms

- `pot_real`: real PES energy; prioritize this for reaction trends.
- `pot_total`: includes bias and is not a real reaction energy.
- Paper `Delta G`: DFT Gibbs free energy.
- Local `reaction_manifest.csv`: SI title electronic-energy differences.

### Geometry Metrics

- Final or best-frame `pot_real` relative to initial reactants.
- All-atom RMSD and reactive-core RMSD to `product.xyz` and `ts.xyz`.
- Bond-forming distances: `min(B-X)` and `min(N-C_small)`.
- For CO2: `min(B-O)` and `min(N-CO2(C))`.
- For CS2: `min(B-S)` and `min(N-CS2(C))`.
- For H2CO: `min(B-O)` and `min(N-H2CO(C))`.
- For MeCN: `min(B-N_small)` and `min(N-MeCN(C))`.
- For MeCH=NMe: `min(B-N_small)` and `min(N-imine(C))`.

Atom order can differ between `product.xyz`, `ts.xyz`, and run outputs. Use
element/fragment-aware reordering or Kabsch-aligned reactive-core RMSD; minimum
distances are more robust for bond metrics.

### Recommended Validation Order

1. `1-tBu__CO2`: baseline system with existing long records.
2. `1-tBu__H2CO`: lowest reported Gibbs barrier.
3. `1-Me__CO2` and `1-PMe2__CO2`: substituent comparisons.
4. `1-tBu__CS2`: small-molecule comparison with `B-S`.
5. `1-Ph__CO2`, `1-SiMe3__CO2`, `1-tBu__MeCN`, `1-tBu__MeCH=NMe`: higher or
   harder barriers.
6. `1-CN__CO2`, `1-H__CO2`, and product-only `2-R` to `6-R` series: endpoint
   checks first.

## Minimal Command Template

```bash
rtip-jax synthesize \
  --inputs research/ic5c02384/reactions/1-tBu__CO2/1.xyz research/ic5c02384/reactions/1-tBu__CO2/2.xyz \
  --output research/ic5c02384/reactions/1-tBu__CO2/IS.xyz \
  --dist 5.0 \
  --seed 0
```

If following the 20260524 RC-MD validation style, create a separate output
directory per reaction and record `summary.md`, `summary.json`, best frames, and
key-distance tables in the work log.

## Main Conclusions

- The paper is theoretical prediction, not experimental synthesis.
- `1-tBu + CO2` is the best first validation target: most stable reported Gibbs
  product, lowest CO2 barrier, and both product and TS references in the SI.
- `H2CO` has the lowest small-molecule Gibbs barrier and may be easier than CO2
  in biased MD.
- Coupling is described as a concerted step: terminal N2 nitrogen forms N-C,
  while small-molecule X forms B-X.
- Always distinguish paper Gibbs free energy, SI electronic energy, DeePMD
  `pot_real`, and bias-included `pot_total`.
