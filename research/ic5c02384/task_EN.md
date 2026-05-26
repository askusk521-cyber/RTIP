# ic5c02384 Task Summary

## Literature Topic

The paper is **Predicting Dinitrogen Activation and Coupling with Carbon
Dioxide and Other Small Molecules by Methyleneborane: A Combined DFT and
Machine Learning Study**.

Its central goal is to use DFT calculations and machine-learning analysis to
predict whether activated `methyleneborane-N2` complexes can couple with `CO2`
and other small molecules, forming new N-C bonds or related polar-bond
activation products.

This is a theoretical prediction study, not an experimental synthesis report.
The authors focus on the potential of main-group boron species in N2 coupling
chemistry and try to extend N2/CO2 coupling beyond transition-metal systems.

## Main Reactions

The core reaction can be summarized as:

```text
methyleneborane-N2 complex + CO2 -> N2-CO2 coupling product
```

The terminal nitrogen of the activated N2 fragment forms an N-C bond to the CO2
carbon center, while an oxygen atom of CO2 donates to the boron center to form a
B-O interaction. The paper treats the N2/CO2 coupling as a concerted step.

The substituents compared for `1-R + CO2` are
`R = CN, H, Me, Ph, PMe2, SiMe3, tBu`. The paper's Gibbs free energies show
`1-tBu` as the most favorable case, with
`1-tBu + CO2 -> 1-tBu-CO2.P` at about `-10.1 kcal/mol`. Other favorable systems
include `1-Me` and `1-PMe2`.

The paper also studies `1-tBu` coupling or activation with other small
molecules:

```text
1-tBu + CS2 -> 1-tBu-CS2.P
1-tBu + H2CO -> 1-tBu-H2CO.P
1-tBu + MeCN -> 1-tBu-MeCN.P
1-tBu + MeCH=NMe -> 1-tBu-MeCH=NMe.P
```

The Gibbs barrier trend is approximately:

```text
H2CO < CS2 < CO2 < MeCN < MeCH=NMe
```

The local SI coordinates and manifests use the `CS2` name; follow
`reaction_manifest.csv` and structure filenames.

## Data in This Directory

This directory was converted from ACS Supporting Information
`ic5c02384_si_002.xyz` and organized as an RTIP/JAX-readable dataset.

- `structures/`: 104 single-structure XYZ files.
- `manifest.csv`: structure index, atom count, name, electronic energy from SI
  titles, and path.
- `reaction_manifest.csv`: reaction-organized reactants, products, transition
  states, and relative energies based on SI electronic energies.
- `reactions/*/1.xyz` and `reactions/*/2.xyz`: the two reactants for each
  reaction, suitable for `rtip-jax synthesize --inputs`.
- `reactions/*/product.xyz`: product reference structures from the SI.
- `reactions/*/ts.xyz`: transition-state reference structures from the SI.
- `elements_present.txt`: elements are `B, C, H, N, O, P, S, Si`.

## Suggested Computation Plan

Start with `1-tBu__CO2`. It is the representative low-energy CO2 coupling case
in the paper and has both product and transition-state reference structures.

Recommended order:

1. Use `rtip-jax synthesize` to generate an initial separated `IS.xyz` for
   `1-tBu + CO2`.
2. Use `IS.xyz`, `product.xyz`, and `ts.xyz` to test whether RTIP/JAX or
   follow-up pathway tools can construct a reasonable reaction path.
3. Compare against electronic-energy relative values in `reaction_manifest.csv`
   to screen reactions that are easier to converge or more representative.
4. Extend to `1-Me__CO2`, `1-PMe2__CO2`, `1-Ph__CO2`, and `1-SiMe3__CO2` to
   compare substituent effects.
5. Compare `1-tBu__H2CO`, `1-tBu__CS2`, `1-tBu__MeCN`, and
   `1-tBu__MeCH=NMe` to reproduce the small-molecule activation order.

## Priority

High priority:

- `reactions/1-tBu__CO2`
- `reactions/1-Me__CO2`
- `reactions/1-PMe2__CO2`
- `reactions/1-tBu__H2CO`

Medium priority:

- `reactions/1-Ph__CO2`
- `reactions/1-SiMe3__CO2`
- `reactions/1-tBu__CS2`
- `reactions/1-tBu__MeCN`
- `reactions/1-tBu__MeCH=NMe`

Low priority:

- Product-only `2-R` through `6-R` CO2 product series. These are useful for
  thermodynamic or structural comparison but are not first-batch pathway
  validation tasks.

## Notes

- Energy differences in `reaction_manifest.csv` come from SI electronic
  energies, not the paper's Gibbs free energies, so the values differ from the
  paper tables.
- RTIP/JAX is not a DFT program. Real-PES reproduction requires a DeePMD model
  covering `B/C/H/N/O/P/S/Si`, or another reliable PES provider.
- DeePMD `--type-map` must match the element order used during model training.
- `product.xyz` and `ts.xyz` are literature SI reference structures, not
  JAX-generated structures.

## Minimal Start Command

```bash
cd JAX
python -m rtip_jax.cli synthesize \
  --inputs examples/ic5c02384/reactions/1-tBu__CO2/1.xyz examples/ic5c02384/reactions/1-tBu__CO2/2.xyz \
  --output examples/ic5c02384/reactions/1-tBu__CO2/IS.xyz \
  --dist 5.0 \
  --seed 0
```
