"""Generalized synthesis-MD runner for size_scaling validation.

Runs attractive SynthesisPot NVT MD on ANY ic5c02384 reaction without a
per-reaction reorder map: the synthesis bias only needs ``mol_index`` (fragment
split from 1.xyz + 2.xyz), so it sidesteps the IS-vs-product atom-ordering
mismatch that blocks the directed attractive-to-product mode.

Per-seed variation comes from the random fragment layout/rotation in
``synthesize_layout``.  Analysis reuses the generalized ``rcmd_lib`` pipeline
(auto-detected B / N2 / small-molecule / B-X core) so B-X and N-C distances and
RMSD-to-reference are reported uniformly across reactions.

The whole point is to compare ``size_scaling`` OFF vs ON with everything else
identical: OFF leaves the (large) system under-driven by the 1/N-diluted bias,
ON restores a size-consistent per-atom driving force.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
from jax import random

import rcmd_lib
from rtip_jax.config import Para
from rtip_jax.constants import BOHR_TO_ANGSTROM
from rtip_jax.external import DeepMDBoundary, DeepMDPES
from rtip_jax.pes import SynthesisPot
from rtip_jax.system import System
from rtip_jax.workflows import run_rtip_nvt_md, synthesize_layout, temperature
from rtip_jax.workflows.md import atom_masses


def _combined_and_mol_index(reaction_dir: Path) -> tuple[System, tuple[tuple[int, ...], ...]]:
    frag1 = System.read_xyz(reaction_dir / "1.xyz")
    frag2 = System.read_xyz(reaction_dir / "2.xyz")
    coord = jnp.concatenate([frag1.coord, frag2.coord], axis=0)
    atom_type = tuple(frag1.atom_type or ()) + tuple(frag2.atom_type or ())
    if len(atom_type) != coord.shape[0]:
        atom_type = None
    combined = System(coord=coord, atom_type=atom_type, pot=0.0)
    mol_index = (
        tuple(range(frag1.natom)),
        tuple(range(frag1.natom, frag1.natom + frag2.natom)),
    )
    return combined, mol_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generalized synthesis-MD size_scaling runner.")
    p.add_argument("--reaction-dir", required=True)
    p.add_argument("--model", default="/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt")
    p.add_argument("--a0", type=float, default=0.0002)
    p.add_argument("--max-step", type=int, default=1500)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--temp-bath", type=float, default=300.0)
    p.add_argument("--tau", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size-scaling", action="store_true")
    p.add_argument("--synth-dist", type=float, default=5.0)
    p.add_argument("--print-step", type=int, default=1)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--skip-analysis", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    reaction_dir = Path(args.reaction_dir)
    if not reaction_dir.exists():
        fallback = Path(__file__).resolve().parent / "reactions" / reaction_dir.name
        if fallback.exists():
            reaction_dir = fallback

    combined, mol_index = _combined_and_mol_index(reaction_dir)
    synth_bohr = args.synth_dist / BOHR_TO_ANGSTROM
    initial = synthesize_layout(combined, mol_index, float(synth_bohr), key=random.PRNGKey(args.seed))

    reaction = reaction_dir.name
    scale_tag = "ON" if args.size_scaling else "OFF"
    output_prefix = args.output_prefix or f"synth_{reaction}_a{args.a0}_s{args.max_step}_seed{args.seed}_{scale_tag}"
    output_dir = Path(args.output_dir or f"{output_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    para = Para(
        a0=args.a0,
        max_step=args.max_step,
        print_step=args.print_step,
        dt=args.dt,
        temp_bath=args.temp_bath,
        tau=args.tau,
        size_scaling=args.size_scaling,
    )
    structure_file = output_dir / f"{output_prefix}.pdb"
    table_file = output_dir / f"{output_prefix}.out"

    config = SynthesisPot(
        initial_state=initial,
        mol_index=mol_index,
        para=para,
        str_output_file=str(structure_file),
        output_file=str(table_file),
    )

    pes = DeepMDPES(DeepMDBoundary(model=args.model))
    print(f"reaction={reaction}")
    print(f"reaction_dir={reaction_dir}")
    print(f"natom={initial.natom}")
    print(f"mol_index_sizes={[len(m) for m in mol_index]}")
    print(f"model={args.model}")
    print(f"output_dir={output_dir}")
    print(f"output_prefix={output_prefix}")
    print(f"a0={args.a0}")
    print(f"max_step={args.max_step}")
    print(f"dt={args.dt}")
    print(f"temp_bath={args.temp_bath}")
    print(f"tau={args.tau}")
    print(f"seed={args.seed}")
    print(f"size_scaling={args.size_scaling}")
    print(f"synth_dist={args.synth_dist}")

    masses = atom_masses(initial)
    result = run_rtip_nvt_md(
        config,
        pes,
        perturb=False,
        initial_velocity=None,
        write_outputs=True,
    )
    print(f"history_steps={len(result.history)}")
    print(f"final_state={result.history[-1].state_decision if result.history else 'unknown'}")
    print(f"final_energy_Ha={float(result.system.pot):.12f}")
    print(f"final_temp_K={float(temperature(result.velocity, masses)):.8f}")

    if not args.skip_analysis:
        summary = rcmd_lib.analyze_generalized_job(
            pdb_file=structure_file,
            out_file=table_file,
            reaction_dir=reaction_dir,
            input_file=None,
            output_dir=output_dir,
            output_prefix=output_prefix,
        )
        print("final_metrics=" + str(summary["final_metrics"]))
        print("best_bond_forming=" + str(summary["best_bond_forming"]))
        print(f"summary_json={output_dir / 'summary.json'}")
        print(f"summary_md={output_dir / 'summary.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
