"""Command-line entry points for the staged JAX migration."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from jax import random

from rtip_jax.config import Para, format_default_para, load_para
from rtip_jax.external.cp2k import Cp2kBoundary
from rtip_jax.external.deepmd import DeepMDBoundary, DeepMDPES
from rtip_jax.io.outputs import output_rtip
from rtip_jax.pes import HarmonicPES, RepulsivePot
from rtip_jax.system import System
from rtip_jax.workflows import (
    run_idwm_repulsive_path_sampling,
    run_rtip_nvt_md,
    run_rtip_repulsive_path_sampling,
    synthesize_layout,
)


def parse_mol_index(value: str) -> tuple[tuple[int, ...], ...]:
    """Parse `0,1;2,3` into molecule-index tuples."""

    molecules: list[tuple[int, ...]] = []
    for group in value.split(";"):
        group = group.strip()
        if not group:
            continue
        molecules.append(tuple(int(item.strip()) for item in group.split(",") if item.strip()))
    if not molecules:
        raise argparse.ArgumentTypeError("mol-index must contain at least one molecule")
    if any(len(molecule) == 0 for molecule in molecules):
        raise argparse.ArgumentTypeError("mol-index cannot contain an empty molecule")
    return tuple(molecules)


def parse_type_map(value: str) -> tuple[str, ...]:
    """Parse `O,H,Ca` into a DeePMD type-map tuple."""

    type_map = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(type_map) == 0:
        raise argparse.ArgumentTypeError("type-map must contain at least one element symbol")
    return type_map


def _load_para(path: str | None, max_step: int | None = None) -> Para:
    para = Para() if path is None else load_para(path)
    if max_step is not None:
        para = replace(para, max_step=max_step)
    return para


def _cmd_show_default_config(_args: argparse.Namespace) -> int:
    sys.stdout.write(format_default_para())
    return 0


def _cmd_cp2k_boundary(args: argparse.Namespace) -> int:
    boundary = Cp2kBoundary(input_file=args.input, output_file=args.output, mpi_comm=args.mpi_comm)
    sys.stdout.write(json.dumps(asdict(boundary), indent=2, sort_keys=True) + "\n")
    return 0


def _cmd_deepmd_boundary(args: argparse.Namespace) -> int:
    boundary = DeepMDBoundary(model=args.model, type_map=args.type_map)
    sys.stdout.write(json.dumps(asdict(boundary), indent=2, sort_keys=True) + "\n")
    return 0


def _cmd_synthesize(args: argparse.Namespace) -> int:
    system = System.read_xyz(args.input)
    key = random.PRNGKey(args.seed)
    out = synthesize_layout(system, args.mol_index, args.dist, key=key)
    out.write_xyz(args.output, create_new_file=True, step=0)
    return 0


def _cmd_mock_pathway(args: argparse.Namespace) -> int:
    system = System.read_xyz(args.input)
    para = _load_para(args.config, args.max_step)
    paths = output_rtip(base_dir=args.output_dir)
    config = RepulsivePot(
        local_min=system,
        nearby_ts=(),
        para=para,
        str_output_file=str(paths.structure),
        output_file=str(paths.table),
    )
    pes = HarmonicPES(k=args.k, center=system.coord)
    key = random.PRNGKey(args.seed)
    if args.method == "rtip":
        run_rtip_repulsive_path_sampling(config, pes, key=key, line_search=None, write_outputs=True)
    else:
        run_idwm_repulsive_path_sampling(config, pes, key=key, line_search=None, write_outputs=True)
    return 0


def _cmd_mock_md(args: argparse.Namespace) -> int:
    system = System.read_xyz(args.input)
    para = _load_para(args.config, args.max_step)
    paths = output_rtip(base_dir=args.output_dir)
    config = RepulsivePot(
        local_min=system,
        nearby_ts=(),
        para=para,
        str_output_file=str(paths.structure),
        output_file=str(paths.table),
    )
    pes = HarmonicPES(k=args.k, center=system.coord)
    run_rtip_nvt_md(config, pes, key=random.PRNGKey(args.seed), perturb=not args.no_perturb, write_outputs=True)
    return 0


def _deepmd_pes(args: argparse.Namespace) -> DeepMDPES:
    return DeepMDPES(DeepMDBoundary(model=args.model, type_map=args.type_map))


def _cmd_deepmd_pathway(args: argparse.Namespace) -> int:
    system = System.read_xyz(args.input)
    para = _load_para(args.config, args.max_step)
    paths = output_rtip(base_dir=args.output_dir)
    config = RepulsivePot(
        local_min=system,
        nearby_ts=(),
        para=para,
        str_output_file=str(paths.structure),
        output_file=str(paths.table),
    )
    pes = _deepmd_pes(args)
    key = random.PRNGKey(args.seed)
    if args.method == "rtip":
        run_rtip_repulsive_path_sampling(config, pes, key=key, write_outputs=True)
    else:
        run_idwm_repulsive_path_sampling(config, pes, key=key, write_outputs=True)
    return 0


def _cmd_deepmd_md(args: argparse.Namespace) -> int:
    system = System.read_xyz(args.input)
    para = _load_para(args.config, args.max_step)
    paths = output_rtip(base_dir=args.output_dir)
    config = RepulsivePot(
        local_min=system,
        nearby_ts=(),
        para=para,
        str_output_file=str(paths.structure),
        output_file=str(paths.table),
    )
    run_rtip_nvt_md(
        config,
        _deepmd_pes(args),
        key=random.PRNGKey(args.seed),
        perturb=not args.no_perturb,
        write_outputs=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtip-jax")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_config = subparsers.add_parser("show-default-config", help="Print default Para values as JSON.")
    show_config.set_defaults(func=_cmd_show_default_config)

    boundary = subparsers.add_parser("cp2k-boundary", help="Print the documented legacy CP2K boundary contract.")
    boundary.add_argument("--input", default="cp2k.inp")
    boundary.add_argument("--output", default="cp2k.out")
    boundary.add_argument("--mpi-comm", type=int, default=None)
    boundary.set_defaults(func=_cmd_cp2k_boundary)

    deepmd_boundary = subparsers.add_parser("deepmd-boundary", help="Print the DeePMD replacement PES contract.")
    deepmd_boundary.add_argument("--model", required=True)
    deepmd_boundary.add_argument("--type-map", type=parse_type_map, default=None)
    deepmd_boundary.set_defaults(func=_cmd_deepmd_boundary)

    synthesize = subparsers.add_parser("synthesize", help="Create a separated synthesis layout from an XYZ file.")
    synthesize.add_argument("--input", required=True)
    synthesize.add_argument("--output", required=True)
    synthesize.add_argument("--mol-index", required=True, type=parse_mol_index)
    synthesize.add_argument("--dist", type=float, default=5.0)
    synthesize.add_argument("--seed", type=int, default=0)
    synthesize.set_defaults(func=_cmd_synthesize)

    mock_pathway = subparsers.add_parser("mock-pathway", help="Run a deterministic mock repulsive pathway workflow.")
    mock_pathway.add_argument("--input", required=True)
    mock_pathway.add_argument("--output-dir", default=".")
    mock_pathway.add_argument("--config", default=None)
    mock_pathway.add_argument("--max-step", type=int, default=None)
    mock_pathway.add_argument("--method", choices=("rtip", "idwm"), default="rtip")
    mock_pathway.add_argument("--seed", type=int, default=0)
    mock_pathway.add_argument("--k", type=float, default=1.0)
    mock_pathway.set_defaults(func=_cmd_mock_pathway)

    mock_md = subparsers.add_parser("mock-md", help="Run a deterministic mock RTIP NVT MD workflow.")
    mock_md.add_argument("--input", required=True)
    mock_md.add_argument("--output-dir", default=".")
    mock_md.add_argument("--config", default=None)
    mock_md.add_argument("--max-step", type=int, default=None)
    mock_md.add_argument("--seed", type=int, default=0)
    mock_md.add_argument("--k", type=float, default=1.0)
    mock_md.add_argument("--no-perturb", action="store_true")
    mock_md.set_defaults(func=_cmd_mock_md)

    deepmd_pathway = subparsers.add_parser("deepmd-pathway", help="Run RTIP/IDWM repulsive pathway sampling with DeePMD.")
    deepmd_pathway.add_argument("--input", required=True)
    deepmd_pathway.add_argument("--model", required=True)
    deepmd_pathway.add_argument("--type-map", type=parse_type_map, default=None)
    deepmd_pathway.add_argument("--output-dir", default=".")
    deepmd_pathway.add_argument("--config", default=None)
    deepmd_pathway.add_argument("--max-step", type=int, default=None)
    deepmd_pathway.add_argument("--method", choices=("rtip", "idwm"), default="rtip")
    deepmd_pathway.add_argument("--seed", type=int, default=0)
    deepmd_pathway.set_defaults(func=_cmd_deepmd_pathway)

    deepmd_md = subparsers.add_parser("deepmd-md", help="Run RTIP NVT MD with DeePMD.")
    deepmd_md.add_argument("--input", required=True)
    deepmd_md.add_argument("--model", required=True)
    deepmd_md.add_argument("--type-map", type=parse_type_map, default=None)
    deepmd_md.add_argument("--output-dir", default=".")
    deepmd_md.add_argument("--config", default=None)
    deepmd_md.add_argument("--max-step", type=int, default=None)
    deepmd_md.add_argument("--seed", type=int, default=0)
    deepmd_md.add_argument("--no-perturb", action="store_true")
    deepmd_md.set_defaults(func=_cmd_deepmd_md)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
