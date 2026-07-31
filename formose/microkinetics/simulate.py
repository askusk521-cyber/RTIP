"""Formose microkinetic simulation (paper Section 3.5 / Figure 4).

Model: identical to the paper's program
(https://github.com/MillenniumDream/Microkinetics): mass-action rates with
the transition-state-theory prefactor ``k = kT/h * exp(-E/rt)``, the 28-step
Table S1 network (already barrier-elevated to >= 5 kcal/mol), and the Table
S1 initial concentrations.

Runtime method: the paper integrates the ODE with explicit Euler at
dt = 5e-12 s (2e11+ steps).  Here the *same ODE system* is integrated with
scipy's adaptive LSODA solver; the Euler solution converges to this ODE
solution as dt -> 0, so the results are equivalent within solver tolerance.
The deviation (integration method, not model) is recorded in RECORD.md.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

KB = 1.3806505e-23
H_PLANCK = 6.62606896e-34
KCAL_TO_EV = 0.0433641
TEMPERATURE = 338.15  # 65 C (paper/Table S1)
T_END = 2.0  # seconds of simulated time

INITIAL_CONCENTRATIONS = {
    "s1": 0.35,   # CH2O
    "s4": 0.05,   # HOCH2CHO
    "OH-": 0.06,
    "H2O": 55.5,
}


def parse_equation(equation: str) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    """Parse a Table S1 equation like `2 + 1 -> 3` or `3 + H2O -> 4 + OH-`."""

    left, right = equation.replace("−", "-").split("⇄")

    def parse_side(side: str) -> tuple[tuple[str, int], ...]:
        terms: list[tuple[str, int]] = []
        for token in side.replace("+", " ").split():
            token = token.strip()
            # Table S1 uses bare species numbers as labels; all stoichiometric
            # coefficients are 1.
            if token.isdigit():
                name = "s" + token
            elif token == "OH-":
                name = "OH-"
            elif token == "H2O":
                name = "H2O"
            else:
                raise ValueError(f"unknown species token: {token!r}")
            terms.append((name, 1))
        return tuple(terms)

    return parse_side(left), parse_side(right)


def load_table_s1(path: Path) -> list[dict]:
    reactions = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reactants, products = parse_equation(row["equation"])
            reactions.append({
                "index": int(row["index"]),
                "reactants": reactants,
                "products": products,
                "barrier": (
                    float(row["forward_barrier_kcal_mol"]),
                    float(row["reverse_barrier_kcal_mol"]),
                ),
            })
    return reactions


def build_ode(reactions: list[dict]):
    species: list[str] = []
    for reaction in reactions:
        for name, _multiple in reaction["reactants"] + reaction["products"]:
            if name not in species:
                species.append(name)
    index = {name: i for i, name in enumerate(species)}

    rt = (8.31451 / 96485.3) * TEMPERATURE
    prefactor = KB * TEMPERATURE / H_PLANCK

    kf = np.zeros(len(reactions))
    kr = np.zeros(len(reactions))
    for i, reaction in enumerate(reactions):
        kf[i] = prefactor * math.exp(-reaction["barrier"][0] * KCAL_TO_EV / rt)
        kr[i] = prefactor * math.exp(-reaction["barrier"][1] * KCAL_TO_EV / rt)

    # stoichiometry: (reaction, species) -> signed multiple
    stoich = np.zeros((len(reactions), len(species)))
    for i, reaction in enumerate(reactions):
        for name, multiple in reaction["reactants"]:
            stoich[i, index[name]] -= multiple
        for name, multiple in reaction["products"]:
            stoich[i, index[name]] += multiple

    reactant_idx = [[index[name] for name, _m in reaction["reactants"]] for reaction in reactions]
    product_idx = [[index[name] for name, _m in reaction["products"]] for reaction in reactions]

    def rhs(t: float, c: np.ndarray) -> np.ndarray:
        rates = np.zeros(len(reactions))
        for i in range(len(reactions)):
            forward = kf[i]
            for j in reactant_idx[i]:
                forward *= c[j]
            reverse = kr[i]
            for j in product_idx[i]:
                reverse *= c[j]
            rates[i] = forward - reverse
        return stoich.T @ rates

    return species, index, rhs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path("data/table_S1.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("microkinetics/runs"))
    args = parser.parse_args()

    reactions = load_table_s1(args.table)
    species, index, rhs = build_ode(reactions)
    c0 = np.zeros(len(species))
    for name, value in INITIAL_CONCENTRATIONS.items():
        c0[index[name]] = value

    print(f"species={len(species)} reactions={len(reactions)}")
    solution = solve_ivp(
        rhs,
        (0.0, T_END),
        c0,
        method="LSODA",
        rtol=1e-9,
        atol=1e-16,
        dense_output=True,
        max_step=0.01,
    )
    if not solution.success:
        raise RuntimeError(f"solver failed: {solution.message}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t = solution.t
    with open(args.output_dir / "concentrations.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s"] + species)
        for k in range(len(t)):
            writer.writerow([f"{t[k]:.9f}"] + [f"{solution.y[i, k]:.9e}" for i in range(len(species))])

    # --- Paper Figure 4 checks ---
    c2 = solution.y[index["s2"]]
    c4 = solution.y[index["s4"]]
    c5 = solution.y[index["s5"]]
    c11 = solution.y[index["s11"]]
    c19 = solution.y[index["s19"]]
    c23 = solution.y[index["s23"]]
    c26 = solution.y[index["s26"]]

    # Formyl anion steady level and its coupling rate R2 = kf2 * c2 * c1.
    rt = (8.31451 / 96485.3) * TEMPERATURE
    prefactor = KB * TEMPERATURE / H_PLANCK
    kf2 = prefactor * math.exp(-11.7 * KCAL_TO_EV / rt)
    c2_late = c2[-1]
    r2_late = kf2 * c2_late * solution.y[index["s1"]][-1]

    # Retroaldol step R28: 26 -> 4 + 5; net rate sign change (paper: ~0.76 s).
    kf28 = prefactor * math.exp(-13.5 * KCAL_TO_EV / rt)
    kr28 = prefactor * math.exp(-4.5 * KCAL_TO_EV / rt)
    r28 = kf28 * c26 - kr28 * c4 * c5
    sign_change = None
    for k in range(1, len(t)):
        if r28[k - 1] <= 0.0 < r28[k]:
            sign_change = t[k]
            break

    tetrose_total = c19 + c23
    summary = {
        "formyl_anion_c_M": float(c2_late),
        "formaldehyde_dimerization_rate_mol_L_s": float(r2_late),
        "retroaldol_net_rate_positive_at_s": sign_change,
        "ribose_c_M": float(c11[-1]),
        "linear_tetrose_c_M": float(tetrose_total[-1]),
        "glycolaldehyde_c_M": float(c4[-1]),
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
