"""Compound orders vs plain blocks (experiment 2026-11-16): equal-compute grid and pre-registered decision.

Families in the results are named ``<variant>_<structure>``: variant ``k12`` (orders [1,2], reference),
``k1`` (orders [1], equal bytes, same steps), ``k1eq`` (orders [1], equal wall time as the paired K=[1,2]
fit), ``k123`` (orders [1,2,3]); structure ``stride``, ``rand1``, ``rand2`` (permutation kind and seed).

For every non-reference point the report gives the bytes the K=[1,2] curve of the same structure needs
for the same validation relL2 (log-log interpolation along its Pareto front, never extrapolated) and the
byte saving of K=[1,2] = 1 - bytes_K12 / bytes_variant (positive = compound orders are better).

Decision (fixed before the run): compound orders stay the main variant if, at the smallest budget, K=[1,2]
saves >= 10 % bytes against K=[1] at equal compute (``k1eq``) for every structure on both layers.

  grid:   python -m quic_wc.pilot.compare_orders grid --phase-a <dir> --template <grid_a.json> --out <grid_b.json>
  report: python -m quic_wc.pilot.compare_orders report --results <dir> [<dir> ...] --out <file.md>
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from quic_wc.pilot.summarize import METRIC

THRESHOLD = 0.10
BASE_STEPS = 3000


def load_entries(directories: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """layer name -> real-teacher entries of all given result directories."""
    out: dict[str, list[dict[str, Any]]] = {}
    for directory in directories:
        path = directory / "results.json"
        res = json.loads((path if path.exists() else directory / "results.partial.json").read_text())
        out.setdefault(res["layer"]["name"], []).extend(
            q for q in res["quic_wc"] if q["teacher"]["kind"] == "real")
    return out


def split(family: str) -> tuple[str, str]:
    variant, _, structure = family.partition("_")
    return variant, structure


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    front, best = [], math.inf
    for ratio, err in sorted(points):
        if err < best:
            front.append((ratio, err))
            best = err
    return front


def bytes_for_error(front: list[tuple[float, float]], err: float) -> tuple[float, str] | None:
    """K=[1,2] bytes for ``err``: ("=", interpolated), ("<", cheapest point) when even the cheapest K=[1,2]
    point is better, (">", largest point) when no K=[1,2] point reaches ``err``."""
    if not front:
        return None
    if err > front[0][1]:
        return front[0][0], "<"
    if err < front[-1][1]:
        return front[-1][0], ">"
    for (r0, e0), (r1, e1) in zip(front, front[1:]):
        if e0 >= err >= e1:
            if e0 == e1:
                return r0, "="
            t = (math.log(err) - math.log(e0)) / (math.log(e1) - math.log(e0))
            return math.exp(math.log(r0) + t * (math.log(r1) - math.log(r0))), "="
    return front[0][0], "="  # single-point front with equal error


def saving_text(b12: tuple[float, str] | None, ratio: float) -> tuple[str, str, float | None, str]:
    """(bytes text, saving text, saving or its bound, relation of the true saving to the value)."""
    if b12 is None:
        return "—", "—", None, ""
    value, rel = b12
    saving = 1 - value / ratio
    sav_rel = {"=": "", "<": ">", ">": "<"}[rel]
    return f"{rel if rel != '=' else ''}{value:.4f}", f"{sav_rel}{saving:.1%}", saving, sav_rel


def grid(args: argparse.Namespace) -> None:
    """Phase B: K=[1] with the wall time of the K=[1,2] fit closest in bytes (same structure)."""
    template = json.loads(args.template.read_text())
    specs = {s["name"]: s for s in template["experiments"]}
    entries = next(iter(load_entries([args.phase_a]).values()))
    exps = []
    for e in entries:
        variant, structure = split(e["family"])
        if variant != "k1":
            continue
        refs = [r for r in entries if r["family"] == f"k12_{structure}"]
        ref = min(refs, key=lambda r: abs(math.log(r["byte_ratio"] / e["byte_ratio"])))
        steps = max(BASE_STEPS, round(BASE_STEPS * ref["seconds_total_phase"] / e["seconds_total_phase"], -2))
        stages = e["stages"][0]
        exps.append({"name": f"k1eq_{structure}_L{stages}",
                     "rank": specs[e["family"]]["rank"], "left": dict(specs[e["family"]]["left"], num_stages=stages),
                     "fit": {"steps": int(steps)}, "paired_with": ref["name"],
                     "description": f"equal wall time as {ref['name']} ({ref['seconds_total_phase']:.0f}s vs "
                                    f"{e['seconds_total_phase']:.0f}s at {BASE_STEPS} steps)"})
    out = template | {"description": "Phase B (equal compute for K=[1]) generated from " + str(args.phase_a),
                      "experiments": exps}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    for x in exps:
        print(x["name"], x["fit"]["steps"], x["description"])


def family_of(e: dict[str, Any]) -> str:
    """Family without the ``_L<k>`` suffix of single (non-sweep) phase-B specs."""
    return re.sub(r"_L\d+$", "", e["family"])


def report(args: argparse.Namespace) -> None:
    lines = ["# Compound řády proti prostým blokům (2026-11-16)", "",
             f"Metrika: validační relL2 bez biasu (`{METRIC}`), BF16 artefakt. `K12 bajty` = bajty, které křivka "
             "K=[1,2] stejné struktury potřebuje na stejnou chybu (log-log interpolace, bez extrapolace). "
             "Úspora K12 = 1 − K12 bajty / bajty varianty (kladná = compound řády lepší).", ""]
    decision: dict[str, Any] = {}
    for layer, entries in load_entries(args.results).items():
        lines += [f"## `{layer}`", ""]
        groups: dict[str, list[dict[str, Any]]] = {}
        for e in entries:
            groups.setdefault(family_of(e), []).append(e)
        structures = sorted({split(f)[1] for f in groups})
        lines += ["| struktura | varianta | L | kroky | poměr bajtů | relL2 | čas [s] | K12 bajty | úspora K12 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for structure in structures:
            ref = groups.get(f"k12_{structure}", [])
            front = pareto([(e["byte_ratio"], e["metrics"][METRIC]) for e in ref])
            for variant in ("k12", "k1", "k1eq", "k123"):
                for e in sorted(groups.get(f"{variant}_{structure}", []), key=lambda e: e["byte_ratio"]):
                    err, ratio = e["metrics"][METRIC], e["byte_ratio"]
                    b12 = None if variant == "k12" else bytes_for_error(front, err)
                    b_txt, s_txt, saving, rel = saving_text(b12, ratio)
                    lines.append(f"| {structure} | {variant} | {e['stages'][0]} | {e['fit']['steps_completed']} | "
                                 f"{ratio:.4f} | {err:.4f} | {e['seconds_total_phase']:.0f} | "
                                 f"{b_txt} | {s_txt} |")
                    if variant == "k1eq":
                        decision.setdefault(layer, {}).setdefault(structure, []).append((ratio, saving, rel, s_txt))
        lines.append("")
    lines += ["## Rozhodnutí podle předem stanoveného kritéria", "",
              f"Kritérium: při nejmenším rozpočtu K=[1,2] ušetří ≥ {THRESHOLD:.0%} bajtů proti K=[1] se stejným "
              "výpočtem (`k1eq`) pro každou strukturu na obou vrstvách.", ""]
    ok = bool(decision)
    for layer, per in decision.items():
        for structure, pts in sorted(per.items()):
            ratio, saving, rel, s_txt = min(pts, key=lambda p: p[0])
            # a lower bound (">") above the threshold passes; an upper bound ("<") passes only if it is >= too
            passed = saving is not None and saving >= THRESHOLD and rel in ("", ">")
            ok &= passed
            lines.append(f"- `{layer}`, {structure}: poměr bajtů {ratio:.4f}, úspora K12 "
                         f"{s_txt} → "
                         f"{'splněno' if passed else 'nesplněno'}")
    lines += ["", f"**Výsledek: {'compound řády zůstávají hlavní variantou' if ok else 'kritérium nesplněno'}.**", ""]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--phase-a", type=Path, required=True)
    g.add_argument("--template", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--results", type=Path, nargs="+", required=True)
    r.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    grid(args) if args.command == "grid" else report(args)


if __name__ == "__main__":
    main()
