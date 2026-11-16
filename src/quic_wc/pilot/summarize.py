"""Cross-layer summary of pilot results: error/bytes curves, gate M3 and bytes needed for a target error.

For every results directory (one layer) and bias policy it lists the real-teacher QuIC-WC entries
(sorted by bytes) with the equal-byte ASVD reference, and for each target ``tau`` of the
validation relL2 without bias the smallest stored size reaching it: QuIC-WC = smallest evaluated
entry with error <= tau (adaptive L read off the measured curve); ASVD = smallest evaluated rank
with error <= tau (from the baseline list and the equal-budget references); "—" means no evaluated
point reaches tau. Because measured points depend on grid density, the table also gives an
*interpolated* size: piecewise-linear in log(bytes)-log(error) along each method's Pareto front
(running minimum of the error over increasing bytes), never extrapolated; when the smallest measured
size already reaches tau only an upper bound ("≤") is shown and no saving is computed.

Usage: python -m quic_wc.pilot.summarize --results <dir> [<dir> ...] --out <file.md> [--targets 0.2 0.1 0.05]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRIC = "exact_val_rel_l2_no_bias"


def asvd_points(results: dict[str, Any], policy: str) -> dict[int, tuple[float, float]]:
    """rank -> (byte_ratio, relL2) from baselines and equal-budget references of the real teacher."""
    points: dict[int, tuple[float, float]] = {}
    for e in results["baselines"]:
        if e["method"] == "asvd" and e["bias_policy"] == policy:
            points[e["rank"]] = (e["byte_ratio"], e["metrics"][METRIC])
    for q in results["quic_wc"]:
        if q["teacher"]["kind"] != "real" or q["bias_policy"] != policy:
            continue
        for key in ("asvd_r_bytes", "asvd_r_bytes_ceil", "asvd_r_params"):
            ref = q["equal_budget"].get(key)
            if ref and ref["rank"] > 0:
                points[ref["rank"]] = (ref["byte_ratio"], ref["metrics"][METRIC])
    return dict(sorted(points.items()))


def smallest(points: list[tuple[str, float, float]], tau: float) -> tuple[str, float] | None:
    ok = [(label, ratio) for label, ratio, err in points if err <= tau]
    return min(ok, key=lambda x: x[1]) if ok else None


def interpolated_ratio(points: list[tuple[str, float, float]], tau: float) -> float | None:
    """Byte ratio where the Pareto front crosses tau (log-log linear between measured points).

    Returns None when tau is not bracketed by two measured points (not reached, or already reached
    by the smallest measured size, where only an upper bound would be known)."""
    import math

    front, best = [], math.inf
    for _, ratio, err in sorted(points, key=lambda p: p[1]):
        if err < best:
            front.append((ratio, err))
            best = err
    for (r0, e0), (r1, e1) in zip(front, front[1:]):
        if e0 >= tau >= e1 and r0 > 0:
            if e0 == e1:
                return r0
            t = (math.log(tau) - math.log(e0)) / (math.log(e1) - math.log(e0))
            return math.exp(math.log(r0) + t * (math.log(r1) - math.log(r0)))
    return None


def bound(best: tuple[str, float] | None, value: float | None) -> str:
    """Interpolated ratio, or "<= smallest measured size" when only an upper bound is known."""
    return f"≤{best[1]:.4f}" if value is None and best is not None else fmt(value)


def fmt(v: float | None, d: int = 4) -> str:
    return "—" if v is None else f"{v:.{d}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets", type=float, nargs="+", default=[0.4, 0.3, 0.2, 0.1, 0.05])
    args = parser.parse_args()
    lines = ["# Souhrn pilotu přes vrstvy", "",
             f"Metrika: validační relL2 bez biasu (`{METRIC}`), nasaditelná BF16 podoba. Poměr bajtů vůči husté BF16 "
             "vrstvě. ASVD ≥B = nejmenší rank s alespoň stejnými bajty. Rezerva = 1 − QuIC/ASVD.", ""]
    summary: dict[str, Any] = {}
    for directory in args.results:
        res = json.loads((directory / "results.json").read_text())
        lay = res["layer"]
        lines += [f"## `{lay['name']}` ({lay['out_features']}×{lay['in_features']}, "
                  f"{'s biasem' if res.get('has_bias', True) else 'bez biasu'})", "", f"Zdroj: `{directory}`.", ""]
        gate = res.get("gate_m3", {})
        for policy in ("fixed", "corrected"):
            real = [q for q in res["quic_wc"] if q["teacher"]["kind"] == "real" and q["bias_policy"] == policy]
            families = sorted({q.get("family", "all") for q in real}, key=lambda f: (len(f), f))
            for family in families:
                lines += summarize_family(res, policy, family, sorted(
                    (q for q in real if q.get("family", "all") == family), key=lambda q: q["tensor_bytes"]),
                    gate, args.targets, summary)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))


def summarize_family(res: dict[str, Any], policy: str, family: str, entries: list[dict[str, Any]],
                     gate: dict[str, Any], targets: list[float], summary: dict[str, Any]) -> list[str]:
    lay = res["layer"]
    lines: list[str] = []
    lines += [f"### bias `{policy}`, rodina `{family}`" + (
        f" (brána M3 vrstvy pro `{policy}`: {'PROŠLA' if gate[policy]['pass'] else 'NEPROŠLA'})"
        if policy in gate else ""), "",
              "| varianta | L | rank | poměr bajtů | QuIC relL2 | cos_nb | ASVD ≥B (r: relL2) | rezerva |",
              "|---|---|---:|---:|---:|---:|---|---:|"]
    for q in entries:
        g = q["gate_m3"]
        ref = q["equal_budget"]["asvd_r_bytes_ceil"]
        lines.append(f"| {q['name']} | {q['stages'][0]}/{q['stages'][1]} | {q['config']['rank']} | "
                     f"{q['byte_ratio']:.4f} | {q['metrics'][METRIC]:.4f} | {q['metrics']['cos_no_bias_mean']:.4f} | "
                     f"{ref['rank']}: {ref['metrics'][METRIC]:.4f} | {fmt(g['relative_margin'], 3)} |")
    asvd = asvd_points(res, policy)
    quic_pts = [(q["name"], q["byte_ratio"], q["metrics"][METRIC]) for q in entries]
    asvd_pts = [(f"r{r}", ratio, err) for r, (ratio, err) in asvd.items()]
    lines += ["", "| cílová relL2 τ | QuIC-WC změřeno: nejmenší varianta (poměr bajtů) | ASVD změřeno: nejmenší "
              "rank (poměr bajtů) | QuIC interpol. poměr bajtů | ASVD interpol. poměr bajtů | úspora bajtů "
              "(interpol.) |", "|---:|---|---|---:|---:|---:|"]
    per_tau = {}
    for tau in targets:
        q_best, a_best = smallest(quic_pts, tau), smallest(asvd_pts, tau)
        q_int, a_int = interpolated_ratio(quic_pts, tau), interpolated_ratio(asvd_pts, tau)
        saving = None if not (q_int and a_int) else 1 - q_int / a_int
        per_tau[str(tau)] = {"quic_measured": q_best, "asvd_measured": a_best, "quic_interpolated": q_int,
                             "asvd_interpolated": a_int, "byte_saving_interpolated": saving}
        lines.append(f"| {tau} | {f'{q_best[0]} ({q_best[1]:.4f})' if q_best else '—'} | "
                     f"{f'{a_best[0]} ({a_best[1]:.4f})' if a_best else '—'} | {bound(q_best, q_int)} | "
                     f"{bound(a_best, a_int)} | {fmt(saving, 3)} |")
    lines.append("")
    summary.setdefault(lay["name"], {}).setdefault(policy, {"gate": gate.get(policy), "families": {}})
    summary[lay["name"]][policy]["families"][family] = {
        "points": [{"name": q["name"], "byte_ratio": q["byte_ratio"], "rel_l2": q["metrics"][METRIC],
                    "asvd_ceil_rel_l2": q["equal_budget"]["asvd_r_bytes_ceil"]["metrics"][METRIC]}
                   for q in entries], "targets": per_tau}
    return lines


if __name__ == "__main__":
    main()
