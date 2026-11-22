"""Pass counts 1-5: summary table and the bank grid (fig4). SUPPLEMENTARY.

Reads results.csv (written by analyse.py), the graded rows and the
answer-first rows of every pass count that has two draws, and writes:

    passes.csv  per pass count: floor, answer-first and ceiling NCRI
                (mean of draws 1 and 2, with each draw), invalid %, shot-echo rate
    fig4.png    one panel per NCRI bank: bank accuracy against NCRI

Usage (from the repository root):
    python varloop/passes.py
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "varloop"))
from nocot import place as P                  # noqa: E402
from followup import DrawRows, ShotEcho       # noqa: E402

import matplotlib                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt               # noqa: E402

OUT_DIR = "varloop"
PASSES = (1, 2, 3, 4, 5)
DRAWS = (1, 2)
MODELS_CSV = "data/release/models_ncri15_2.csv"
RUNGS_CSV = "data/release/rungs_ncri15_2.csv"

# Ordinal blue ramp (dataviz reference palette, steps 250..650) for passes 1-5.
BLUE_RAMP = {1: "#86b6ef", 2: "#5598e7", 3: "#2a78d6", 4: "#1c5cab", 5: "#104281"}
MARKER_SIZE = {1: 4.5, 2: 5.5, 3: 6.5, 4: 7.5, 5: 8.5}
# Reserved for Dory (one charcoal ramp, light to dark); not drawn here.
CHARCOAL_RAMP = ("#a8a7a1", "#86857f", "#64635e", "#43423e", "#22211f")
PALE_GREY = "#d9d8d3"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


class PassSetting:
    """Rows and summary numbers of one pass count."""

    def __init__(self, passes, results):
        self.passes = passes
        self.label = f"p{passes}"
        self.draws = {d: DrawRows(self.label, d) for d in DRAWS}
        self.results = {int(r["draw"]): r for r in results
                        if r["setting"] == self.label and r["basis"] == "place_py"}

    def available(self):
        return all(self.draws[d].exists() for d in DRAWS) and \
            all(d in self.results for d in DRAWS)

    def ncri(self, column):
        values = [float(self.results[d][column]) for d in DRAWS]
        return float(np.mean(values)), values

    def bank_accuracy(self, bank, reading):
        """Mean over draws of the bank's accuracy on scorable rows."""
        accs = []
        for d in DRAWS:
            dr = self.draws[d]
            rows = (dr.answer_first if reading == "answer_first" else dr.official)[bank]
            scorable = [r for r in rows if r.get("scorable")]
            accs.append(sum(bool(r["correct"]) for r in scorable) / len(scorable))
        return float(np.mean(accs))

    def invalid_percent(self):
        values = [float(self.results[d]["invalid_pct"]) for d in DRAWS]
        return float(np.mean(values))

    def echo_rate(self, echo):
        n = n_echo = 0
        for d in DRAWS:
            for row in echo.table(self.draws[d]):
                n += row["n_reasoned"]
                n_echo += row["n_echo"]
        return n_echo / n if n else float("nan"), n / len(DRAWS)


class RungCurve:
    """Expected bank accuracy at ability theta, from the sealed rung b and c:
    item-weighted mean of c + (1 - c) * sigmoid(theta - b)."""

    def __init__(self):
        self.by_bank = {}
        for r in csv.DictReader(open(RUNGS_CSV)):
            if r["rung_id"] in P.RUNGS:
                self.by_bank.setdefault(r["bank"], []).append(
                    (float(r["b"]), float(r["c"]), int(r["n_items"])))

    def curve(self, bank, displays):
        rungs = self.by_bank[bank]
        total = sum(n for _b, _c, n in rungs)
        out = []
        for disp in displays:
            theta = P.theta_from_display(disp)
            acc = sum(n * (c + (1 - c) / (1 + math.exp(-(theta - b))))
                      for b, c, n in rungs) / total
            out.append(acc)
        return out


def write_table(settings, echo):
    rows = []
    for s in settings:
        floor, floor_d = s.ncri("ncri_all")
        af, af_d = s.ncri("ncri_af_all")
        ceil, ceil_d = s.ncri("ncri_ceil_all")
        floor_x, _ = s.ncri("ncri_excl")
        af_x, _ = s.ncri("ncri_af_excl")
        ceil_x, _ = s.ncri("ncri_ceil_excl")
        rate, n_reasoned = s.echo_rate(echo)
        rows.append({
            "passes": s.passes,
            "floor_all": round(floor, 2), "floor_d1_d2": f"{floor_d[0]:.2f} / {floor_d[1]:.2f}",
            "answer_first_all": round(af, 2), "af_d1_d2": f"{af_d[0]:.2f} / {af_d[1]:.2f}",
            "ceiling_all": round(ceil, 2), "ceil_d1_d2": f"{ceil_d[0]:.2f} / {ceil_d[1]:.2f}",
            "floor_excl": round(floor_x, 2), "answer_first_excl": round(af_x, 2),
            "ceiling_excl": round(ceil_x, 2),
            "invalid_pct": round(s.invalid_percent(), 1),
            "shot_echo_rate": round(rate, 4), "reasoned_rows_per_draw": round(n_reasoned, 1),
            "domains": s.results[1]["domains_all"],
        })
    with open(os.path.join(OUT_DIR, "passes.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def fig4(settings, path):
    models = list(csv.DictReader(open(MODELS_CSV)))
    curve = RungCurve()
    banks = sorted(curve.by_bank)
    banks = [b for b in banks if f"ncri_{b}" in models[0]]
    first, last = settings[0], settings[-1]

    def gain(bank):
        return (last.bank_accuracy(bank, "answer_first")
                - first.bank_accuracy(bank, "answer_first"))
    banks.sort(key=gain, reverse=True)

    xs = np.linspace(-10, 150, 200)
    cols = 5
    rows_n = math.ceil(len(banks) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.1 * cols, 2.7 * rows_n),
                             sharex=True, sharey=True, facecolor=SURFACE)
    for ax in axes.flat[len(banks):]:
        ax.axis("off")
    for ax, bank in zip(axes.flat, banks):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=7)
        mx = [float(m["ncri15_2"]) for m in models if m[f"ncri_{bank}"] not in ("", None)]
        my = [float(m[f"ncri_{bank}"]) for m in models if m[f"ncri_{bank}"] not in ("", None)]
        ax.scatter(mx, my, s=6, color=PALE_GREY, lw=0, zorder=1)
        ax.plot(xs, curve.curve(bank, xs), color=INK_2, lw=1, zorder=2)

        ax_x = [s.ncri("ncri_af_all")[0] for s in settings]
        ax_y = [s.bank_accuracy(bank, "answer_first") for s in settings]
        ax.plot(ax_x, ax_y, color=BLUE_RAMP[3], lw=0.8, zorder=3)
        for s, x, y in zip(settings, ax_x, ax_y):
            ax.plot(x, y, "o", ms=MARKER_SIZE[s.passes], color=BLUE_RAMP[s.passes],
                    mec="white", mew=1.2, zorder=4)
            fx, fy = s.ncri("ncri_all")[0], s.bank_accuracy(bank, "floor")
            ax.plot(fx, fy, "o", ms=MARKER_SIZE[s.passes], mfc="none",
                    mec=BLUE_RAMP[s.passes], mew=1.2, zorder=4)
        for s, x, y in ((first, ax_x[0], ax_y[0]), (last, ax_x[-1], ax_y[-1])):
            ax.annotate(str(s.passes), (x, y), textcoords="offset points",
                        xytext=(5, 3), fontsize=7, color=INK)
        ax.set_title(f"{bank}  (gain {gain(bank):+.2f})", fontsize=8, color=INK, loc="left")
        ax.grid(color=GRID, lw=0.6)
        ax.set_xlim(-10, 150)
        ax.set_ylim(-0.03, 1.03)
    for ax in axes[-1]:
        ax.set_xlabel("NCRI 15.2 display", fontsize=8, color=INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("bank accuracy", fontsize=8, color=INK_2)
    fig.suptitle(
        "Bank accuracy against NCRI. Grey dots: released models; line: expected accuracy from "
        "the sealed rung b and c.\nBlue: TML-d24-varloop at passes "
        f"{first.passes}-{last.passes} (lighter/smaller to darker/larger), answer-first, "
        "draws 1-2 averaged; hollow = official floor.\nSorted by answer-first gain from "
        f"{first.passes} to {last.passes} passes. Charcoal is reserved for Dory. "
        "Varloop NCRI is UNRANKED under the 0.20 rule.",
        fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    results = list(csv.DictReader(open(os.path.join(OUT_DIR, "results.csv"))))
    settings = [s for s in (PassSetting(p, results) for p in PASSES) if s.available()]
    print("pass counts with two draws:", [s.passes for s in settings])
    rows = write_table(settings, ShotEcho())
    for row in rows:
        print(row)
    fig4(settings, os.path.join(OUT_DIR, "fig4.png"))


if __name__ == "__main__":
    main()
