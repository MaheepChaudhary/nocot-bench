"""Scoring and deliverables for the passes=1 vs passes=3 NCRI study.

Reads the graded rows of each setting and draw, places them on the sealed
NCRI 15.2 ladder with nocot.place (bootstrap 1000), and writes:

    results.csv   one row per setting, draw and scoring basis
    banks.csv     per-bank accuracy at p1 and p3 with a paired bootstrap CI
    fig1.png      per-bank accuracy difference (p3 - p1), sorted, with CIs
    fig2.png      NCRI at p1 and p3, all domains and without recheck_v2 + recon

Two scoring bases are reported, because the code and AGENTS.md disagree:
    place_py   what nocot.place computes (it ignores the 0.20 cell rule)
    a101       the 0.20 cell rule applied as AGENTS.md section 4 states it:
               cells flagged a101_excluded_unmeasured are dropped

Usage (from the repository root):
    python varloop/analyse.py
"""
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from nocot import grade as G          # noqa: E402  (repository code, unchanged)
from nocot.place import rungs_from_rows  # noqa: E402

import matplotlib                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt       # noqa: E402

OUT_DIR = "varloop"
PLACEMENT_DIR = os.path.join(OUT_DIR, "placement")
BOOTSTRAP = 1000
SEED = 20261122
SUBSET_DROP = ("recheck_v2", "recon")
SETTINGS = ("p1", "p3")
GRADED_DIRS = {1: "graded/{s}", 2: "graded_draw2/{s}"}

# Reference palette (dataviz skill, categorical slots 1 and 2) and text inks.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


class GradedSetting:
    """All graded rows of one setting and one draw."""

    def __init__(self, setting, draw):
        self.setting = setting
        self.draw = draw
        self.directory = GRADED_DIRS[draw].format(s=setting)
        self.paths = sorted(glob.glob(os.path.join(self.directory, "*.graded.jsonl")))
        self.rows_by_bank = {}
        for path in self.paths:
            rows = [json.loads(line) for line in open(path) if line.strip()]
            if rows:
                self.rows_by_bank[rows[0]["domain"]] = rows
        self.rungs = rungs_from_rows(self.paths)[0]
        self.ceiling_rungs = rungs_from_rows(self._write_valid_only())[0]
        self.ncri_banks = sorted({key.split(":")[0] for key in self.rungs})

    def _write_valid_only(self):
        """Copies of the graded files holding only VALID rows, for the
        conditional-valid ceiling. The graded files themselves are untouched."""
        directory = os.path.join(PLACEMENT_DIR, "ceiling_rows",
                                 f"{self.setting}_d{self.draw}")
        os.makedirs(directory, exist_ok=True)
        paths = []
        for path in self.paths:
            out_path = os.path.join(directory, os.path.basename(path))
            with open(path) as src, open(out_path, "w") as dst:
                for line in src:
                    if line.strip() and json.loads(line).get("valid"):
                        dst.write(line)
            paths.append(out_path)
        return paths

    def bank_summary(self, bank):
        _shots, evals = G.load_bank(bank, G.DATA)
        return G.summarise(self.rows_by_bank[bank], evals)

    def excluded_banks(self):
        """NCRI banks that the 0.20 cell rule marks as unmeasured."""
        return [b for b in self.ncri_banks
                if self.bank_summary(b)["a101_excluded_unmeasured"]]

    def _ncri_scorable_rows(self):
        return [r for b in self.ncri_banks for r in self.rows_by_bank[b]
                if r.get("scorable")]

    def invalid_percent(self):
        rows = self._ncri_scorable_rows()
        return 100.0 * sum(bool(r.get("invalid")) for r in rows) / len(rows)

    def blind_percent(self):
        rows = self._ncri_scorable_rows()
        return 100.0 * sum(r.get("witness_verdict") == "BLIND" for r in rows) / len(rows)


class Placer:
    """Runs nocot.place on a {bank:rung: [k, n]} dictionary."""

    def place(self, rungs, name):
        os.makedirs(PLACEMENT_DIR, exist_ok=True)
        rungs_path = os.path.join(PLACEMENT_DIR, name + "_rungs.json")
        out_path = os.path.join(PLACEMENT_DIR, name + ".json")
        with open(rungs_path, "w") as f:
            json.dump({k: list(v) for k, v in rungs.items()}, f, indent=1)
        subprocess.run(
            [sys.executable, "-m", "nocot.place", "--rungs", rungs_path,
             "--model", name, "--bootstrap", str(BOOTSTRAP), "--out", out_path],
            check=True, capture_output=True, text=True,
        )
        ncri = json.load(open(out_path))["ncri"]
        interval = ncri["interval"]
        return {
            "ncri": ncri["display"],
            "lo": interval["display_lo"],
            "hi": interval["display_hi"],
            "domains": f"{ncri['coverage_domains']}/{ncri['coverage_total']}",
            "ranked": ncri["ranked"],
        }


def drop_banks(rungs, banks):
    return {k: v for k, v in rungs.items() if k.split(":")[0] not in banks}


class ResultsTable:
    """results.csv: NCRI all / excluding the two banks, per setting, draw, basis."""

    def __init__(self, placer):
        self.placer = placer
        self.rows = []

    def add(self, graded):
        excluded = graded.excluded_banks()
        bases = {"place_py": (graded.rungs, graded.ceiling_rungs),
                 "a101": (drop_banks(graded.rungs, excluded),
                          drop_banks(graded.ceiling_rungs, excluded))}
        for basis, (rungs, ceiling_rungs) in bases.items():
            name = f"{graded.setting}_d{graded.draw}_{basis}"
            full = self.placer.place(rungs, name + "_all")
            subset = self.placer.place(drop_banks(rungs, SUBSET_DROP), name + "_excl")
            ceil_full = self.placer.place(ceiling_rungs, name + "_all_ceiling")
            ceil_subset = self.placer.place(drop_banks(ceiling_rungs, SUBSET_DROP),
                                            name + "_excl_ceiling")
            self.rows.append({
                "setting": graded.setting, "draw": graded.draw, "basis": basis,
                "ncri_all": round(full["ncri"], 2),
                "ci_all": f"[{full['lo']:.2f}, {full['hi']:.2f}]",
                "ncri_excl": round(subset["ncri"], 2),
                "ci_excl": f"[{subset['lo']:.2f}, {subset['hi']:.2f}]",
                "ncri_ceil_all": round(ceil_full["ncri"], 2),
                "ci_ceil_all": f"[{ceil_full['lo']:.2f}, {ceil_full['hi']:.2f}]",
                "ncri_ceil_excl": round(ceil_subset["ncri"], 2),
                "ci_ceil_excl": f"[{ceil_subset['lo']:.2f}, {ceil_subset['hi']:.2f}]",
                "domains_all": full["domains"], "domains_excl": subset["domains"],
                "ranked_all": full["ranked"], "ranked_excl": subset["ranked"],
                "invalid_pct": round(graded.invalid_percent(), 1),
                "blind_pct": round(graded.blind_percent(), 1),
                "a101_excluded_banks": " ".join(excluded) if basis == "a101" else "",
                "_full": full, "_subset": subset,
            })

    def write(self, path):
        fields = [k for k in self.rows[0] if not k.startswith("_")]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

    def find(self, setting, draw, basis):
        for row in self.rows:
            if (row["setting"], row["draw"], row["basis"]) == (setting, draw, basis):
                return row
        raise KeyError((setting, draw, basis))


class BankComparison:
    """Per-bank floor accuracy at p1 and p3, paired over items."""

    def __init__(self, graded_p1, graded_p3):
        self.graded_p1 = graded_p1
        self.graded_p3 = graded_p3
        self.rng = np.random.default_rng(SEED)

    @staticmethod
    def correct_by_item(rows):
        # Floor reading: reasoned/invalid rows are already correct=False.
        return {str(r["problem_number"]): bool(r["correct"])
                for r in rows if r.get("scorable")}

    def compare(self, bank):
        a = self.correct_by_item(self.graded_p1.rows_by_bank[bank])
        b = self.correct_by_item(self.graded_p3.rows_by_bank[bank])
        items = sorted(set(a) & set(b))
        x = np.array([a[i] for i in items], dtype=float)
        y = np.array([b[i] for i in items], dtype=float)
        picks = self.rng.integers(0, len(items), size=(BOOTSTRAP, len(items)))
        diffs = y[picks].mean(axis=1) - x[picks].mean(axis=1)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return {
            "bank": bank,
            "kind": "ncri" if bank in self.graded_p1.ncri_banks else "knowledge",
            "n": len(items),
            "acc_p1": round(x.mean(), 4), "acc_p3": round(y.mean(), 4),
            "diff": round(y.mean() - x.mean(), 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "ceil_p1": self.ceiling(self.graded_p1, bank),
            "ceil_p3": self.ceiling(self.graded_p3, bank),
            "invalid_pct_p1": round(100 * self.graded_p1.bank_summary(bank)["nonvalid_fraction"], 1),
            "invalid_pct_p3": round(100 * self.graded_p3.bank_summary(bank)["nonvalid_fraction"], 1),
        }

    @staticmethod
    def ceiling(graded, bank):
        """Conditional-valid accuracy (grade.summarise); empty if no valid rows."""
        value = graded.bank_summary(bank)["accuracy_ceiling_condvalid"]
        return "" if value is None else round(value, 4)

    def all_banks(self):
        banks = sorted(set(self.graded_p1.rows_by_bank) & set(self.graded_p3.rows_by_bank))
        return [self.compare(b) for b in banks]


class Figures:
    """fig1 (per-bank differences) and fig2 (NCRI by setting)."""

    @staticmethod
    def _style(ax):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)

    def fig1(self, bank_rows, path):
        rows = sorted(bank_rows, key=lambda r: r["diff"])
        fig, ax = plt.subplots(figsize=(8, 9), facecolor=SURFACE)
        self._style(ax)
        for i, r in enumerate(rows):
            color = ORANGE if r["bank"] in SUBSET_DROP else BLUE
            ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=color, lw=2,
                    solid_capstyle="round")
            ax.plot(r["diff"], i, "o", ms=8, color=color, mec=SURFACE, mew=2)
        ax.axvline(0, color=INK_2, lw=1)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{r['bank']}  (n={r['n']})" for r in rows], color=INK)
        for label, r in zip(ax.get_yticklabels(), rows):
            if r["bank"] in SUBSET_DROP:
                label.set_fontweight("bold")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_xlabel("accuracy difference, p3 − p1 (floor reading; 95% paired item bootstrap)",
                      color=INK_2)
        ax.set_title("Per-bank accuracy, passes=3 minus passes=1\n"
                     "orange = recheck_v2 and recon (the excluded pair)",
                     color=INK, fontsize=11, loc="left")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def fig2(self, results, path):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True, facecolor=SURFACE)
        titles = {"place_py": "As computed by nocot.place\n(0.20 cell rule not applied)",
                  "a101": "0.20 cell rule applied (AGENTS.md §4)\nboth settings UNRANKED"}
        for ax, basis in zip(axes, ("place_py", "a101")):
            self._style(ax)
            for offset, setting, color in ((-0.12, "p1", BLUE), (0.12, "p3", ORANGE)):
                row = results.find(setting, 1, basis)
                for x, key in ((0, "_full"), (1, "_subset")):
                    p = row[key]
                    ax.plot([x + offset] * 2, [p["lo"], p["hi"]], color=color, lw=2,
                            solid_capstyle="round")
                    ax.plot(x + offset, p["ncri"], "o", ms=8, color=color,
                            mec=SURFACE, mew=2, label=setting if x == 0 else None)
                    ax.annotate(f"{p['ncri']:.1f}\n{p['domains']}", (x + offset, p["hi"]),
                                textcoords="offset points", xytext=(0, 4), ha="center",
                                fontsize=8, color=INK_2)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["all domains", "excl. recheck_v2 + recon"], color=INK)
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(20, 68)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_title(titles[basis], color=INK, fontsize=10, loc="left")
        axes[0].set_ylabel("NCRI display (95% item bootstrap, draw 1)", color=INK_2)
        axes[0].legend(frameon=False, labelcolor=INK, loc="lower left")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def main():
    placer = Placer()
    results = ResultsTable(placer)
    graded = {}
    for draw in (1, 2):
        for setting in SETTINGS:
            g = GradedSetting(setting, draw)
            if not g.paths:
                print(f"skip {setting} draw {draw}: no graded rows")
                continue
            graded[(setting, draw)] = g
            results.add(g)
    results.write(os.path.join(OUT_DIR, "results.csv"))

    comparison = BankComparison(graded[("p1", 1)], graded[("p3", 1)])
    bank_rows = comparison.all_banks()
    with open(os.path.join(OUT_DIR, "banks.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(bank_rows[0]))
        writer.writeheader()
        writer.writerows(bank_rows)

    figures = Figures()
    figures.fig1(bank_rows, os.path.join(OUT_DIR, "fig1.png"))
    figures.fig2(results, os.path.join(OUT_DIR, "fig2.png"))
    for row in results.rows:
        print({k: v for k, v in row.items() if not k.startswith("_")})


if __name__ == "__main__":
    main()
