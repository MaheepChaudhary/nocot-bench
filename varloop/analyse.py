"""Scoring and deliverables for the passes=1 vs passes=3 NCRI study.

Reads the graded rows of each setting and draw, places them on the sealed
NCRI 15.2 ladder with nocot.place (bootstrap 1000), and writes:

    results.csv   one row per setting, draw and scoring basis
    banks.csv     per-bank accuracy at p1 and p3 with a paired bootstrap CI
    (figures are made by followup.py)

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


OUT_DIR = "varloop"
PLACEMENT_DIR = os.path.join(OUT_DIR, "placement")
BOOTSTRAP = 1000
SEED = 20261122
SUBSET_DROP = ("recheck_v2", "recon")
SETTINGS = ("p1", "p2", "p3", "p4", "p5")   # missing settings are skipped
GRADED_DIRS = {1: "graded/{s}", 2: "graded_draw2/{s}", 3: "graded_draw3/{s}"}



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
        self.n_regraded = 0
        self.n_regraded_unparseable = 0
        self.answer_first_rows_by_bank = {}
        self.answer_first_rungs = rungs_from_rows(self._write_answer_first())[0]

    def _write_answer_first(self):
        """SUPPLEMENTARY reading. A 'reasoned' row holds a bare answer in its
        content, written before the model opened its thinking block. Here
        that answer is graded with the repository's own grader. Every other
        row (valid, content-CoT loops, empty) keeps its official score."""
        directory = os.path.join(PLACEMENT_DIR, "answer_first_rows",
                                 f"{self.setting}_d{self.draw}")
        os.makedirs(directory, exist_ok=True)
        paths = []
        for path in self.paths:
            bank = os.path.basename(path).split("__")[1].replace(".graded.jsonl", "")
            rows = self.rows_by_bank.get(bank, [])
            items = {}
            if any(r.get("reason") == "reasoned" for r in rows):
                _shots, evals = G.load_bank(bank, G.DATA)
                items = {str(e.get("problem_number")): e for e in evals}
            new_rows = []
            for row in rows:
                row = dict(row)
                if row.get("scorable") and row.get("reason") == "reasoned":
                    item = items[str(row["problem_number"])]
                    result = G.grade(row.get("raw_text") or "", item, bank)
                    row["correct"] = bool(result["is_correct"])
                    row["answer_first_regraded"] = True
                    self.n_regraded += 1
                    if result["predicted"] is None:
                        self.n_regraded_unparseable += 1
                new_rows.append(row)
            self.answer_first_rows_by_bank[bank] = new_rows
            out_path = os.path.join(directory, os.path.basename(path))
            with open(out_path, "w") as dst:
                for row in new_rows:
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            paths.append(out_path)
        return paths

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
        bases = {"place_py": (graded.rungs, graded.ceiling_rungs,
                              graded.answer_first_rungs),
                 "a101": (drop_banks(graded.rungs, excluded),
                          drop_banks(graded.ceiling_rungs, excluded),
                          drop_banks(graded.answer_first_rungs, excluded))}
        for basis, (rungs, ceiling_rungs, af_rungs) in bases.items():
            name = f"{graded.setting}_d{graded.draw}_{basis}"
            full = self.placer.place(rungs, name + "_all")
            subset = self.placer.place(drop_banks(rungs, SUBSET_DROP), name + "_excl")
            ceil_full = self.placer.place(ceiling_rungs, name + "_all_ceiling")
            ceil_subset = self.placer.place(drop_banks(ceiling_rungs, SUBSET_DROP),
                                            name + "_excl_ceiling")
            af_full = self.placer.place(af_rungs, name + "_all_answer_first")
            af_subset = self.placer.place(drop_banks(af_rungs, SUBSET_DROP),
                                          name + "_excl_answer_first")
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
                # Supplementary answer-first reading (not an official NCRI).
                "ncri_af_all": round(af_full["ncri"], 2),
                "ci_af_all": f"[{af_full['lo']:.2f}, {af_full['hi']:.2f}]",
                "ncri_af_excl": round(af_subset["ncri"], 2),
                "ci_af_excl": f"[{af_subset['lo']:.2f}, {af_subset['hi']:.2f}]",
                "af_rows_regraded": graded.n_regraded,
                "af_rows_unparseable": graded.n_regraded_unparseable,
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
        # Separate stream, so the floor intervals stay as published.
        self.rng_af = np.random.default_rng(SEED + 1)

    @staticmethod
    def correct_by_item(rows):
        # Floor reading: reasoned/invalid rows are already correct=False.
        return {str(r["problem_number"]): bool(r["correct"])
                for r in rows if r.get("scorable")}

    def paired(self, rows_p1, rows_p3, rng):
        """Mean accuracy at each setting and a paired item-bootstrap CI of the
        difference p3 - p1."""
        a = self.correct_by_item(rows_p1)
        b = self.correct_by_item(rows_p3)
        items = sorted(set(a) & set(b))
        x = np.array([a[i] for i in items], dtype=float)
        y = np.array([b[i] for i in items], dtype=float)
        picks = rng.integers(0, len(items), size=(BOOTSTRAP, len(items)))
        diffs = y[picks].mean(axis=1) - x[picks].mean(axis=1)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return len(items), x.mean(), y.mean(), lo, hi

    def compare(self, bank):
        n, acc1, acc3, lo, hi = self.paired(self.graded_p1.rows_by_bank[bank],
                                            self.graded_p3.rows_by_bank[bank], self.rng)
        _n, af1, af3, af_lo, af_hi = self.paired(
            self.graded_p1.answer_first_rows_by_bank[bank],
            self.graded_p3.answer_first_rows_by_bank[bank], self.rng_af)
        return {
            "bank": bank,
            "kind": "ncri" if bank in self.graded_p1.ncri_banks else "knowledge",
            "n": n,
            "acc_p1": round(acc1, 4), "acc_p3": round(acc3, 4),
            "diff": round(acc3 - acc1, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "af_p1": round(af1, 4), "af_p3": round(af3, 4),
            "af_diff": round(af3 - af1, 4),
            "af_ci_lo": round(af_lo, 4), "af_ci_hi": round(af_hi, 4),
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


def main():
    placer = Placer()
    results = ResultsTable(placer)
    graded = {}
    for draw in GRADED_DIRS:
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

    for row in results.rows:
        print({k: v for k, v in row.items() if not k.startswith("_")})


if __name__ == "__main__":
    main()
