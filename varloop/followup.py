"""Follow-up analysis for the passes=1 vs passes=3 study. SUPPLEMENTARY.

Uses only rows that already exist (graded rows of each draw, and the
answer-first rows written by analyse.py). grade.py and place.py are used, not
changed. The official floor and ceiling in results.csv are not touched.

    1. gap_intervals.csv  paired item bootstrap (2000) on the p3 - p1 NCRI gap
    2. shot_echo.csv      reasoned rows whose content is a k-shot answer
    3. paired_valid.csv   accuracy on items valid at both settings
    4. banks_avg.csv      per-bank differences averaged over draws 1 and 2,
                          with the draw-to-draw change; groups.csv
    5. fig1.png, fig2.png, fig3.png

Usage (from the repository root):
    python varloop/followup.py
"""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from nocot import grade as G                     # noqa: E402
from nocot import place as P                     # noqa: E402

import matplotlib                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402

OUT_DIR = "varloop"
BOOTSTRAP = 2000
SEED = 20261123
SETTINGS = ("p1", "p3")
SUBSET_DROP = ("recheck_v2", "recon")
GRADED_DIRS = {1: "graded/{s}", 2: "graded_draw2/{s}", 3: "graded_draw3/{s}"}
AF_DIR = "varloop/placement/answer_first_rows/{s}_d{d}"
K_SHOT = 10   # nocot.run DEFAULT_K_SHOT; rows were run with the default

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#efeee9"


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


class DrawRows:
    """Graded rows (official) and answer-first rows of one setting and draw."""

    def __init__(self, setting, draw):
        self.setting = setting
        self.draw = draw
        self.official = self._load(GRADED_DIRS[draw].format(s=setting))
        self.answer_first = self._load(AF_DIR.format(s=setting, d=draw))

    @staticmethod
    def _load(directory):
        by_bank = {}
        for path in sorted(glob.glob(os.path.join(directory, "*.graded.jsonl"))):
            rows = read_jsonl(path)
            if rows:
                by_bank[rows[0]["domain"]] = rows
        return by_bank

    def exists(self):
        return bool(self.official) and bool(self.answer_first)


class RungMap:
    """Which sealed rung each item sits on, following place.rungs_from_rows
    exactly (A232 annex skipped, hard rungs resolved, first row wins)."""

    def __init__(self):
        self.hard = P.ncri_hard_rung_of(None)
        self.annex = {(b, str(p)) for b, p in P.ANNEXED_ITEMS}

    def items(self, rows_by_bank):
        """{(bank, pn): (rung, scorable, correct, valid)} for on-rung rows."""
        out = {}
        for rows in rows_by_bank.values():
            for r in rows:
                bank, pn, scorable, correct = P._row_view(r)
                if not scorable or (bank, pn) in self.annex:
                    continue
                rid = r.get("rung")
                if rid not in P.RUNGS:
                    rid = self.hard.get((bank, pn))
                if rid not in P.RUNGS or (bank, pn) in out:
                    continue
                out[(bank, pn)] = (rid, bool(correct), bool(r.get("valid")))
        return out


class GapBootstrap:
    """Paired item bootstrap on the p3 - p1 NCRI gap.

    Items are resampled within each rung (as place.bootstrap_interval does),
    and the SAME resampled items are used for p1 and p3. Pooled draws: every
    item counts once per draw, so a rung's n doubles."""

    def __init__(self, rows, rung_map):
        self.rows = rows
        self.rung_map = rung_map

    def _arrays(self, draws, reading):
        """Item list, rung of each item, and per-setting (k, n) per item."""
        tables = {(s, d): self.rung_map.items(
                      self.rows[(s, d)].official if reading != "answer_first"
                      else self.rows[(s, d)].answer_first)
                  for s in SETTINGS for d in draws}
        keys = sorted(set.intersection(*(set(t) for t in tables.values())))
        rung = [tables[(SETTINGS[0], draws[0])][key][0] for key in keys]
        k, n = {}, {}
        for s in SETTINGS:
            k[s] = np.zeros(len(keys))
            n[s] = np.zeros(len(keys))
            for d in draws:
                t = tables[(s, d)]
                for i, key in enumerate(keys):
                    _rid, correct, valid = t[key]
                    if reading == "ceiling" and not valid:
                        continue
                    k[s][i] += correct
                    n[s][i] += 1
        return keys, rung, k, n

    def run(self, draws, reading, subset):
        keys, rung, k, n = self._arrays(draws, reading)
        keep = [i for i, key in enumerate(keys)
                if not (subset and key[0] in SUBSET_DROP)]
        by_rung = {}
        for i in keep:
            by_rung.setdefault(rung[i], []).append(i)
        groups = [np.array(v) for v in by_rung.values()]
        base = np.concatenate(groups)
        rung_of = [rung[i] for i in range(len(keys))]

        point = {s: self._fast_display(rung_of, base, k[s], n[s]) for s in SETTINGS}
        rng = np.random.default_rng(SEED)
        gaps, disp = [], {s: [] for s in SETTINGS}
        for _ in range(BOOTSTRAP):
            idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
            d = {s: self._fast_display(rung_of, idx, k[s], n[s]) for s in SETTINGS}
            for s in SETTINGS:
                disp[s].append(d[s])
            gaps.append(d["p3"] - d["p1"])
        lo, hi = np.percentile(gaps, [2.5, 97.5])
        return {
            "draws": "+".join(str(d) for d in draws), "reading": reading,
            "domains": "excl_recheck_recon" if subset else "all",
            "n_items": len(base),
            "ncri_p1": round(point["p1"], 2), "ncri_p3": round(point["p3"], 2),
            "p1_lo": round(np.percentile(disp["p1"], 2.5), 2),
            "p1_hi": round(np.percentile(disp["p1"], 97.5), 2),
            "p3_lo": round(np.percentile(disp["p3"], 2.5), 2),
            "p3_hi": round(np.percentile(disp["p3"], 97.5), 2),
            "gap": round(point["p3"] - point["p1"], 2),
            "gap_lo": round(lo, 2), "gap_hi": round(hi, 2),
            "gap_excludes_0": bool(lo > 0 or hi < 0),
        }

    @staticmethod
    def _fast_display(rung_of, idx, k, n):
        counts = {}
        for i in idx:
            rid = rung_of[i]
            kk, nn = counts.get(rid, (0.0, 0.0))
            counts[rid] = (kk + k[i], nn + n[i])
        theta, _ = P.theta_map(counts)
        return P.display(theta)


class ShotEcho:
    """Reasoned rows whose content equals the answer of one of the bank's
    k-shot examples, judged with the repository's own grader."""

    def __init__(self):
        self.shots = {}

    def _shots(self, bank):
        if bank not in self.shots:
            shots, _evals = G.load_bank(bank, G.DATA)
            self.shots[bank] = shots[:K_SHOT]
        return self.shots[bank]

    def is_echo(self, text, bank):
        return any(G.grade(text or "", shot, bank)["is_correct"]
                   for shot in self._shots(bank))

    def table(self, draw_rows):
        """Per bank: echo rate on reasoned rows, and on valid rows (baseline);
        answer-first accuracy of reasoned rows split by echo / non-echo."""
        out = []
        for bank, rows in sorted(draw_rows.official.items()):
            af = {str(r["problem_number"]): r for r in draw_rows.answer_first[bank]}
            reasoned = [r for r in rows if r.get("scorable") and r.get("status") == "reasoned"]
            valid = [r for r in rows if r.get("scorable") and r.get("valid")]
            echo_flags = [self.is_echo(r.get("raw_text"), bank) for r in reasoned]
            af_ok = [bool(af[str(r["problem_number"])]["correct"]) for r in reasoned]
            n_echo = sum(echo_flags)
            out.append({
                "setting": draw_rows.setting, "draw": draw_rows.draw, "bank": bank,
                "n_reasoned": len(reasoned), "n_echo": n_echo,
                "echo_rate": round(n_echo / len(reasoned), 4) if reasoned else "",
                "n_valid": len(valid),
                "echo_rate_valid": (round(sum(self.is_echo(r.get("raw_text"), bank)
                                              for r in valid) / len(valid), 4)
                                    if valid else ""),
                "af_correct_echo": sum(o for o, e in zip(af_ok, echo_flags) if e),
                "af_correct_nonecho": sum(o for o, e in zip(af_ok, echo_flags) if not e),
                "n_nonecho": len(reasoned) - n_echo,
            })
        return out


class PairedItems:
    """Paired accuracy differences over items, with a paired bootstrap."""

    def __init__(self):
        self.rng = np.random.default_rng(SEED + 7)

    def diff(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(x) == 0:
            return {"n": 0, "acc_p1": "", "acc_p3": "", "diff": "", "lo": "", "hi": ""}
        picks = self.rng.integers(0, len(x), size=(BOOTSTRAP, len(x)))
        diffs = y[picks].mean(axis=1) - x[picks].mean(axis=1)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return {"n": len(x), "acc_p1": round(x.mean(), 4), "acc_p3": round(y.mean(), 4),
                "diff": round(y.mean() - x.mean(), 4), "lo": round(lo, 4), "hi": round(hi, 4)}

    @staticmethod
    def by_item(rows, field="correct", only_valid=False):
        return {str(r["problem_number"]): float(bool(r[field])) for r in rows
                if r.get("scorable") and (r.get("valid") or not only_valid)}

    def valid_only(self, rows_p1, rows_p3):
        a = self.by_item(rows_p1, only_valid=True)
        b = self.by_item(rows_p3, only_valid=True)
        keys = sorted(set(a) & set(b))
        return [a[k] for k in keys], [b[k] for k in keys]

    @staticmethod
    def averaged(row_sets):
        """Item -> mean correctness over draws (items present in every draw)."""
        maps = [PairedItems.by_item(rows) for rows in row_sets]
        keys = set.intersection(*(set(m) for m in maps))
        return {k: sum(m[k] for m in maps) / len(maps) for k in keys}


class Figures:
    @staticmethod
    def _style(ax):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)

    def fig1(self, banks, path):
        rows = sorted(banks, key=lambda r: r["af_diff_avg"])
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 9.5), sharey=True,
                                      gridspec_kw={"width_ratios": [3, 2]},
                                      facecolor=SURFACE)
        for a in (ax, ax2):
            self._style(a)
        for i, r in enumerate(rows):
            if r["bank"] in SUBSET_DROP:
                for a in (ax, ax2):
                    a.axhspan(i - 0.45, i + 0.45, color=BAND, zorder=0)
            knowledge = r["kind"] == "knowledge"
            ax.plot([r["af_lo_avg"], r["af_hi_avg"]], [i, i], color=INK_2, lw=2,
                    solid_capstyle="round", zorder=2)
            ax.plot(r["af_diff_avg"], i, "s" if knowledge else "o", ms=8, zorder=3,
                    mfc=SURFACE if knowledge else INK, mec=INK, mew=1.5)
            ax2.plot(r["invalid_p3_avg"], i, "o", ms=6, color=ORANGE, mec=SURFACE, mew=1)
            ax2.plot(r["invalid_p1_avg"], i, "o", ms=10, mfc="none", mec=BLUE, mew=2)
        ax.axvline(0, color=INK_2, lw=1)
        ax2.axvline(20, color=INK_2, lw=1, ls="--")
        ax2.text(20.5, len(rows) - 0.4, "20% cell rule", fontsize=8, color=INK_2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{r['bank']}  (n={r['n']})" for r in rows], color=INK)
        for label, r in zip(ax.get_yticklabels(), rows):
            if r["bank"] in SUBSET_DROP:
                label.set_fontweight("bold")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax2.grid(axis="x", color=GRID, lw=0.8)
        ax.set_xlabel("answer-first accuracy, p3 − p1 (mean of draws 1–2)\n"
                      "95% paired item bootstrap", color=INK_2)
        ax2.set_xlabel("non-valid rows, % (mean of draws 1–2)", color=INK_2)
        ax.set_title("Per-bank difference, answer-first reading (supplementary)\n"
                     "● reasoning bank   □ knowledge bank   shaded = recheck_v2, recon",
                     color=INK, fontsize=10, loc="left")
        ax2.set_title("Invalid rate per bank", color=INK, fontsize=10, loc="left")
        ax2.plot([], [], "o", ms=10, mfc="none", mec=BLUE, mew=2, label="p1")
        ax2.plot([], [], "o", ms=6, color=ORANGE, label="p3")
        ax2.legend(frameon=False, labelcolor=INK, loc="lower left", ncol=2,
                   bbox_to_anchor=(0.45, 1.0))
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def fig2(self, gaps, path):
        readings = (("floor", "floor\n(official)"), ("answer_first", "answer-first\n(supplementary)"),
                    ("ceiling", "ceiling\n(official reading)"))
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, facecolor=SURFACE)
        for ax, domains, title in zip(axes, ("all", "excl_recheck_recon"),
                                      ("All domains", "Excluding recheck_v2 and recon")):
            self._style(ax)
            for x, (reading, label) in enumerate(readings):
                g = gaps[(reading, domains)]
                for offset, s, color in ((-0.13, "p1", BLUE), (0.13, "p3", ORANGE)):
                    ax.plot([x + offset] * 2, [g[f"{s}_lo"], g[f"{s}_hi"]], color=color,
                            lw=2, solid_capstyle="round")
                    ax.plot(x + offset, g[f"ncri_{s}"], "o", ms=8, color=color, mec=SURFACE,
                            mew=2, label=s if x == 0 else None)
                ax.annotate(f"gap {g['gap']:+.1f}\n[{g['gap_lo']:+.1f}, {g['gap_hi']:+.1f}]",
                            (x, max(g["p1_hi"], g["p3_hi"])), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=8, color=INK_2)
            ax.set_xticks(range(len(readings)))
            ax.set_xticklabels([label for _r, label in readings], color=INK, fontsize=9)
            ax.set_xlim(-0.5, len(readings) - 0.5)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_title(title, color=INK, fontsize=10, loc="left")
        axes[0].set_ylim(25, 72)
        axes[0].set_ylabel("NCRI display, draws 1+2 pooled\n"
                           "(95% paired item bootstrap; both UNRANKED under the 0.20 rule)",
                           color=INK_2)
        axes[0].legend(frameon=False, labelcolor=INK, loc="upper left", ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def fig3(self, echo_banks, path):
        rows = sorted([r for r in echo_banks if r["n_reasoned_avg"] >= 1],
                      key=lambda r: r["echo_p3_avg"] if r["echo_p3_avg"] != "" else -1)
        fig, ax = plt.subplots(figsize=(8, 7.5), facecolor=SURFACE)
        self._style(ax)
        for i, r in enumerate(rows):
            if r["echo_p3_avg"] != "":
                ax.plot(100 * r["echo_p3_avg"], i, "o", ms=6, color=ORANGE, mec=SURFACE, mew=1)
            if r["echo_p1_avg"] != "":
                ax.plot(100 * r["echo_p1_avg"], i, "o", ms=10, mfc="none", mec=BLUE, mew=2)
        ax.plot([], [], "o", ms=10, mfc="none", mec=BLUE, mew=2, label="p1")
        ax.plot([], [], "o", ms=6, color=ORANGE, label="p3")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{r['bank']}  (reasoned p1/p3: {r['n_p1']:.0f}/{r['n_p3']:.0f})"
                            for r in rows], color=INK, fontsize=8)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_xlim(-3, 103)
        ax.set_xlabel("reasoned rows whose content is a k-shot answer, % (mean of draws 1–2)",
                      color=INK_2)
        ax.set_title("Shot echo in reasoned rows, per bank\n"
                     "(no marker = no reasoned rows at that setting)",
                     color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, labelcolor=INK, loc="lower right")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = {}
    for d in GRADED_DIRS:
        for s in SETTINGS:
            dr = DrawRows(s, d)
            if dr.exists():
                rows[(s, d)] = dr
    draws = sorted({d for (_s, d) in rows})
    print("draws available:", draws)

    # 1. Gap intervals.
    boot = GapBootstrap(rows, RungMap())
    draw_sets = [[d] for d in draws] + [[1, 2]]
    gap_rows = []
    for ds in draw_sets:
        for reading in ("floor", "answer_first", "ceiling"):
            for subset in (False, True):
                gap_rows.append(boot.run(ds, reading, subset))
                print(gap_rows[-1])
    write_csv(os.path.join(OUT_DIR, "gap_intervals.csv"), gap_rows)

    # 2. Shot echo.
    echo = ShotEcho()
    echo_rows = [r for (s, d), dr in sorted(rows.items()) for r in echo.table(dr)]
    write_csv(os.path.join(OUT_DIR, "shot_echo.csv"), echo_rows)

    # 3 and 4. Paired valid-only, per-bank averages, groups.
    paired = PairedItems()
    valid_rows, bank_rows, echo_banks = [], [], []
    banks = sorted(rows[("p1", 1)].official)
    kinds = {}
    ncri_banks = {key.split(":")[0] for key in P.RUNGS}
    for bank in banks:
        kinds[bank] = "ncri" if bank in ncri_banks else "knowledge"
        for d in draws:
            x, y = paired.valid_only(rows[("p1", d)].official[bank], rows[("p3", d)].official[bank])
            total = len(PairedItems.by_item(rows[("p1", d)].official[bank]))
            r = paired.diff(x, y)
            valid_rows.append({"bank": bank, "kind": kinds[bank], "draw": d,
                               "n_items": total, "n_valid_both": r["n"], **{k: r[k] for k in
                               ("acc_p1", "acc_p3", "diff", "lo", "hi")}})

        def avg_pair(attr):
            m1 = PairedItems.averaged([getattr(rows[("p1", d)], attr)[bank] for d in (1, 2)])
            m3 = PairedItems.averaged([getattr(rows[("p3", d)], attr)[bank] for d in (1, 2)])
            keys = sorted(set(m1) & set(m3))
            return paired.diff([m1[k] for k in keys], [m3[k] for k in keys])

        def one_draw(attr, d):
            a = PairedItems.by_item(getattr(rows[("p1", d)], attr)[bank])
            b = PairedItems.by_item(getattr(rows[("p3", d)], attr)[bank])
            keys = sorted(set(a) & set(b))
            return np.mean([b[k] for k in keys]) - np.mean([a[k] for k in keys])

        def invalid(s, d):
            sc = [r for r in rows[(s, d)].official[bank] if r.get("scorable")]
            return 100.0 * sum(bool(r.get("invalid")) for r in sc) / len(sc)

        fl, af = avg_pair("official"), avg_pair("answer_first")
        bank_rows.append({
            "bank": bank, "kind": kinds[bank], "n": fl["n"],
            "floor_diff_d1": round(one_draw("official", 1), 4),
            "floor_diff_d2": round(one_draw("official", 2), 4),
            "floor_diff_avg": fl["diff"], "floor_lo_avg": fl["lo"], "floor_hi_avg": fl["hi"],
            "floor_change_d2_d1": round(one_draw("official", 2) - one_draw("official", 1), 4),
            "af_diff_d1": round(one_draw("answer_first", 1), 4),
            "af_diff_d2": round(one_draw("answer_first", 2), 4),
            "af_diff_avg": af["diff"], "af_lo_avg": af["lo"], "af_hi_avg": af["hi"],
            "af_change_d2_d1": round(one_draw("answer_first", 2) - one_draw("answer_first", 1), 4),
            "invalid_p1_avg": round((invalid("p1", 1) + invalid("p1", 2)) / 2, 1),
            "invalid_p3_avg": round((invalid("p3", 1) + invalid("p3", 2)) / 2, 1),
            "invalid_p1_d3": round(invalid("p1", 3), 1) if ("p1", 3) in rows else "",
            "invalid_p3_d3": round(invalid("p3", 3), 1) if ("p3", 3) in rows else "",
        })

        def echo_avg(s):
            sel = [e for e in echo_rows if e["bank"] == bank and e["setting"] == s
                   and e["draw"] in (1, 2)]
            n = sum(e["n_reasoned"] for e in sel)
            return (round(sum(e["n_echo"] for e in sel) / n, 4) if n else ""), n / 2
        e1, n1 = echo_avg("p1")
        e3, n3 = echo_avg("p3")
        echo_banks.append({"bank": bank, "echo_p1_avg": e1, "echo_p3_avg": e3,
                           "n_p1": n1, "n_p3": n3, "n_reasoned_avg": n1 + n3})
    write_csv(os.path.join(OUT_DIR, "paired_valid.csv"), valid_rows)
    write_csv(os.path.join(OUT_DIR, "banks_avg.csv"), bank_rows)

    # Pooled valid-only and group comparisons (items pooled across banks).
    group_rows = []
    for group in ("ncri", "knowledge", "all"):
        members = [b for b in banks if group == "all" or kinds[b] == group]
        for d in draws:
            xs, ys = [], []
            for b in members:
                x, y = paired.valid_only(rows[("p1", d)].official[b], rows[("p3", d)].official[b])
                xs += x
                ys += y
            r = paired.diff(xs, ys)
            group_rows.append({"comparison": "valid_only", "group": group, "draws": str(d), **r})
        for attr, name in (("official", "floor"), ("answer_first", "answer_first")):
            xs, ys = [], []
            for b in members:
                m1 = PairedItems.averaged([getattr(rows[("p1", d)], attr)[b] for d in (1, 2)])
                m3 = PairedItems.averaged([getattr(rows[("p3", d)], attr)[b] for d in (1, 2)])
                keys = sorted(set(m1) & set(m3))
                xs += [m1[k] for k in keys]
                ys += [m3[k] for k in keys]
            group_rows.append({"comparison": name, "group": group, "draws": "1+2 mean",
                               **paired.diff(xs, ys)})
    write_csv(os.path.join(OUT_DIR, "groups.csv"), group_rows)

    # 5. Figures.
    figures = Figures()
    figures.fig1(bank_rows, os.path.join(OUT_DIR, "fig1.png"))
    pooled = {(g["reading"], g["domains"]): g for g in gap_rows if g["draws"] == "1+2"}
    figures.fig2(pooled, os.path.join(OUT_DIR, "fig2.png"))
    figures.fig3(echo_banks, os.path.join(OUT_DIR, "fig3.png"))

    # Stop conditions, printed for the report.
    def echo_rate(s):
        sel = [e for e in echo_rows if e["setting"] == s and e["draw"] in (1, 2)]
        return sum(e["n_echo"] for e in sel) / sum(e["n_reasoned"] for e in sel)
    print("STOP? echo p1 %.3f p3 %.3f" % (echo_rate("p1"), echo_rate("p3")))
    for g in group_rows:
        print("GROUP", g)


if __name__ == "__main__":
    main()
