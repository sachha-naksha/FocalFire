"""Does fixing ISSUES #1 and #3 change the episodic TF enrichment?

Design
------
For every (branch, episode) the expensive prefix -- smoothing, ``build_episode_grn``,
``filter_edges_by_significance_and_direction`` -- is computed **once**. Only then does
the run fork into two variants that differ solely in the two fixed functions:

    legacy : pre-fix TF-expression slice (ISSUES #3) + pre-fix positional force
             alignment (ISSUES #1), from ``legacy.py``
    fixed  : the current ``focal.temporal`` code

Both variants then go through *identical* downstream code (percentile edge selection,
LF annotation, hypergeometric ORA), so any difference in the enrichment tables is
attributable to the fix and not to a rerun, a reseed, or a parameter change.

Outputs (written to ``--outdir``, default ``./results``)
-------------------------------------------------------
    enrichment_<branch>_ep<i>_<variant>.csv   full ORA table per variant
    comparison_<branch>_ep<i>.csv             per-TF side-by-side join
    summary.csv                               one row per (branch, episode)
    summary.md                                human-readable verdict
    run_config.json                           exactly what was run

Usage
-----
    python run_validation.py --outdir results
See ``submit.sbatch`` for the cluster invocation.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from focal.base.enrichment import calculate_tf_episodic_enrichment
from focal.temporal._episodes import EpisodeDynamics
from focal.temporal._forces import calculate_force_curves_parallel
from legacy import legacy_calculate_force_curves, legacy_compute_tf_expression
from slim_loader import load_slim

# --------------------------------------------------------------------------- #
# Configuration                                                                #
#                                                                              #
# NOTE ON PROVENANCE. The committed enrichment_ep{i}_{pb,gc}.csv files (Jun 2025)  #
# do not record the settings that produced them.                                   #
#                                                                                  #
#   * The LATENT FACTOR is now established, not guessed -- see DEFAULT_LF_FILES.    #
#   * The trajectory ranges and episode geometry remain a reconstruction:           #
#       PB = (0, 2), GC = (0, 3) are the post-bifurcation branches (node 0 is the   #
#       bifurcation, node 1 the root; the entire range (1, x) takes 40 points);     #
#       num_points=20 with points_per_episode=5 gives exactly the 4 episodes on     #
#       disk, with the q = 5 used everywhere else in the codebase.                  #
#                                                                                  #
# The legacy-vs-fixed verdict does NOT depend on the reconstructed half, because    #
# both variants are run on identical inputs. Override with the CLI flags to compare #
# against the committed CSVs on their own terms.                                    #
# --------------------------------------------------------------------------- #

BCELL = "/work/nvme/bhdw/asachan/data_files/firefate/bcell"
DEFAULT_DYNAMIC_H5 = f"{BCELL}/outs/dynamic.h5"
DEFAULT_REFERENCE_DIR = f"{BCELL}/outs/intermediate_tmp_files/direct_effect_enrichment"

# The cellular program behind the committed enrichment_ep{i}_{pb,gc}.csv files is the
# UNION of the two state-discriminative latent factors, HLA- genes dropped -- exactly
# the (commented-out) `lf_genes = list(set(z11_genes + z3_genes))` line in
# Fig3_1_LF_local_dynamics.ipynb.
#
# Established by inversion rather than assumed: the 57 distinct genes appearing in
# `genes_in_lf` across all eight committed CSVs are covered 100% by Z11 u Z3 (57 genes
# after dropping HLA-), with zero union members never observed. Neither factor alone
# fits -- Z11 covers 59.6%, Z3 covers 50.9%.
DEFAULT_LF_FILES = ",".join(
    [
        f"{BCELL}/latent_factors/feature_list_Z11_GC_PB.txt",
        f"{BCELL}/latent_factors/feature_list_Z3_GC_PB.txt",
    ]
)

BRANCHES = {"pb": (0, 2), "gc": (0, 3)}


def load_lf_genes(paths: str) -> list[str]:
    """Union of one or more SLIDE feature lists, ``HLA-`` genes dropped as in the notebooks.

    ``paths`` is a comma-separated list of ``feature_list_*.txt`` TSVs.
    """
    genes: set[str] = set()
    for path in [p.strip() for p in paths.split(",") if p.strip()]:
        df = pd.read_csv(path, sep="\t", header=0)
        df = df[~df["names"].str.contains("HLA-")]
        n_before = len(genes)
        genes |= set(df["names"].tolist())
        print(
            f"[config]   {os.path.basename(path)}: {len(df)} genes "
            f"(+{len(genes) - n_before} new)",
            flush=True,
        )
    return sorted(genes)


def enrich_from_avg_force(
    avg_force_df: pd.DataFrame, lf_genes: list[str], percentile: float
) -> tuple[pd.DataFrame, dict]:
    """Percentile edge selection -> LF annotation -> hypergeometric ORA.

    Replicates ``EpisodeDynamics.select_top_edges`` + ``annotate_lf_in_grn`` +
    ``calculate_enrichment`` so both variants share one code path.
    """
    threshold = np.percentile(np.abs(avg_force_df["avg_force"]), percentile)
    edges = avg_force_df[np.abs(avg_force_df["avg_force"]) >= threshold].copy()
    edges["is_in_lf"] = edges.index.get_level_values(1).isin(lf_genes)

    lf_active = edges[edges["is_in_lf"]].index.get_level_values(1).unique()
    all_targets = edges.index.get_level_values(1).unique()

    enr = calculate_tf_episodic_enrichment(
        edges, total_lf_genes=len(lf_active), total_genes_in_grn=len(all_targets)
    )
    enr = enr.sort_values(by="enrichment_score", ascending=False)
    enr = enr[enr["enrichment_score"] != 0].reset_index(drop=True)

    stats = {
        "force_threshold": float(threshold),
        "n_edges_selected": int(len(edges)),
        "n_tfs_in_grn": int(edges.index.get_level_values(0).nunique()),
        "n_targets_in_grn": int(len(all_targets)),
        "n_lf_active": int(len(lf_active)),
    }
    return enr, stats


def compare(fixed: pd.DataFrame, legacy: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, dict]:
    """Side-by-side per-TF join plus scalar agreement metrics."""
    f = fixed.set_index("TF")[["p_value", "enrichment_score", "genes_in_lf"]]
    l = legacy.set_index("TF")[["p_value", "enrichment_score", "genes_in_lf"]]
    joined = f.join(l, how="outer", lsuffix="_fixed", rsuffix="_legacy")
    joined["es_delta"] = joined["enrichment_score_fixed"] - joined["enrichment_score_legacy"]
    joined = joined.sort_values("enrichment_score_fixed", ascending=False)

    sig_f = set(fixed.loc[fixed["p_value"] <= alpha, "TF"])
    sig_l = set(legacy.loc[legacy["p_value"] <= alpha, "TF"])
    union = sig_f | sig_l
    both = joined.dropna(subset=["enrichment_score_fixed", "enrichment_score_legacy"])

    rho = np.nan
    if len(both) >= 3:
        rho = float(
            both["enrichment_score_fixed"].corr(both["enrichment_score_legacy"], method="spearman")
        )

    top20_f = list(fixed.sort_values("enrichment_score", ascending=False)["TF"].head(20))
    top20_l = list(legacy.sort_values("enrichment_score", ascending=False)["TF"].head(20))

    metrics = {
        "n_tfs_fixed": int(len(fixed)),
        "n_tfs_legacy": int(len(legacy)),
        f"n_sig_fixed_p{alpha}": int(len(sig_f)),
        f"n_sig_legacy_p{alpha}": int(len(sig_l)),
        "n_sig_shared": int(len(sig_f & sig_l)),
        "n_sig_only_fixed": int(len(sig_f - sig_l)),
        "n_sig_only_legacy": int(len(sig_l - sig_f)),
        "n_sig_union": int(len(union)),
        # Both sets empty is genuine agreement, not a missing value.
        "sig_jaccard": float(len(sig_f & sig_l) / len(union)) if union else 1.0,
        "es_spearman_shared": rho,
        "top20_overlap": int(len(set(top20_f) & set(top20_l))),
        "tfs_only_fixed": ";".join(sorted(sig_f - sig_l)[:25]),
        "tfs_only_legacy": ";".join(sorted(sig_l - sig_f)[:25]),
    }
    return joined.reset_index(), metrics


def run_branch(
    branch: str, traj_range: tuple, args, lf_genes: list[str], outdir: Path
) -> list[dict]:
    print(f"\n{'=' * 78}\n[{branch}] trajectory_range={traj_range}\n{'=' * 78}", flush=True)
    t0 = time.time()
    net = load_slim(args.dynamic_h5)
    print(f"[{branch}] network loaded in {time.time() - t0:.0f}s", flush=True)

    epi = EpisodeDynamics(
        dictys_dynamic_object=net,
        output_folder=str(outdir),
        trajectory_range=traj_range,
        num_points=args.num_points,
        dist=args.dist,
        sparsity=args.sparsity,
    )

    t0 = time.time()
    lcpm, dtime = epi.compute_expression_curves()
    print(
        f"[{branch}] expression curves {lcpm.shape} in {time.time() - t0:.0f}s",
        flush=True,
    )

    rows = []
    n_episodes = args.num_points // args.points_per_episode
    for ep in range(1, n_episodes + 1):
        sl = slice((ep - 1) * args.points_per_episode, ep * args.points_per_episode)
        tag = f"{branch}_ep{ep}"
        print(f"\n--- [{tag}] time_slice={sl} ---", flush=True)

        t0 = time.time()
        beta = epi.build_episode_grn(time_slice=sl)
        print(f"[{tag}] episode GRN {beta.shape} in {time.time() - t0:.0f}s", flush=True)

        t0 = time.time()
        filtered = epi.filter_edges(
            n_processes=args.n_processes,
            chunk_size=args.filter_chunk_size,
            pval_threshold=args.pval_threshold,
        )
        print(
            f"[{tag}] filtered edges {filtered.shape} in {time.time() - t0:.0f}s",
            flush=True,
        )
        if len(filtered) == 0:
            print(f"[{tag}] no edges survived the filter; skipping", flush=True)
            continue

        beta_for_force = filtered.drop("p_value", axis=1)

        # ---------------- fixed (current library) ----------------
        tf_fixed = epi.compute_tf_expression()
        force_fixed = calculate_force_curves_parallel(
            beta_curves=beta_for_force,
            tf_expression=tf_fixed,
            n_processes=args.n_processes,
            chunk_size=args.force_chunk_size,
        )
        avg_fixed = force_fixed.mean(axis=1).to_frame(name="avg_force")
        enr_fixed, st_fixed = enrich_from_avg_force(avg_fixed, lf_genes, args.percentile)

        # ---------------- legacy (pre-fix) ----------------
        tf_legacy = legacy_compute_tf_expression(epi.lcpm_dcurve, filtered)
        force_legacy = legacy_calculate_force_curves(
            beta_for_force, tf_legacy, chunk_size=args.force_chunk_size
        )
        avg_legacy = force_legacy.mean(axis=1).to_frame(name="avg_force")
        enr_legacy, st_legacy = enrich_from_avg_force(avg_legacy, lf_genes, args.percentile)

        # how different were the inputs, before enrichment?
        expr_identical = bool(np.allclose(tf_fixed.values, tf_legacy.values, equal_nan=True))
        force_corr = float(
            pd.Series(avg_fixed["avg_force"].values).corr(
                pd.Series(avg_legacy["avg_force"].values), method="spearman"
            )
        )

        enr_fixed.to_csv(outdir / f"enrichment_{tag}_fixed.csv", index=False)
        enr_legacy.to_csv(outdir / f"enrichment_{tag}_legacy.csv", index=False)

        joined, metrics = compare(enr_fixed, enr_legacy, args.alpha)
        joined.to_csv(outdir / f"comparison_{tag}.csv", index=False)

        row = {
            "branch": branch,
            "episode": ep,
            "time_slice": f"{sl.start}:{sl.stop}",
            "n_edges_episode": int(len(beta)),
            "n_edges_filtered": int(len(filtered)),
            "tf_expression_identical": expr_identical,
            "avg_force_spearman": force_corr,
            **{f"fixed_{k}": v for k, v in st_fixed.items()},
            **{f"legacy_{k}": v for k, v in st_legacy.items()},
            **metrics,
        }
        rows.append(row)
        print(
            f"[{tag}] TFs fixed={metrics['n_tfs_fixed']} legacy={metrics['n_tfs_legacy']}  "
            f"sig-Jaccard={metrics['sig_jaccard']:.3f}  "
            f"ES-spearman={metrics['es_spearman_shared']:.3f}  "
            f"top20-overlap={metrics['top20_overlap']}/20",
            flush=True,
        )

        del beta, filtered, beta_for_force, force_fixed, force_legacy
        gc.collect()

    del net, epi
    gc.collect()
    return rows


def write_summary(df: pd.DataFrame, outdir: Path, args) -> None:
    lines = [
        "# Episodic enrichment: pre-fix vs post-fix",
        "",
        "Both variants were computed from **identical** smoothed networks and identical",
        "filtered edge sets; they differ only in ISSUES #1 (force/expression alignment)",
        "and #3 (episode time-slice for regulator expression).",
        "",
        f"- `dynamic.h5`: `{args.dynamic_h5}`",
        f"- LF gene set: union of `{args.lf_files}` "
        f"({args.n_lf_genes} genes after dropping `HLA-`)",
        f"- num_points={args.num_points}, points_per_episode={args.points_per_episode}, "
        f"dist={args.dist}, sparsity={args.sparsity}",
        f"- percentile={args.percentile}, pval_threshold={args.pval_threshold}, "
        f"significance alpha={args.alpha}",
        "",
        "## Per-episode agreement",
        "",
        "| branch | ep | edges filtered | TFs fixed | TFs legacy | sig∩ | only fixed | "
        "only legacy | Jaccard | ES ρ | top20 ∩ | expr identical |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['branch'].upper()} | {r['episode']} | {r['n_edges_filtered']:,} | "
            f"{r['n_tfs_fixed']} | {r['n_tfs_legacy']} | {r['n_sig_shared']} | "
            f"{r['n_sig_only_fixed']} | {r['n_sig_only_legacy']} | "
            f"{r['sig_jaccard']:.3f} | {r['es_spearman_shared']:.3f} | "
            f"{r['top20_overlap']}/20 | {r['tf_expression_identical']} |"
        )

    changed = df[(df["sig_jaccard"] < 1.0) | (df["top20_overlap"] < 20)]
    lines += [
        "",
        "## Verdict",
        "",
        f"- episodes compared: **{len(df)}**",
        f"- episodes where the enriched-TF set changed: **{len(changed)}**",
        f"- mean significant-TF Jaccard: **{np.nanmean(df['sig_jaccard']):.3f}**",
        f"- mean ES Spearman (TFs present in both): "
        f"**{np.nanmean(df['es_spearman_shared']):.3f}**",
        f"- mean top-20 overlap: **{df['top20_overlap'].mean():.1f}/20**",
        f"- episodes with no significant TF in either variant: "
        f"**{int((df['n_sig_union'] == 0).sum())}**",
        "",
    ]
    if len(changed):
        lines.append("TFs gained or lost at the significance threshold, per episode:")
        lines.append("")
        for _, r in changed.iterrows():
            lines.append(f"- **{r['branch'].upper()} ep{r['episode']}**")
            if r["tfs_only_fixed"]:
                lines.append(f"  - only after fix: `{r['tfs_only_fixed']}`")
            if r["tfs_only_legacy"]:
                lines.append(f"  - only before fix: `{r['tfs_only_legacy']}`")
    else:
        lines.append("The fix did not change the enriched TF set in any episode.")

    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dynamic-h5", default=DEFAULT_DYNAMIC_H5)
    p.add_argument("--lf-files", default=DEFAULT_LF_FILES,
                   help="comma-separated feature_list_*.txt TSVs; their union is the program")
    p.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "results"))
    p.add_argument("--branches", default="pb,gc")
    p.add_argument("--num-points", type=int, default=20)
    p.add_argument("--points-per-episode", type=int, default=5)
    p.add_argument("--dist", type=float, default=0.001)
    p.add_argument("--sparsity", type=float, default=0.01)
    p.add_argument("--percentile", type=float, default=98.0)
    p.add_argument("--pval-threshold", type=float, default=0.001)
    p.add_argument("--alpha", type=float, default=0.05, help="significance cut for the TF-set comparison")
    p.add_argument("--n-processes", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    p.add_argument("--filter-chunk-size", type=int, default=8000)
    p.add_argument("--force-chunk-size", type=int, default=30000)
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lf_genes = load_lf_genes(args.lf_files)
    args.n_lf_genes = len(lf_genes)
    print(f"[config] {len(lf_genes)} LF genes (union, HLA- dropped)", flush=True)
    print(f"[config] n_processes={args.n_processes}", flush=True)
    (outdir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    rows: list[dict] = []
    for branch in args.branches.split(","):
        branch = branch.strip()
        if branch not in BRANCHES:
            raise ValueError(f"Unknown branch {branch!r}; expected one of {list(BRANCHES)}.")
        rows += run_branch(branch, BRANCHES[branch], args, lf_genes, outdir)

    if not rows:
        print("No episodes produced results.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "summary.csv", index=False)
    write_summary(df, outdir, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
