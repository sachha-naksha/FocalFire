# Does the episodic-force fix change the TF enrichment?

Validation harness for two `CONFIRMED` defects from [`../temporal/ISSUES.md`](../temporal/ISSUES.md),
both fixed in `src/focal/temporal/`:

| # | where | what was wrong | fix |
|---|---|---|---|
| **1** | `_forces.py :: calculate_force_curves_chunk` | TF expression was ordered by `value_counts()` (descending target count) and attached to rows grouped in dictys `nids[0]` (alphabetical) order. The two agree with probability $1/n!$, so most edges were scaled by **another TF's** expression. | reindex expression onto the row-level TF labels (`tf_expression.reindex(row_tfs)`), raising on a missing regulator |
| **3** | `_episodes.py :: compute_tf_expression` | always sliced `iloc[:, 0:n_time_cols]` — the **first** $q$ pseudotime points of the branch — then relabelled them `time_0..`, so every episode after the first used **episode-1** regulator expression. | `build_episode_grn` records `self.time_slice`; `compute_tf_expression` slices with it and checks the width |

Both defects sit *downstream* of the smoothing and the edge filter, so they change
forces and therefore the percentile edge selection and the enrichment — but not the
episodic GRN's edge membership before force ranking.

## Design

For every (branch, episode) the expensive prefix — smoothing, `build_episode_grn`,
`filter_edges_by_significance_and_direction` — is computed **once**, then the run forks:

```
                     ┌─ fixed  : current focal.temporal
smoothing → episode  │
GRN → edge filter ───┤
                     └─ legacy : legacy.py (verbatim pre-fix functions)
                                 ↓
              identical downstream: percentile selection → LF annotation → hypergeometric ORA
```

Both variants go through the *same* downstream code, so a difference in the enrichment
tables is attributable to the fix — not to a rerun, a reseed, or a parameter change.
This is deliberately **not** a comparison against the committed
`enrichment_ep{i}_{pb,gc}.csv` files, which would confound the fix with any parameter
drift since June.

## Files

| file | role |
|---|---|
| `run_validation.py` | the driver; writes every output below |
| `legacy.py` | verbatim pre-fix `calculate_force_curves_chunk` + `compute_tf_expression`, including the same balanced chunking (chunk boundaries change which TFs co-occur, and therefore how badly the old ordering skewed) |
| `slim_loader.py` | memory-lean `dynamic.h5` loader — see below |
| `submit.sbatch` | SLURM submission |
| `results/` | outputs |

### Why `slim_loader.py` exists

`dictys.net.network.from_file` reads **every** dataset under `prop` eagerly. For the
B-cell network (551 TFs × 11 907 targets × 194 windows) that is ~35 GB before any
smoothing: `es/w`, `es/w_in`, `es/w_n` at 10.2 GB each, three boolean masks, and
`nc/readcount`. The episodic pipeline only touches `es/w`, `ns/cpm` and the
trajectory/point objects, and `network.check()` validates only the shapes of whatever
is present — so skipping the rest is safe and drops peak resident to ~11 GB, which is
what makes the 64 GB request comfortable.

> The returned object is deliberately **wrong** for anything that reads `w_in` or the
> edge masks (force waves, `TFForceValidation`). It is a harness convenience only.

## Which network variable the episodic path uses

Worth recording, because it is easy to misread: `dictys.net.stat.net` defaults to
**`varname='w'`** — the raw *direct-effect* network. `EpisodeDynamics.build_episode_grn`
and `SmoothedCurvesGRN._regulation_curves_parallel` both call `stat.net(obj)` with no
`varname`, so **episodic GRNs are built on `w`**, while `get_beta_curves` (force waves,
validation) defaults to `varname='w_in'`, the normalised *total-effect* network. The
comment in `Fig3_1_LF_local_dynamics.ipynb` cell 19 claiming `w_in` is wrong for that
cell.

## Parameters, and one provenance gap

The repository does **not** record the settings that produced the committed
`enrichment_ep{i}_{pb,gc}.csv` files. One of them has since been pinned down; the rest
remain a documented reconstruction.

### The cellular program — established, not assumed

The program is the **union of Z11 and Z3 (GC_PB)** with `HLA-` genes dropped (57 genes),
i.e. the commented-out `lf_genes = list(set(z11_genes + z3_genes))` line in
`Fig3_1_LF_local_dynamics.ipynb`. Recovered by inverting the committed outputs:

| candidate | n | covers the 57 genes seen in `genes_in_lf` across all 8 CSVs |
|---|---|---|
| `Z11_GC_PB` alone | 34 | 59.6 % |
| `Z3_GC_PB` alone | 29 | 50.9 % |
| **`Z11 ∪ Z3`** | **57** | **100 %**, and every union member is observed |
| any `*_KO` factor | ~40 | ≤ 5.3 % |

### The published GRN scale — also recovered, and usable as a calibration target

`recover_published_params.py` inverts the two hypergeometric population parameters out
of the committed CSVs. For each TF row, with $k = |\texttt{genes\_in\_lf}|$ and
$n = k + |\texttt{genes\_dwnstrm}|$,

$$\mathrm{ES} = \frac{k}{nK/N} \quad\Longrightarrow\quad \frac{N}{K} = \frac{\mathrm{ES}\cdot n}{k},$$

so **every row independently determines $N/K$**; the absolute scale then follows from
$p = \texttt{hypergeom.sf}(k-1, N, K, n)$ by scanning integer $K$, which is bounded above
by the 57-gene program, so the scan covers the whole feasible space. The result is an
exact inversion, not a fit — `ratio_cv` ~ 10⁻¹⁶ and `max_abs_dlog10p` ~ 10⁻¹⁴:

| branch | ep | $N$ targets in episodic GRN | $K$ program genes active | median TF out-degree |
|---|---|---|---|---|
| PB | 1 | 4414 | 38 | 58.5 |
| PB | 2 | 3912 | 44 | 24 |
| PB | 3 | 3921 | 39 | 24 |
| PB | 4 | 4731 | 44 | 25 |
| GC | 1 | 4771 | 46 | 59 |
| GC | 2 | 3856 | 40 | 54 |
| GC | 3 | 5298 | 45 | 41.5 |
| GC | 4 | 5696 | 47 | 49 |

Written to `published_params.csv`. **A rerun whose trajectory range, `num_points`,
`points_per_episode` and `percentile` are right should land in the same range**
(3900–5700 targets, 38–47 active program genes). `summary.csv` reports
`fixed_n_targets_in_grn` and `fixed_n_lf_active` for exactly this comparison — if they
come out an order of magnitude off, the reconstruction below is wrong even though the
legacy-vs-fixed verdict still stands.

### Still reconstructed

- `PB = (0, 2)`, `GC = (0, 3)` — the post-bifurcation branches (node 0 is the
  bifurcation, node 1 the root; a single branch is sampled at 20 points, the entire
  range `(1, x)` at 40);
- `num_points=20`, `points_per_episode=5` → exactly the 4 episodes on disk, with the
  `q = 5` used everywhere else in the codebase;
- `percentile=98`, `pval_threshold=1e-3`.

**The legacy-vs-fixed verdict does not depend on these**, since both variants see
identical inputs. If you know the original settings, pass them as flags and rerun:

```bash
python run_validation.py --num-points 40 --points-per-episode 10 --percentile 84
python run_validation.py --lf-files /path/to/feature_list_A.txt,/path/to/feature_list_B.txt
```

## Running

```bash
cd Focal/tests/episodic_fix_validation
sbatch submit.sbatch                 # 4 cpus, 64G, 8h, account bhdw-delta-cpu
```

Resources match the interactive `srun` template except `--time`, raised from 2 h to 8 h:
the one-sample *t*-test in `filter_edges_by_significance_and_direction` is a Python loop
over ~10⁶ edges per episode and runs 8 times (2 branches × 4 episodes). Raising
`--cpus-per-task` shortens it roughly linearly — the driver reads `SLURM_CPUS_PER_TASK`.

Locally, on a smaller slice:

```bash
python run_validation.py --branches pb --num-points 5 --points-per-episode 5
```

## Outputs (`results/`)

| file | contents |
|---|---|
| `enrichment_<branch>_ep<i>_fixed.csv` | full ORA table, post-fix — same schema as the production `enrichment_*.csv` |
| `enrichment_<branch>_ep<i>_legacy.csv` | the same, pre-fix |
| `comparison_<branch>_ep<i>.csv` | per-TF join: `p_value`/`enrichment_score`/`genes_in_lf` for both variants plus `es_delta` |
| `summary.csv` | one row per (branch, episode): edge counts, TF counts, significant-set overlap, ES Spearman, top-20 overlap |
| `summary.md` | the verdict, including which TFs are gained or lost at the significance threshold |
| `run_config.json` | exactly what was run |

`summary.csv` also carries two diagnostics that isolate *which* defect bit:

- `tf_expression_identical` — `False` for every episode except the first proves defect #3
  was active (episode 1 has `time_slice = 0:q`, where old and new agree by construction);
- `avg_force_spearman` — how far defect #1 moved the edge ranking that the 98th-percentile
  selection acts on.

## Test-suite changes that came with the fix

Per the protocol in `ISSUES.md` ("a fix turns the `xfail` into an `XPASS` failure — that
is the prompt to delete the `currently` test and drop the marker"):

- promoted to ordinary assertions: `test_expression_is_taken_from_the_episode_s_own_time_window`,
  `test_expression_is_matched_to_the_right_regulator` (`test_episodes.py`),
  `test_expression_is_matched_by_tf_name_not_by_row_order` (`test_episodic_units.py`);
- deleted: `test_currently_reuses_the_first_time_points_for_every_episode`,
  `test_currently_swaps_expression_between_tfs`.

`pytest tests/temporal -q` → **307 passed, 3 xfailed** (the remaining xfails are
ISSUES #2, #4 and #5, which are untouched).
