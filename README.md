# Air Computing Pool (ACP) — GPU Sharing Simulation

Campus-scale GPU sharing simulator for evaluating scheduling policies under heterogeneous workloads.  
Compares **No Sharing / FCFS / Owner Priority / Preemptive** across load levels, workload mixes, and participation incentives.

---

## Overview

| Feature | Detail |
|---|---|
| Users | 18 (6 Low / 6 Mid / 6 High tier) |
| GPU tiers | 9 levels (GTX 1650 → A100) |
| Scheduling methods | No Sharing, FCFS, Owner Priority, Preemptive |
| Key metric | Turn-Around Time (TAT) per user / tier group |
| Participation model | Iterative rational-agent decision (join if shared TAT ≤ standalone TAT) |

---

## Repository Structure

```
simulation/
├── config.py                          # Global settings (users, GPU tiers, arrival rates)
├── definitions.py                     # Task / GPU / User data classes
├── task_patterns.py                   # Task arrival & size generation (Poisson + log-normal)
│
├── simulation_no_sharing.py           # Baseline: each user uses own GPU only
├── simulation_with_sharing.py         # FCFS sharing
├── simulation_with_sharing_owner_priority.py    # Owner-priority sharing
├── simulation_with_sharing_owner_preemption.py  # Preemptive owner-priority sharing
│
├── simulation_iterative_wrapper.py    # Iterative participation optimizer
│
├── run_custom_user_arrival_web.py     # Multi-load sweep (called by Streamlit UI)
├── run_random_hetero_fixed_load_web.py  # Heterogeneous workload sweep (100 trials/load)
├── run_participation_cascade.py       # Participation cascade at fixed load=0.8
├── run_hetero_scenarios.py            # 4 heterogeneous workload scenarios
│
├── plot_paper_graphs_from_csv.py      # Band plots (mean ± min/max) from trial CSV
├── plot_paper_figures.py              # Paper-ready PDF figure generation
│
└── results.py                         # Statistics & visualization helpers

streamlit_app.py                       # Web UI (scenario runner + result viewer)
requirements.txt
```

---

## Scheduling Methods

| Method | Description |
|---|---|
| **No Sharing** | Each user runs tasks only on their own GPU |
| **FCFS** | Shared pool, first-come-first-served |
| **Owner Priority** | Owner's tasks always jump ahead of guests |
| **Preemptive** | Owner's tasks preempt running guest tasks immediately |

**Protection Ratio** = TAT (shared) / TAT (No Sharing).  
≤ 1.0 means the owner is not disadvantaged by sharing.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The web UI provides three execution modes:

| Menu | Script | Description |
|---|---|---|
| Load sweep (uniform) | `run_custom_user_arrival_web.py` | 4 methods × multiple load points, 100 trials |
| Participation cascade | `run_participation_cascade.py` | Fixed load=0.8, iterative rational participation |
| Hetero workload scenarios | `run_hetero_scenarios.py` | 4 workload mixes × 100 trials |

---

## Running Experiments Directly

All scripts are run from the **repository root** (not from `simulation/`).

### Load sweep with band plots

```bash
python simulation/run_custom_user_arrival_web.py
```

Outputs to `simulation/outputs/random_hetero_fixed_load/`:
- `overall_avg_tat_band.png` — system-wide average TAT
- `low/mid/high_tier_tat_band.png` — per-tier group TAT
- `protection_ratio_without_fcfs.png` — Tier-9 protection ratio

### Participation cascade

```bash
python simulation/run_participation_cascade.py
```

Outputs to `simulation/outputs/participation_cascade/`:
- `cascade_high_tier.png` — high-tier participation count per iteration (3 methods)
- `cascade_stacked_3panel.png` — Low/Mid/High stacked bar per scenario

```bash
python simulation/run_participation_cascade.py --replot   # regenerate graphs from saved JSON
```

### Heterogeneous workload scenarios

```bash
python simulation/run_hetero_scenarios.py
```

Runs 4 scenarios × 100 trials each:

| Scenario | Training ratio |
|---|---|
| `uniform` | All users: 0.3 |
| `low_heavy` | Low=0.7, Mid=0.3, High=0.1 |
| `high_heavy` | Low=0.1, Mid=0.3, High=0.7 |
| `random` | Resampled uniformly per trial |

Outputs to `simulation/outputs/hetero_scenarios/{scenario}/`:
- `low/mid/high_tier_tat.png`, `protection_ratio.png`

---

## Output Directories

```
simulation/outputs/
├── random_hetero_fixed_load/     # Load sweep results (band plots + CSV)
├── participation_cascade/        # Cascade simulation graphs + JSON
└── hetero_scenarios/
    ├── uniform/
    ├── low_heavy/
    ├── high_heavy/
    └── random/
```

---

## Configuration (`simulation/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `NUM_USERS` | 18 | Total number of users |
| `SIMULATION_TIME` | 864000 | Observation window (seconds, = 10 days) |
| `ARRIVAL_RATE` | auto | Set by load target in each run script |
| `GPU_PERFORMANCE_LEVELS` | 9 tiers | Relative GPU speed (1.0×–10.0×) |
| `GPU_TIER_ASSIGNMENT` | dict | Maps tier → list of user IDs |
| `RANDOM_SEED` | 42 | Global RNG seed for reproducibility |

Task sizes follow a **log-normal distribution** parameterized by mean and std:
- Inference: mean=9 580 s, std=7 000 s
- Training: mean=412 180 s, std=600 000 s

---

## Reproducing Results

1. Set `RANDOM_SEED = 42` in `config.py` (default).
2. Run the desired script.
3. Outputs are deterministic for fixed seeds; the `random` hetero scenario uses its own `rng = np.random.default_rng(42)`.

> **Note:** Output files in `simulation/outputs/` are excluded from version control (see `.gitignore`).
