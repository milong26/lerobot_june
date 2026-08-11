# SIC-Guided Incremental Data Selection - Step 4 Report

## Physical Grid

- **X range**: determined from 500 episodes `obj_x` (environment_state[:, 0])
- **Y range**: determined from 500 episodes `obj_y` (environment_state[:, 1])
- **Grid size**: 5x5 = 25 cells
- **Cell centers**: saved in `results/step1_grid_info.json`

## B0 Episodes

- **Size**: 25 episodes (1 per cell, closest to cell center in physical space)
- **Episode indices**: saved in `results/step2_b0_episodes.json`

## SIC Convergence

- **Method**: cosine distance (1 - cosine_similarity)
- **N_add per round**: 10 episodes
- **Top K cells**: 3 cells with highest SIC selected per round
- **Max per cell**: 5 episodes before switching to next cell
- **Convergence threshold**: 0.01 * initial_SIC for 2 consecutive rounds
- **SIC curve**: `results/sic_curves.png`
- **Cell SIC heatmap**: `results/cell_sic_heatmap.png`

## Budget Subsets

| Budget | File | Episodes |
|--------|------|----------|
| N=50   | `results/subsets_by_budget.json` (key: "N_50") | 50 |
| N=100  | `results/subsets_by_budget.json` (key: "N_100") | 100 |
| N=150  | `results/subsets_by_budget.json` (key: "N_150") | 150 |
| N=200  | `results/subsets_by_budget.json` (key: "N_200") | 200 |
| N=300  | `results/subsets_by_budget.json` (key: "N_300") | 300 |

## Training Scripts

Located in `scripts/`:
- `run_ours_50_seed42.sh`
- `run_ours_100_seed42.sh`
- `run_ours_150_seed42.sh`
- `run_ours_200_seed42.sh`
- `run_ours_300_seed42.sh`

Training output: `training_output/N{budget}/seed42/`

## Attention Visualization

Code: `personal/work2/attention_fig/plot_attention.py`

Usage:
```bash
python plot_attention.py --model-paths /path/to/model1 /path/to/model2
```

## File Structure

```
personal/work2/ours/method/
├── step0_load_data.py            # Load 500 episodes, extract positions + embeddings
├── step1_define_grid.py          # Define 5x5 grid, generate cell centers
├── step2_b0_construction.py      # Construct B0 (25 episodes)
├── step3_sic_iteration.py        # SIC incremental iteration
├── step4_output_subsets.py       # Output subsets by budget
├── scripts/
│   ├── run_ours_50_seed42.sh
│   ├── run_ours_100_seed42.sh
│   ├── run_ours_150_seed42.sh
│   ├── run_ours_200_seed42.sh
│   └── run_ours_300_seed42.sh
├── results/
│   ├── step0_all_data.pkl
│   ├── step0_episode_meta.csv
│   ├── step1_grid_info.json
│   ├── step1_grid_visualization.png
│   ├── step2_b0_episodes.json
│   ├── selection_log.json
│   ├── sic_curves.png
│   ├── cell_sic_heatmap.png
│   ├── budget_subsets.pkl
│   └── subsets_by_budget.json
└── REPORT.md
```