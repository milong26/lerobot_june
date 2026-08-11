#!/usr/bin/env python
"""
Step 4: Output subsets for different budgets (N=50, 100, 150, 200, 300) as JSON.
Format: {"N_50": [ep_idx, ...], "N_100": [...], ...}
"""
import json
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BUDGETS = [50, 100, 150, 200, 300]


def main():
    print("=" * 60)
    print("Step 4: Output subsets for different budgets")
    print("=" * 60)

    budget_path = RESULTS_DIR / "budget_subsets.pkl"
    with open(budget_path, "rb") as f:
        budget_subsets = pickle.load(f)

    output = {}
    for budget in BUDGETS:
        key = f"N_{budget}"
        if budget in budget_subsets:
            episodes = budget_subsets[budget]
            output[key] = episodes
            print(f"{key}: {len(episodes)} episodes")
        else:
            print(f"{key}: NOT REACHED (available: {sorted(budget_subsets.keys())})")

    out_path = RESULTS_DIR / "subsets_by_budget.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved subsets to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()