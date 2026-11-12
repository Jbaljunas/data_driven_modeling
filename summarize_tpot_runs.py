import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank completed TPOT array runs using training-only CV scores."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main():
    args = parse_args()
    record_paths = sorted(args.results_dir.glob("seed-*/run.json"))
    if not record_paths:
        raise FileNotFoundError(
            f"No completed run records found in {args.results_dir}/seed-*/run.json"
        )

    records = [json.loads(path.read_text()) for path in record_paths]
    table = pd.DataFrame(records).sort_values(
        "best_cv_score",
        ascending=False,
        ignore_index=True,
    )
    required = {"search_seed", "split_seed", "best_cv_score", "pipeline_path"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Run records are missing required fields: {sorted(missing)}")
    if table["split_seed"].nunique() != 1:
        raise ValueError("Cannot compare runs that used different train/test splits.")

    table.to_csv(args.results_dir / "array_summary.csv", index=False)
    best = table.iloc[0].to_dict()
    selection = {
        "selection_rule": (
            "Highest training-only 5-fold cross-validation balanced accuracy "
            "among completed runs with the same fixed split."
        ),
        "selected_run": best,
        "warning": (
            "This record was selected without using the held-out test set. "
            "Evaluate it once, after reviewing stability and interpretability."
        ),
    }
    (args.results_dir / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )

    print(table[["search_seed", "candidate_count", "best_cv_score"]].to_string(index=False))
    print(f"\nCandidate selected for one final test evaluation: seed {best['search_seed']}")


if __name__ == "__main__":
    main()
