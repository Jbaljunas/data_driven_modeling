import argparse
import json
from pathlib import Path

import joblib
import sklearn
import tpot
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one bounded TPOT search and save its training-only record."
    )
    parser.add_argument("--search-seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=802)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-time-mins", type=float, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    dataset = load_breast_cancer(as_frame=True)
    X_train, _, y_train, _ = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.20,
        stratify=dataset.target,
        random_state=args.split_seed,
    )

    search = TPOTClassifier(
        search_space="linear",
        scorers=["balanced_accuracy"],
        cv=5,
        max_time_mins=args.max_time_mins,
        max_eval_time_mins=min(1.0, args.max_time_mins / 2),
        n_jobs=1,
        random_state=args.search_seed,
        verbose=5,
    )
    search.fit(X_train, y_train)

    records = search.evaluated_individuals
    best_cv_score = float(records["balanced_accuracy_score"].max())
    pipeline_path = args.output_dir / "pipeline.joblib"
    joblib.dump(search.fitted_pipeline_, pipeline_path)

    record = {
        "search_seed": args.search_seed,
        "split_seed": args.split_seed,
        "max_time_mins": args.max_time_mins,
        "search_space": "linear",
        "metric": "balanced_accuracy",
        "cv_folds": 5,
        "n_jobs": 1,
        "n_training_observations": len(X_train),
        "n_features": X_train.shape[1],
        "candidate_count": len(records),
        "best_cv_score": best_cv_score,
        "pipeline_path": str(pipeline_path),
        "pipeline_repr": repr(search.fitted_pipeline_),
        "scikit_learn_version": sklearn.__version__,
        "tpot_version": tpot.__version__,
    }
    (args.output_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    records.to_csv(args.output_dir / "evaluated_individuals.csv", index=False)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
