# Data-Driven Components in Modeling

This repository supports the data-driven-modeling portion of CMSE 802. It
examines what observations can contribute to a computational model, when
data-driven methods are appropriate, and why a model output is not enough to
make it trustworthy.

Data-driven components appear in many research workflows. They can summarize
measurements, identify patterns, estimate relationships, automate model
selection, and process text or images. They also inherit uncertainty, bias,
and limitations from their data and training process. Throughout this unit, we
will treat validation as part of modeling rather than as a final check.

## Unit Goals

By the end of this unit, students should be able to:

- explain what data-driven components contribute to a model;
- summarize observations and communicate uncertainty;
- distinguish descriptive statistics from predictive machine learning;
- evaluate whether a dataset and method fit a scientific question;
- use automated and pretrained-model tools without treating their output as
  unquestionable evidence; and
- document enough of an experiment for another researcher to reproduce and
  critique it.

## Four-Meeting Arc

1. **Statistics as data-driven models:** summarize measurements, quantify
   uncertainty, and distinguish association from explanation.
2. **Classical machine learning:** compare regression, classification, and
   clustering while considering train/test data, overfitting, and dataset size.
3. **AutoML on the HPCC:** install and run a bounded TPOT search, record the
   experiment, and evaluate whether an automated recommendation is credible.
4. **Reproducible AutoML arrays on the HPCC:** repeat a bounded TPOT search
   across a small Slurm array, select a candidate without reusing the held-out
   test set, and document the evidence needed to audit the experiment.

The four core notebooks are ready now. A separate instructor- and TA-led
LangChain/LangGraph studio may follow this unit to introduce AI workflows for
unstructured information without changing the unit's assessed AutoML arc.

## Getting Started

Create the Conda environment in the repository:

```bash
conda env create --prefix ./envs --file environment.yml
conda activate ./envs
jupyter lab
```

If the environment already exists, update it instead:

```bash
conda env update --prefix ./envs --file environment.yml --prune
```

Open `Assignment_Instructions_Part_1.ipynb` and work through the notebook in
order. The notebook creates synthetic data so it can run without a download or
an external account.

## Repository Structure

- `Assignment_Instructions_Part_1.ipynb`: Day 1 statistics and uncertainty
  activity.
- `Assignment_Instructions_Part_2.ipynb`: Day 2 classical machine learning
  activity (scikit-learn, overfitting, curse of dimensionality).
- `Assignment_Instructions_Part_3.ipynb`: Day 3 AutoML activity (TPOT, CASH,
  evolutionary pipeline search, and validation).
- `Assignment_Instructions_Part_4.ipynb`: HPCC activity that compares bounded
  TPOT searches across a Slurm job array and records reproducibility evidence.
- `environment.yml`: the software environment for the current activity.
- `README.md`: unit orientation, workflow, and learning goals.

## Scientific Software Principles

Use the course software-quality standards while working in this repository:

- Safe
- Portable
- Reproducible
- Robust
- Literate

Keep changes small, rerun code after modifications, and record the assumptions
behind an analysis. A result is useful only when someone else can understand
how it was produced and judge whether it applies.