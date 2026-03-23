# uta_solver

Python implementation of the UTA (UTilites Additives) family of algorithms for multi-criteria decision analysis. UTA learns additive utility functions from preference rankings supplied by a decision maker.

## Requirements

- Python 3.11+
- numpy, pandas, scipy, scikit-learn

## Installation

Use the provided Makefile:

```bash
make install        # install core package
make install-dev    # install in editable mode
make test           # run tests
make frontend       # launch Streamlit frontend
make examples       # run example scripts
```

## Usage

```python
import pandas as pd
import numpy as np
from uta_solver import UTAEstimator
from uta_solver.criteria import CardinalCriterion, OrdinalCriterion

X = pd.DataFrame({
    "price":     [10.0, 7.0, 3.0, 5.0],
    "size":      [10.0, 6.0, 2.0, 8.0],
    "condition": ["excellent", "good", "poor", "fair"],
})

y = np.array([1, 2, 4, 3])  # preference ranking, 1 = best

model = UTAEstimator(
    criteria=[
        CardinalCriterion("price", n_segments=3, shape="cost"),
        CardinalCriterion("size", n_segments=3, shape="gain"),
        OrdinalCriterion("condition", categories=["poor", "fair", "good", "excellent"]),
    ],
    n_segments=3,
    algorithm="UTASTAR",
)

model.fit(X, y)
print(model.predict(X))        # utility scores
print(model.score(X, y))       # Kendall's tau
print(model.marginal_utilities_)

# Reference-subset workflow (train on subset, score all)
mask = np.array([True, True, False, True])
y_ref = np.array([1, 2, 3])
model.fit(X, y_ref, reference_mask=mask)
scores_all = model.predict(X)

```

## Criterion types

| Type | Description | Example |
|------|-------------|---------|
| `cardinal` | Continuous numeric values, piecewise-linear utility | price, distance |
| `ordinal` | Ordered discrete categories | poor < fair < good < excellent |
| `nominal` | Unordered discrete categories | brand, color |

## Algorithms

| Algorithm | Utility monotonicity | Notes |
|-----------|---------------------|-------|
| `UTASTAR` (default) | Monotonic partial value functions | LP-based, interpretable |
| `UTANM` | Not enforced | More flexible, allows non-monotonic functions |

Notes:
- In `UTASTAR`, nominal criteria are supported with a zero-baseline anchor on the first category level for identifiability.
- Ordinal/nominal matching is value+type based. If your inputs mix representations (for example `1` vs `"1"`), normalize values explicitly.


## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `criteria` | optional | List of Criterion objects matched by name to DataFrame columns |
| `n_segments` | 4 | Number of segments per cardinal criterion |
| `algorithm` | `"UTASTAR"` | `"UTASTAR"` or `"UTANM"` |
| `sigma` | 0.001 | Minimum utility gap between consecutive ranks |
| `theta` | 1.0 | Shape-change penalty weight for `UTANM` |
| `handle_unknown` | `"error"` | Unknown category handling for ordinal/nominal features (`"error"`/`"ignore"`) |
| `extrapolation` | `"clip"` | Cardinal value policy outside breakpoint range (`"clip"`/`"raise"`) |


## Frontend

A Streamlit web interface is included in `frontend/app.py`. Start it with:

```bash
make frontend
```

The app walks through project setup, criterion definition, alternative entry, preference ranking, algorithm configuration, and result visualization.


## JSON format

The frontend accepts JSON format.

```json
{
  "project_name": "my_project",
  "description": "Optional project description",
  "criteria": {
    "price": {
      "type": "cardinal",
      "shape": "cost",
      "n_segments": 4
    }
  },
  "algorithm_settings": {
    "algorithm": "UTASTAR",
    "sigma": 0.001,
    "missing_value_treatment": "assumeAverageValue"
  },
  "alternatives": {
    "reference_alternatives": [
      {
        "Name": "A",
        "price": 10,
        "rank": 1
      }
    ],
    "non_reference_alternatives": [
      {
        "Name": "B",
        "price": 5
      }
    ]
  }
}
```

Rules:
- `description` is optional and is preserved on import/export.
- `alternatives.reference_alternatives[].rank` is required and must be a dense integer ranking over references (`1..k`).
- Non-reference alternatives must not contain `rank`.
- `rank` is a reserved metadata key and cannot be used as a criterion name.
- Top-level `rankings` is not supported.
- `algorithm_settings.reference_names` and `algorithm_settings.output_folder` are not supported.

## CSV format

The frontend also accepts alternatives-only CSV input.

```csv
Name,price,size,condition
Alt A,10,10,excellent
Alt B,7,6,good
Alt C,3,2,poor
```

Rules:
- Include one row per alternative.
- Use `Name` (or `name`) for the alternative identifier column.
- All other columns are treated as criteria.
- Numeric columns are auto-detected as `cardinal` (default `shape: gain`, `n_segments: 2`).
- Non-numeric columns are loaded as pending categorical criteria and must be finalized as `ordinal` or `nominal` in the criteria step.
- CSV does not carry rankings, algorithm settings, or saved results.
