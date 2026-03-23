# UTA Solver: Algorithms and System Description

## 1. Purpose

`uta_solver` is a Python implementation of the UTA family of preference-disaggregation algorithms.
It learns an **additive utility model** from rankings provided by a decision maker.

Given alternatives described by criteria values, the model estimates criterion-level partial value functions and a global utility score:

\[
U(a) = \sum_{j=1}^{m} u_j(x_{aj})
\]

where:

- \(a\): alternative,
- \(j\): criterion index,
- \(x_{aj}\): value of alternative \(a\) on criterion \(j\),
- \(u_j\): learned partial value function for criterion \(j\).

The package supports:

- **UTASTAR**: monotonic model (LP),
- **UTANM**: non-monotonic model (MILP with shape-change control).

---

## 2. Core Concepts

### 2.1 Alternatives and Rankings

- Input `X`: alternatives (rows) with criterion columns.
- Input `y`: ranks (`1 = best`, larger value = worse).
- Ties are allowed (`y[i] == y[k]`).

Internally, ranks are transformed into ordered groups:

- group 1: best rank,
- group 2: next rank,
- ...

This is used to generate:

- equality constraints for tied items within group,
- strict preference constraints between consecutive groups.

### 2.2 Criterion Types

The solver supports three criterion families:

1. **Cardinal**
- numeric values,
- piecewise-linear interpolation between breakpoints.

2. **Ordinal**
- ordered categories,
- one utility variable per category breakpoint.

3. **Nominal**
- unordered categories,
- one utility variable per category breakpoint.

### 2.3 Breakpoints and Variables

Each criterion has breakpoints.
Each breakpoint corresponds to one utility variable named as:

`w_{criterion_order}_{breakpoint_index}`

For cardinal criteria, an alternative contributes via interpolation weights on neighboring breakpoints.
For ordinal/nominal, contribution is one-hot at matching category.

---

## 3. Fitting Pipeline (Shared by UTASTAR and UTANM)

This logic is implemented in `_UTABaseEstimator`.

### 3.1 Input and Reference Set

- Accepts `pandas.DataFrame` or `numpy.ndarray`.
- Optional `reference_mask` lets you fit on a subset of alternatives and still score all alternatives.
- At least 2 reference alternatives are required.

### 3.2 Criterion Construction

Criteria can be provided explicitly or inferred.

Inference rules:

- numeric column -> cardinal (if no explicit criterion/type provided),
- non-numeric requires explicit type info (`ordinal`/`nominal`).

For auto-generated categories:

- categories preserve native runtime values (no forced string casting),
- auto-ordinal categories are sorted; if values are not mutually comparable, fitting fails with an explicit guidance error.

### 3.3 Breakpoint Generation

Cardinal breakpoints can be built by:

- `quantile` (default),
- `uniform`,
- explicit dictionary per criterion.

After creation, empty segments are pruned when no training alternative falls inside them.
This keeps the discretization supported by observed data.

### 3.4 Coefficient Matrix

A dense matrix `A_alt` is built where each row is one alternative, and columns correspond to all breakpoint utility variables.
Utility prediction is then:

\[
\hat{U} = A_{alt} \cdot u
\]

where `u` is the solved utility-variable vector.

---

## 4. UTASTAR Algorithm (LP)

Class: `UTAStarRegressor`

Solver backend: SciPy `linprog(method="highs")`.

### 4.1 Decision Variables

Let:

- `u`: breakpoint utilities,
- `Sp_i`, `Sm_i`: positive/negative error variables for alternative `i`,
- `nommax_c`: auxiliary max variable for nominal criterion `c` (used in normalization).

### 4.2 Objective

Minimize total ranking error:

\[
\min \sum_i (Sp_i + Sm_i)
\]

### 4.3 Structural Constraints

1. **Monotonicity**

- Gain criterion: non-decreasing utility across breakpoints.
- Cost criterion: non-increasing utility across breakpoints.

2. **Nominal constraints**

For each nominal criterion:

- utilities are constrained non-negative,
- first category breakpoint is anchored to zero (baseline anchor),
- `nommax_c` upper-bounds all nominal breakpoint utilities.

This keeps nominal support while preserving LP formulation.

### 4.4 Normalization

When strict rank inequalities exist:

- For each non-nominal criterion, worst utility is anchored to 0.
- Sum of best criterion utilities equals 1.
- Nominal criteria contribute through their `nommax_c`.

If all alternatives are tied (no strict preference groups), utility variables are fixed to 0.

### 4.5 Preference Constraints

1. **Tie constraints** (within same rank group): exact utility equality (including error terms).
2. **Strict group constraints** (between consecutive groups):

\[
U(a) - U(b) + \text{error terms} \ge \delta
\]

where `delta` is `sigma` (`UTAEstimator` API) / `delta` (`regressor` API).

---

## 5. UTANM Algorithm (MILP)

Class: `UTANMRegressor`

Solver backend: SciPy `milp`.

UTANM removes monotonicity enforcement and allows controlled non-monotonic partial value functions.

### 5.1 Main Variable Blocks

UTANM introduces additional MILP variables beyond `u`, `Sp`, `Sm`:

- `w`: segment increments (`u_{k+1} - u_k`) for non-nominal criteria,
- binary sign/state blocks (`p`, `n`, `z`, `spos`, `sneg`, `tpos`, `tneg`),
- `e`: binary shape-change indicators on internal breakpoints,
- normalization auxiliaries:
  - `mprog` (progressive max),
  - `msel` (binary linearization helper),
  - `zmin` (binary indicators for zero-utility availability).

### 5.2 Objective

\[
\min \sum_i (Sp_i + Sm_i) + \theta \sum e
\]

where:

- first term fits rankings,
- second term penalizes non-monotonic shape changes,
- `theta` controls flexibility vs smoothness/interpretability.

### 5.3 Constraint Families

1. **Segment definition**
- `w` equals consecutive breakpoint utility differences.

2. **Sign/state logic**
- Big-M constraints classify each segment behavior with binaries,
- `epsilon_sign` separates strict sign cases.

3. **Shape-change detection**
- internal `e` variables capture sign-pattern transitions.

4. **Optional shape-cap**
- can enforce `sum(e) <= degree` during iterative search.

5. **Preference constraints**
- same tie and strict group logic as UTASTAR.

### 5.4 Normalization

UTANM normalization enforces:

1. Per-criterion progressive maximum utility construction (`mprog`).
2. Global sum of criterion maxima equals 1.
3. Per criterion, at least one breakpoint utility is zero (binary-supported via `zmin`).

This addresses scale and translation identifiability while allowing non-monotone shapes.

### 5.5 Degree Search Policy

`UTANMRegressor.fit` solves sequentially with nonmonotonicity cap `degree = 0..max_nonmonotonicity_degree`.

Selection behavior:

- initial feasible solution becomes baseline,
- with `minimum_improvement == 0`, only strictly better objective replaces current best,
- equal objectives keep earlier solution (lower degree),
- with positive `minimum_improvement`, update requires at least that improvement,
- additional stop conditions include objective threshold and deterioration logic.

Returned metadata includes:

- `selected_nonmonotonicity_degree_`,
- `utanm_iteration_trace_`,
- shape-change flags per criterion,
- solver status/gap diagnostics.

---

## 6. Prediction and Utility Decomposition

### 6.1 Cardinal Extrapolation

At prediction time, cardinal values outside fitted breakpoint range are handled by:

- `clip` (default): value clipped to nearest bound,
- `raise`: error is thrown.

### 6.2 Unknown Categorical Values

For ordinal/nominal criteria:

- `handle_unknown="error"` (default): raises value+type-aware error with known levels,
- `handle_unknown="ignore"`: criterion contribution is skipped (acts as zero additional contribution).

### 6.3 Outputs

After fitting, the model exposes:

- `objective_value_`,
- `partial_values_` per criterion at breakpoints,
- `marginal_increments_` (`diff` of partial values),
- `breakpoint_utilities_` variable map,
- `marginal_utilities_` compatibility map,
- decomposition API via `get_utility_decomposition(X)`.

---

## 7. High-Level API (`UTAEstimator`)

`UTAEstimator` is the user-facing wrapper.

It forwards parameters to the selected backend:

- `algorithm="UTASTAR"` or `"UTANM"`,
- shared controls (`sigma`, breakpoints behavior, unknown handling, extrapolation),
- UTANM controls (`theta`, `big_m`, `epsilon_sign`, degree search settings, MILP options).

It also mirrors learned attributes from the backend model so downstream code can stay algorithm-agnostic.

---

## 8. Validation and Tests

Test suite (`uta_solver/tests`) verifies:

- criterion/breakpoint primitives,
- fit/predict behavior,
- ranking consistency and ties,
- mixed criterion handling,
- type-sensitive category behavior,
- UTASTAR nominal baseline anchoring,
- UTANM normalization properties,
- UTANM degree-selection tie/improvement policy.

Some UTANM policy tests use `monkeypatch` to isolate decision logic from full MILP solving complexity.

---

## 9. Practical Interpretation

- Use **UTASTAR** when monotonic criterion behavior is expected and interpretability is priority.
- Use **UTANM** when preferences can be non-monotonic (e.g., ideal ranges or diminishing/reversing preference patterns).
- Increase `theta` to discourage shape changes.
- Control complexity via `max_nonmonotonicity_degree` and `minimum_improvement`.

The result in both cases is an additive utility model you can use to:

- rank existing alternatives,
- score unseen alternatives,
- inspect criterion-specific learned preference structure.
