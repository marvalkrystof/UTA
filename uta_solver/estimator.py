"""UTA estimators: UTA-Star (LP) and UTA-NM (MILP)."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.stats import kendalltau
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .breakpoints import CardinalBreakpoint
from .criteria import CardinalCriterion, Criterion, NominalCriterion, OrdinalCriterion


ArrayLike = Union[np.ndarray, List[float], List[int]]


@dataclass
class SolverResult:
    success: bool
    status: str
    objective_value: float
    primal_values: np.ndarray
    mip_gap: Optional[float]
    message: str
    raw: Any


def _solve_with_linprog(
    c: np.ndarray,
    A_ub: Optional[np.ndarray],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[np.ndarray],
    b_eq: Optional[np.ndarray],
    bounds: Sequence[Tuple[Optional[float], Optional[float]]],
) -> SolverResult:
    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    status = "optimal" if res.success else f"failed({res.status})"
    msg = res.message if isinstance(res.message, str) else str(res.message)
    return SolverResult(
        success=bool(res.success),
        status=status,
        objective_value=float(res.fun) if res.success else float("nan"),
        primal_values=np.asarray(res.x) if res.success else np.array([]),
        mip_gap=None,
        message=msg,
        raw=res,
    )


def _solve_with_scipy_milp(
    c: np.ndarray,
    integrality: np.ndarray,
    bounds: Bounds,
    constraints: LinearConstraint,
    options: Optional[Dict[str, Any]] = None,
) -> SolverResult:
    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=options or {},
    )
    status = "optimal" if res.success else f"failed({res.status})"
    msg = res.message if isinstance(res.message, str) else str(res.message)

    mip_gap = None
    lower_bound = getattr(res, "mip_dual_bound", None)
    objective = getattr(res, "fun", None)
    if lower_bound is not None and objective is not None and np.isfinite(objective):
        denom = max(abs(float(objective)), 1e-12)
        mip_gap = float(abs(float(objective) - float(lower_bound)) / denom)

    return SolverResult(
        success=bool(res.success),
        status=status,
        objective_value=float(res.fun) if res.success else float("nan"),
        primal_values=np.asarray(res.x) if res.success else np.array([]),
        mip_gap=mip_gap,
        message=msg,
        raw=res,
    )


class _UTABaseEstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        *,
        criteria: Optional[List[Criterion]] = None,
        criterion_types: Optional[Union[List[str], Dict[str, str]]] = None,
        n_segments: int = 4,
        breakpoints: Union[str, Dict[str, Sequence[float]]] = "quantile",
        delta: float = 1e-4,
        extrapolation: str = "clip",
        handle_unknown: str = "error",
    ):
        self.criteria = criteria
        self.criterion_types = criterion_types
        self.n_segments = n_segments
        self.breakpoints = breakpoints
        self.delta = delta
        self.extrapolation = extrapolation
        self.handle_unknown = handle_unknown

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: ArrayLike,
        reference_mask: Optional[Union[np.ndarray, Sequence[bool], pd.Series]] = None,
    ):
        X_all = self._to_dataframe(X)
        y_arr = np.asarray(y)

        self.n_features_in_ = X_all.shape[1]
        self.feature_names_in_ = np.array(X_all.columns)

        X_ref, y_ref, reference_indices = self._extract_reference_subset(X_all, y_arr, reference_mask)

        self.reference_indices_ = reference_indices
        self.reference_X_ = X_ref.copy()
        self.reference_y_ = y_ref.copy()

        self.criteria_ = self._create_criteria(X_ref)
        self.criterion_types_ = {criterion.name: criterion.type for criterion in self.criteria_}

        self.reference_order_ = self._build_reference_order(y_ref)
        self._prepare_design_matrices(X_ref)

        self._fit_solver(X_ref, y_ref)

        self.is_fitted_ = True
        self.utilities_ = self.predict(X_ref)
        self.pairwise_ranking_accuracy_ = self._pairwise_ranking_accuracy(self.utilities_, y_ref)
        self.kendall_tau_ = self._kendall_tau(self.utilities_, y_ref)
        self.ranking_fit_ = self.kendall_tau_
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, ["criteria_", "_u_values_", "is_fitted_"])
        X_df = self._to_dataframe(X)
        coeff = self._build_alternative_coeff_matrix(X_df)
        return coeff @ self._u_values_

    def rank(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        utilities = self.predict(X)
        order = np.argsort(-utilities, kind="mergesort")
        ranks = np.empty_like(order, dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        return ranks

    def predict_rank(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.rank(X)

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: ArrayLike) -> float:
        y_arr = np.asarray(y)
        utilities = self.predict(X)
        return self._kendall_tau(utilities, y_arr)

    def get_partial_value_functions(self) -> Dict[str, dict]:
        check_is_fitted(self, ["partial_values_", "marginal_increments_"])
        result = {}
        for criterion in self.criteria_:
            vals = self.partial_values_[criterion.name]
            increments = self.marginal_increments_[criterion.name]
            if increments.size == 0:
                is_monotonic = True
            else:
                is_monotonic = bool(np.all(increments >= -1e-9) or np.all(increments <= 1e-9))
            shape_flags = self.shape_change_flags_.get(criterion.name)
            n_shape_changes = int(np.sum(shape_flags)) if shape_flags is not None else 0
            result[criterion.name] = {
                "breakpoints": self.breakpoints_[criterion.name],
                "utilities": vals,
                "marginal_increments": increments,
                "is_monotonic": is_monotonic,
                "n_shape_changes": n_shape_changes,
            }
        return result

    def get_utility_decomposition(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        check_is_fitted(self, ["criteria_", "_u_values_"])
        X_df = self._to_dataframe(X)
        rows = []
        for idx in range(len(X_df)):
            row = {}
            total = 0.0
            for criterion in self.criteria_:
                contrib = 0.0
                for bp_idx, coef in self._criterion_coefficients(criterion, X_df.iloc[idx][criterion.name]):
                    contrib += coef * self.partial_values_[criterion.name][bp_idx]
                row[criterion.name] = contrib
                total += contrib
            row["global_utility"] = total
            rows.append(row)
        return pd.DataFrame(rows)

    def _fit_solver(self, X: pd.DataFrame, y: np.ndarray):
        raise NotImplementedError

    def _extract_reference_subset(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        reference_mask: Optional[Union[np.ndarray, Sequence[bool], pd.Series]],
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        n_samples = len(X)
        if reference_mask is None:
            reference_indices = np.arange(n_samples)
            X_ref = X
        else:
            mask_arr = np.asarray(reference_mask)
            if mask_arr.dtype != bool:
                raise ValueError("reference_mask must be boolean.")
            if mask_arr.shape[0] != n_samples:
                raise ValueError("reference_mask length must match number of samples in X.")
            reference_indices = np.where(mask_arr)[0]
            X_ref = X.iloc[reference_indices]

        if len(reference_indices) < 2:
            raise ValueError("At least 2 reference alternatives are required.")
        if y.shape[0] != len(reference_indices):
            raise ValueError("y length must match number of reference alternatives.")

        return X_ref.reset_index(drop=True), np.asarray(y), reference_indices

    def _to_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, np.ndarray):
            if hasattr(self, "feature_names_in_"):
                cols = list(self.feature_names_in_)
            else:
                cols = [f"f{i}" for i in range(X.shape[1])]
            return pd.DataFrame(X, columns=cols)
        raise ValueError("X must be pandas.DataFrame or numpy.ndarray.")

    def _resolve_criterion_type(self, X: pd.DataFrame, col_name: str, col_idx: int) -> str:
        if isinstance(self.criterion_types, dict):
            if col_name not in self.criterion_types:
                raise ValueError(f"criterion_types is missing key '{col_name}'.")
            ctype = str(self.criterion_types[col_name]).lower()
        elif isinstance(self.criterion_types, list):
            if col_idx >= len(self.criterion_types):
                raise ValueError("criterion_types list must have one entry per feature.")
            ctype = str(self.criterion_types[col_idx]).lower()
        else:
            if pd.api.types.is_numeric_dtype(X[col_name]):
                return "cardinal"
            raise ValueError(
                f"No criterion provided for '{col_name}' and it is not numeric. "
                "Add an OrdinalCriterion or NominalCriterion to the criteria list."
            )

        if ctype not in {"cardinal", "ordinal", "nominal"}:
            raise ValueError(f"Unsupported criterion type '{ctype}' for '{col_name}'.")
        return ctype

    def _create_criteria(self, X: pd.DataFrame) -> List[Criterion]:
        by_name: Dict[str, Criterion] = {}
        if self.criteria:
            for criterion in self.criteria:
                if not isinstance(criterion, Criterion):
                    raise ValueError(
                        f"All entries in 'criteria' must be Criterion instances, got {type(criterion).__name__}."
                    )
                by_name[criterion.name] = deepcopy(criterion)

        criteria_result: List[Criterion] = []
        for col_idx, col_name in enumerate(X.columns):
            if col_name in by_name:
                criterion = by_name[col_name]
            else:
                ctype = self._resolve_criterion_type(X, col_name, col_idx)
                if ctype == "cardinal":
                    criterion = CardinalCriterion(col_name, n_segments=self.n_segments, shape="gain")
                elif ctype == "ordinal":
                    categories = pd.Series(X[col_name]).dropna().drop_duplicates().tolist()
                    try:
                        categories = sorted(categories)
                    except TypeError as exc:
                        raise ValueError(
                            f"Unable to auto-sort ordinal categories for '{col_name}' because values are not mutually comparable. "
                            "Provide an explicit OrdinalCriterion(categories=[...]) with the desired order."
                        ) from exc
                    criterion = OrdinalCriterion(col_name, categories=categories)
                else:
                    categories = pd.Series(X[col_name]).dropna().drop_duplicates().tolist()
                    criterion = NominalCriterion(col_name, categories=categories)

            criterion.order = col_idx
            n_seg = int(getattr(criterion, "n_segments", self.n_segments))
            self._create_breakpoints_for_criterion(criterion, X[col_name].values, n_seg)
            criteria_result.append(criterion)
        return criteria_result

    def _create_breakpoints_for_criterion(self, criterion: Criterion, values: np.ndarray, n_segments: int) -> None:
        if not isinstance(criterion, CardinalCriterion):
            criterion.create_breakpoints(n_segments, values)
            return

        if isinstance(self.breakpoints, dict) and criterion.name in self.breakpoints:
            raw = np.asarray(self.breakpoints[criterion.name], dtype=float)
            if raw.size < 2:
                raise ValueError(f"Custom breakpoints for '{criterion.name}' must contain at least 2 points.")
            points = np.sort(raw)
        elif self.breakpoints == "uniform":
            numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
            finite = numeric[np.isfinite(numeric)]
            if finite.size == 0:
                raise ValueError(f"Cardinal criterion '{criterion.name}' has no finite values to build breakpoints.")
            min_val = float(criterion.min_val) if criterion.min_val is not None else float(np.min(finite))
            max_val = float(criterion.max_val) if criterion.max_val is not None else float(np.max(finite))
            points = np.linspace(min_val, max_val, n_segments + 1)
        elif self.breakpoints == "quantile":
            quantiles = np.linspace(0.0, 1.0, n_segments + 1)
            points = np.quantile(values.astype(float), quantiles)
            points = np.maximum.accumulate(points)
            if criterion.min_val is not None:
                points[0] = float(criterion.min_val)
            if criterion.max_val is not None:
                points[-1] = float(criterion.max_val)
        else:
            raise ValueError("breakpoints must be 'uniform', 'quantile', or dict mapping criterion names.")

        points = self._prune_empty_cardinal_segments(points, values, min_segments=1)

        criterion.breakpoints = [CardinalBreakpoint(criterion, float(pos), idx) for idx, pos in enumerate(points)]

    def _prune_empty_cardinal_segments(
        self,
        points: np.ndarray,
        values: np.ndarray,
        min_segments: int = 2,
    ) -> np.ndarray:
        pts = np.sort(np.asarray(points, dtype=float))
        if pts.size < 2:
            raise ValueError("At least two breakpoints are required for a cardinal criterion.")

        numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        finite_values = numeric[np.isfinite(numeric)]
        if finite_values.size == 0:
            return pts

        min_breakpoints = max(2, int(min_segments) + 1)
        pruned = pts.tolist()

        changed = True
        while changed and len(pruned) > min_breakpoints:
            changed = False
            for idx in range(1, len(pruned) - 1):
                left = float(pruned[idx - 1])
                right = float(pruned[idx + 1])
                has_alternative_between = bool(np.any((finite_values > left) & (finite_values < right)))
                if not has_alternative_between:
                    del pruned[idx]
                    changed = True
                    break

        deduped = [pruned[0]]
        for point in pruned[1:]:
            if point != deduped[-1]:
                deduped.append(point)

        if len(deduped) >= 2:
            return np.asarray(deduped, dtype=float)

        value = float(deduped[0])
        return np.asarray([value, value], dtype=float)

    def _build_reference_order(self, y: np.ndarray) -> List[List[int]]:
        ranks = np.asarray(y)
        unique = np.unique(ranks)
        groups: List[List[int]] = []
        for rank in np.sort(unique):
            groups.append(np.where(ranks == rank)[0].tolist())
        return groups

    def _prepare_design_matrices(self, X: pd.DataFrame) -> None:
        self._u_var_index_ = {}
        self._u_var_names_ = []
        cursor = 0
        for criterion in self.criteria_:
            bp_count = len(criterion.breakpoints)
            self._u_var_index_[criterion.name] = np.arange(cursor, cursor + bp_count, dtype=int)
            for bp in criterion.breakpoints:
                self._u_var_names_.append(bp.get_marginal_utility_var_name())
            cursor += bp_count
        self._n_u_vars_ = cursor
        self._alt_coeff_ = self._build_alternative_coeff_matrix(X)

    def _criterion_coefficients(self, criterion: Criterion, value: Any) -> List[Tuple[int, float]]:
        if isinstance(criterion, CardinalCriterion):
            if pd.isna(value):
                return []
            x_val = float(value)
            min_bp = float(criterion.breakpoints[0].position)
            max_bp = float(criterion.breakpoints[-1].position)
            if x_val < min_bp or x_val > max_bp:
                if self.extrapolation == "clip":
                    x_val = float(np.clip(x_val, min_bp, max_bp))
                elif self.extrapolation == "raise":
                    raise ValueError(
                        f"Value {x_val} for criterion '{criterion.name}' is outside fitted breakpoint range [{min_bp}, {max_bp}]."
                    )
                else:
                    raise ValueError("extrapolation must be 'clip' or 'raise'.")
            return criterion.get_utility_coefficients(x_val)

        coeffs = criterion.get_utility_coefficients(value)
        if coeffs:
            return coeffs

        if self.handle_unknown == "error":
            known = [bp.position for bp in criterion.breakpoints]
            raise ValueError(
                f"Unknown category/value {value!r} (type={type(value).__name__}) for criterion '{criterion.name}'. "
                f"Known levels: {known!r}. "
                "Set handle_unknown='ignore' to skip unknown levels."
            )
        if self.handle_unknown == "ignore":
            return []
        raise ValueError("handle_unknown must be 'error' or 'ignore'.")

    def _build_alternative_coeff_matrix(self, X: pd.DataFrame) -> np.ndarray:
        coeff = np.zeros((len(X), self._n_u_vars_))
        for row_idx in range(len(X)):
            row = X.iloc[row_idx]
            for criterion in self.criteria_:
                idxs = self._u_var_index_[criterion.name]
                for bp_idx, weight in self._criterion_coefficients(criterion, row[criterion.name]):
                    coeff[row_idx, idxs[bp_idx]] += weight
        return coeff

    @staticmethod
    def _pairwise_ranking_accuracy(utilities: np.ndarray, y: np.ndarray) -> float:
        n = len(y)
        total = 0
        ok = 0
        for i in range(n):
            for j in range(i + 1, n):
                if y[i] == y[j]:
                    continue
                total += 1
                prefers_i = y[i] < y[j]
                if prefers_i and utilities[i] >= utilities[j] - 1e-9:
                    ok += 1
                elif (not prefers_i) and utilities[j] >= utilities[i] - 1e-9:
                    ok += 1
        if total == 0:
            return 1.0
        return ok / total

    @staticmethod
    def _kendall_tau(utilities: np.ndarray, y: np.ndarray) -> float:
        result = kendalltau(np.asarray(y), -np.asarray(utilities), variant="b")
        tau = result.statistic if hasattr(result, "statistic") else result[0]

        # Kendall's tau is undefined when one side is constant.
        # For estimator scoring, return a neutral 0.0 instead of propagating NaN.
        if not np.isfinite(tau):
            return 0.0
        return float(tau)

    def _postprocess_solution(self, u_values: np.ndarray, sp: np.ndarray, sm: np.ndarray, objective: float) -> None:
        self._u_values_ = u_values.copy()
        self.errors_plus_ = sp.copy()
        self.errors_minus_ = sm.copy()
        self.objective_value_ = float(objective)

        self.breakpoints_ = {}
        self.partial_values_ = {}
        self.marginal_increments_ = {}
        self.breakpoint_utilities_ = {}

        for criterion in self.criteria_:
            idxs = self._u_var_index_[criterion.name]
            vals = u_values[idxs]
            self.breakpoints_[criterion.name] = np.array([bp.position for bp in criterion.breakpoints])
            self.partial_values_[criterion.name] = vals
            self.marginal_increments_[criterion.name] = np.diff(vals)
            for local_idx, bp in enumerate(criterion.breakpoints):
                self.breakpoint_utilities_[bp.get_marginal_utility_var_name()] = float(vals[local_idx])

        compatibility_marginals: Dict[str, float] = {}
        for criterion in self.criteria_:
            increments = self.marginal_increments_[criterion.name]
            for local_idx, increment in enumerate(increments):
                compatibility_marginals[f"w_{criterion.order}_{local_idx}"] = float(increment)
        for i, val in enumerate(sp):
            compatibility_marginals[f"Sp_{i}"] = float(val)
        for i, val in enumerate(sm):
            compatibility_marginals[f"Sm_{i}"] = float(val)
        self.marginal_utilities_ = compatibility_marginals
        self.marginal_values_ = {name: values.copy() for name, values in self.partial_values_.items()}

        if not hasattr(self, "shape_change_flags_"):
            self.shape_change_flags_ = {criterion.name: np.array([], dtype=int) for criterion in self.criteria_}
        if not hasattr(self, "shape_penalty_value_"):
            self.shape_penalty_value_ = 0.0
        if not hasattr(self, "criterion_extrema_"):
            self.criterion_extrema_ = {
                criterion.name: {
                    "min": float(np.min(self.partial_values_[criterion.name])),
                    "max": float(np.max(self.partial_values_[criterion.name])),
                }
                for criterion in self.criteria_
            }


class UTAStarRegressor(_UTABaseEstimator):
    def __init__(
        self,
        *,
        criteria: Optional[List[Criterion]] = None,
        criterion_types: Optional[Union[List[str], Dict[str, str]]] = None,
        n_segments: int = 4,
        breakpoints: Union[str, Dict[str, Sequence[float]]] = "quantile",
        delta: float = 1e-4,
        normalize: bool = True,
        extrapolation: str = "clip",
        handle_unknown: str = "error",
        random_state: Optional[int] = None,
    ):
        super().__init__(
            criteria=criteria,
            criterion_types=criterion_types,
            n_segments=n_segments,
            breakpoints=breakpoints,
            delta=delta,
            extrapolation=extrapolation,
            handle_unknown=handle_unknown,
        )
        self.normalize = normalize
        self.random_state = random_state

    def _fit_solver(self, X: pd.DataFrame, y: np.ndarray):
        n_alts = len(X)
        n_u = self._n_u_vars_
        n_nominal = sum(1 for criterion in self.criteria_ if criterion.type == "nominal")

        sp_start = n_u
        sm_start = n_u + n_alts
        nommax_start = n_u + 2 * n_alts
        n_vars = n_u + 2 * n_alts + n_nominal

        c = np.zeros(n_vars)
        c[sp_start:sp_start + n_alts] = 1.0
        c[sm_start:sm_start + n_alts] = 1.0

        A_ub: List[np.ndarray] = []
        b_ub: List[float] = []
        A_eq: List[np.ndarray] = []
        b_eq: List[float] = []

        bounds: List[Tuple[Optional[float], Optional[float]]] = [(0.0, 1.0)] * n_u
        bounds += [(0.0, None)] * (2 * n_alts)
        bounds += [(0.0, 1.0)] * n_nominal

        nommax_idx_by_criterion: Dict[str, int] = {}
        nom_cursor = 0

        for criterion in self.criteria_:
            idxs = self._u_var_index_[criterion.name]
            if criterion.type == "nominal":
                # Identifiability for nominal criteria in LP form:
                # utilities are non-negative and the first category is used as zero baseline.
                for idx in idxs:
                    row = np.zeros(n_vars)
                    row[idx] = 1.0
                    A_ub.append(-row)
                    b_ub.append(0.0)

                row = np.zeros(n_vars)
                row[idxs[0]] = 1.0
                A_eq.append(row)
                b_eq.append(0.0)

                nommax_idx = nommax_start + nom_cursor
                nom_cursor += 1
                nommax_idx_by_criterion[criterion.name] = nommax_idx
                for idx in idxs:
                    row = np.zeros(n_vars)
                    row[nommax_idx] = 1.0
                    row[idx] = -1.0
                    A_ub.append(-row)
                    b_ub.append(0.0)
                continue

            is_cost = isinstance(criterion, CardinalCriterion) and getattr(criterion, "shape", "gain").lower() == "cost"
            for j in range(1, len(idxs)):
                row = np.zeros(n_vars)
                row[idxs[j]] = 1.0
                row[idxs[j - 1]] = -1.0
                if is_cost:
                    A_ub.append(row)
                    b_ub.append(0.0)
                else:
                    A_ub.append(-row)
                    b_ub.append(0.0)

        has_strict_preferences = len(self.reference_order_) > 1

        if self.normalize and has_strict_preferences:
            sum_best = np.zeros(n_vars)
            for criterion in self.criteria_:
                idxs = self._u_var_index_[criterion.name]
                if criterion.type == "nominal":
                    sum_best[nommax_idx_by_criterion[criterion.name]] = 1.0
                    continue

                is_cost = isinstance(criterion, CardinalCriterion) and getattr(criterion, "shape", "gain").lower() == "cost"
                worst_idx = idxs[-1] if is_cost else idxs[0]
                best_idx = idxs[0] if is_cost else idxs[-1]

                row = np.zeros(n_vars)
                row[worst_idx] = 1.0
                A_eq.append(row)
                b_eq.append(0.0)

                sum_best[best_idx] = 1.0

            A_eq.append(sum_best)
            b_eq.append(1.0)
        elif not has_strict_preferences:
            for criterion in self.criteria_:
                for idx in self._u_var_index_[criterion.name]:
                    row = np.zeros(n_vars)
                    row[idx] = 1.0
                    A_eq.append(row)
                    b_eq.append(0.0)

        for group in self.reference_order_:
            if len(group) > 1:
                anchor = group[0]
                for alt in group[1:]:
                    row = np.zeros(n_vars)
                    row[:n_u] = self._alt_coeff_[alt] - self._alt_coeff_[anchor]
                    row[sp_start + alt] = 1.0
                    row[sm_start + alt] = -1.0
                    row[sp_start + anchor] = -1.0
                    row[sm_start + anchor] = 1.0
                    A_eq.append(row)
                    b_eq.append(0.0)

        self.n_reference_constraints_ = 0
        for g_idx in range(len(self.reference_order_) - 1):
            better_group = self.reference_order_[g_idx]
            worse_group = self.reference_order_[g_idx + 1]
            for a in better_group:
                for b in worse_group:
                    row = np.zeros(n_vars)
                    row[:n_u] = self._alt_coeff_[a] - self._alt_coeff_[b]
                    row[sp_start + a] = 1.0
                    row[sm_start + a] = -1.0
                    row[sp_start + b] = -1.0
                    row[sm_start + b] = 1.0
                    A_ub.append(-row)
                    b_ub.append(-self.delta)
                    self.n_reference_constraints_ += 1

        result = _solve_with_linprog(
            c=c,
            A_ub=np.array(A_ub) if A_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(A_eq) if A_eq else None,
            b_eq=np.array(b_eq) if b_eq else None,
            bounds=bounds,
        )
        if not result.success:
            raise RuntimeError(f"UTA-Star solver failed ({result.status}): {result.message}")

        self.solver_ = "highs"
        self.solver_status_ = result.status
        self.solver_message_ = result.message
        self.mip_gap_ = None

        x = result.primal_values
        u = x[:n_u]
        sp = x[sp_start:sm_start]
        sm = x[sm_start:nommax_start]

        self.shape_change_flags_ = {criterion.name: np.array([], dtype=int) for criterion in self.criteria_}
        self.shape_penalty_value_ = 0.0
        self.criterion_extrema_ = {
            criterion.name: {
                "min": float(np.min(u[self._u_var_index_[criterion.name]])),
                "max": float(np.max(u[self._u_var_index_[criterion.name]])),
            }
            for criterion in self.criteria_
        }
        self._postprocess_solution(u, sp, sm, float(result.objective_value))


class UTANMRegressor(_UTABaseEstimator):
    def __init__(
        self,
        *,
        criteria: Optional[List[Criterion]] = None,
        criterion_types: Optional[Union[List[str], Dict[str, str]]] = None,
        n_segments: int = 4,
        breakpoints: Union[str, Dict[str, Sequence[float]]] = "quantile",
        delta: float = 1e-4,
        theta: float = 1.0,
        local_shape_penalties: Optional[Dict[str, Sequence[float]]] = None,
        big_m: float = 10.0,
        epsilon_sign: float = 1e-6,
        max_nonmonotonicity_degree: int = 2,
        objective_threshold: float = 0.01,
        minimum_improvement: float = 0.0,
        normalize: bool = True,
        extrapolation: str = "clip",
        handle_unknown: str = "error",
        time_limit: Optional[float] = None,
        mip_rel_gap: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            criteria=criteria,
            criterion_types=criterion_types,
            n_segments=n_segments,
            breakpoints=breakpoints,
            delta=delta,
            extrapolation=extrapolation,
            handle_unknown=handle_unknown,
        )
        self.theta = theta
        self.local_shape_penalties = local_shape_penalties
        self.big_m = big_m
        self.epsilon_sign = epsilon_sign
        self.max_nonmonotonicity_degree = max_nonmonotonicity_degree
        self.objective_threshold = objective_threshold
        self.minimum_improvement = minimum_improvement
        self.normalize = normalize
        self.time_limit = time_limit
        self.mip_rel_gap = mip_rel_gap
        self.random_state = random_state

    def _fit_solver(self, X: pd.DataFrame, y: np.ndarray):
        max_degree = max(0, int(self.max_nonmonotonicity_degree))
        min_improvement = max(0.0, float(self.minimum_improvement))
        objective_threshold = max(0.0, float(self.objective_threshold))

        best_solution: Optional[Dict[str, Any]] = None
        best_degree = 0
        trace: List[Dict[str, float]] = []

        for degree in range(max_degree + 1):
            solution = self._solve_with_shape_cap(X=X, y=y, shape_cap=degree)
            objective = float(solution["objective_value"])
            trace.append({"degree": float(degree), "objective": objective})

            if best_solution is None:
                best_solution = solution
                best_degree = degree
            else:
                previous_best_obj = float(best_solution["objective_value"])
                improvement = previous_best_obj - objective
                if min_improvement <= 1e-12:
                    if improvement > 1e-12:
                        best_solution = solution
                        best_degree = degree
                    elif improvement < -1e-12:
                        break
                elif improvement >= min_improvement - 1e-12:
                    best_solution = solution
                    best_degree = degree
                else:
                    break

            if objective <= objective_threshold + 1e-12:
                break

            if float(best_solution["objective_value"]) - float(degree) * min_improvement < 0.0:
                break

        if best_solution is None:
            raise RuntimeError("UTA-NM failed to produce a feasible solution.")

        self.solver_ = str(best_solution["solver"])
        self.solver_status_ = str(best_solution["solver_status"])
        self.solver_message_ = str(best_solution["solver_message"])
        self.mip_gap_ = best_solution["mip_gap"]
        self.n_reference_constraints_ = int(best_solution["n_reference_constraints"])

        self.shape_change_flags_ = best_solution["shape_change_flags"]
        self.shape_penalty_value_ = float(best_solution["shape_penalty_value"])
        self.criterion_extrema_ = best_solution["criterion_extrema"]

        self.selected_nonmonotonicity_degree_ = int(best_degree)
        self.utanm_iteration_trace_ = trace

        self._postprocess_solution(
            best_solution["u"],
            best_solution["sp"],
            best_solution["sm"],
            float(best_solution["objective_value"]),
        )

    def _solve_with_shape_cap(self, X: pd.DataFrame, y: np.ndarray, shape_cap: Optional[int]) -> Dict[str, Any]:
        n_alts = len(X)
        n_u = self._n_u_vars_

        sequence_criteria = [criterion for criterion in self.criteria_ if criterion.type != "nominal"]

        segment_meta = []
        for criterion in sequence_criteria:
            n_segments = max(len(criterion.breakpoints) - 1, 0)
            segment_meta.append((criterion.name, n_segments))

        n_w = sum(seg for _, seg in segment_meta)
        n_internal = sum(max(seg - 1, 0) for _, seg in segment_meta)

        n_mprog = sum(len(criterion.breakpoints) for criterion in self.criteria_)
        n_msel = sum(max(len(criterion.breakpoints) - 1, 0) for criterion in self.criteria_)
        n_zmin = sum(len(criterion.breakpoints) for criterion in self.criteria_)

        cursor = 0
        idx_u = np.arange(cursor, cursor + n_u, dtype=int)
        cursor += n_u

        idx_w = np.arange(cursor, cursor + n_w, dtype=int)
        cursor += n_w

        idx_sp = np.arange(cursor, cursor + n_alts, dtype=int)
        cursor += n_alts
        idx_sm = np.arange(cursor, cursor + n_alts, dtype=int)
        cursor += n_alts

        idx_p = np.arange(cursor, cursor + n_w, dtype=int)
        cursor += n_w
        idx_n = np.arange(cursor, cursor + n_w, dtype=int)
        cursor += n_w
        idx_z = np.arange(cursor, cursor + n_w, dtype=int)
        cursor += n_w

        idx_spos = np.arange(cursor, cursor + n_w, dtype=int)
        cursor += n_w
        idx_sneg = np.arange(cursor, cursor + n_w, dtype=int)
        cursor += n_w
        idx_tpos = np.arange(cursor, cursor + n_w, dtype=int)
        cursor += n_w
        idx_tneg = np.arange(cursor, cursor + n_w, dtype=int)
        cursor += n_w

        idx_e = np.arange(cursor, cursor + n_internal, dtype=int)
        cursor += n_internal

        idx_mprog = np.arange(cursor, cursor + n_mprog, dtype=int)
        cursor += n_mprog

        idx_msel = np.arange(cursor, cursor + n_msel, dtype=int)
        cursor += n_msel

        idx_zmin = np.arange(cursor, cursor + n_zmin, dtype=int)
        cursor += n_zmin

        n_vars = cursor

        crit_segment_idxs: Dict[str, np.ndarray] = {}
        crit_internal_idxs: Dict[str, np.ndarray] = {}
        crit_mprog_idxs: Dict[str, np.ndarray] = {}
        crit_msel_idxs: Dict[str, np.ndarray] = {}
        crit_zmin_idxs: Dict[str, np.ndarray] = {}

        w_cursor = 0
        e_cursor = 0
        mprog_cursor = 0
        msel_cursor = 0
        zmin_cursor = 0

        for criterion in self.criteria_:
            gamma = max(len(criterion.breakpoints) - 1, 0)
            if criterion.type == "nominal":
                crit_segment_idxs[criterion.name] = np.array([], dtype=int)
                crit_internal_idxs[criterion.name] = np.array([], dtype=int)
            else:
                crit_segment_idxs[criterion.name] = idx_w[w_cursor:w_cursor + gamma]
                w_cursor += gamma

                internal = max(gamma - 1, 0)
                crit_internal_idxs[criterion.name] = idx_e[e_cursor:e_cursor + internal]
                e_cursor += internal

            bp_count = len(criterion.breakpoints)
            crit_mprog_idxs[criterion.name] = idx_mprog[mprog_cursor:mprog_cursor + bp_count]
            mprog_cursor += bp_count

            msel_count = max(bp_count - 1, 0)
            crit_msel_idxs[criterion.name] = idx_msel[msel_cursor:msel_cursor + msel_count]
            msel_cursor += msel_count

            crit_zmin_idxs[criterion.name] = idx_zmin[zmin_cursor:zmin_cursor + bp_count]
            zmin_cursor += bp_count

        c = np.zeros(n_vars)
        c[idx_sp] = 1.0
        c[idx_sm] = 1.0

        if len(idx_e):
            e_weights = np.ones(len(idx_e), dtype=float)
            for criterion in sequence_criteria:
                internal_idxs = crit_internal_idxs[criterion.name]
                if len(internal_idxs) == 0:
                    continue
                if self.local_shape_penalties and criterion.name in self.local_shape_penalties:
                    local = np.asarray(self.local_shape_penalties[criterion.name], dtype=float)
                    if len(local) != len(internal_idxs):
                        raise ValueError(
                            f"local_shape_penalties['{criterion.name}'] has length {len(local)} but expected {len(internal_idxs)}."
                        )
                    first = int(internal_idxs[0] - idx_e[0])
                    e_weights[first:first + len(local)] = local
            c[idx_e] = self.theta * e_weights

        rows: List[np.ndarray] = []
        lbs: List[float] = []
        ubs: List[float] = []

        def add_constraint(pairs: Sequence[Tuple[int, float]], lb: float = -np.inf, ub: float = np.inf) -> None:
            row = np.zeros(n_vars)
            for index, val in pairs:
                row[int(index)] += float(val)
            rows.append(row)
            lbs.append(lb)
            ubs.append(ub)

        for criterion in sequence_criteria:
            u_idxs = self._u_var_index_[criterion.name]
            w_idxs = crit_segment_idxs[criterion.name]
            if len(w_idxs) == 0:
                continue

            to_local = (w_idxs - idx_w[0]).astype(int)
            p_idxs = idx_p[to_local]
            n_idxs = idx_n[to_local]
            z_idxs = idx_z[to_local]
            spos_idxs = idx_spos[to_local]
            sneg_idxs = idx_sneg[to_local]
            tpos_idxs = idx_tpos[to_local]
            tneg_idxs = idx_tneg[to_local]

            for seg in range(len(w_idxs)):
                add_constraint(
                    [(w_idxs[seg], 1.0), (u_idxs[seg + 1], -1.0), (u_idxs[seg], 1.0)],
                    lb=0.0,
                    ub=0.0,
                )
                add_constraint([(p_idxs[seg], 1.0), (n_idxs[seg], 1.0), (z_idxs[seg], 1.0)], lb=1.0, ub=1.0)

                add_constraint([(w_idxs[seg], 1.0), (p_idxs[seg], -self.big_m)], ub=0.0)
                add_constraint([(w_idxs[seg], 1.0), (p_idxs[seg], -self.big_m)], lb=self.epsilon_sign - self.big_m)

                add_constraint([(w_idxs[seg], 1.0), (n_idxs[seg], self.big_m)], lb=0.0)
                add_constraint([(w_idxs[seg], 1.0), (n_idxs[seg], self.big_m)], ub=self.big_m - self.epsilon_sign)

            add_constraint([(spos_idxs[0], 1.0), (p_idxs[0], -1.0)], lb=0.0, ub=0.0)
            add_constraint([(sneg_idxs[0], 1.0), (n_idxs[0], -1.0)], lb=0.0, ub=0.0)

            for seg in range(1, len(w_idxs)):
                add_constraint([(tpos_idxs[seg], 1.0), (z_idxs[seg], -1.0)], ub=0.0)
                add_constraint([(tpos_idxs[seg], 1.0), (spos_idxs[seg - 1], -1.0)], ub=0.0)
                add_constraint([(tpos_idxs[seg], 1.0), (z_idxs[seg], -1.0), (spos_idxs[seg - 1], -1.0)], lb=-1.0)

                add_constraint([(tneg_idxs[seg], 1.0), (z_idxs[seg], -1.0)], ub=0.0)
                add_constraint([(tneg_idxs[seg], 1.0), (sneg_idxs[seg - 1], -1.0)], ub=0.0)
                add_constraint([(tneg_idxs[seg], 1.0), (z_idxs[seg], -1.0), (sneg_idxs[seg - 1], -1.0)], lb=-1.0)

                add_constraint([(spos_idxs[seg], 1.0), (p_idxs[seg], -1.0), (tpos_idxs[seg], -1.0)], lb=0.0, ub=0.0)
                add_constraint([(sneg_idxs[seg], 1.0), (n_idxs[seg], -1.0), (tneg_idxs[seg], -1.0)], lb=0.0, ub=0.0)

            internal_idxs = crit_internal_idxs[criterion.name]
            for j in range(len(internal_idxs)):
                e_idx = internal_idxs[j]
                prev_seg = j
                next_seg = j + 1
                add_constraint([(e_idx, 1.0), (spos_idxs[prev_seg], -1.0), (n_idxs[next_seg], -1.0)], lb=-1.0)
                add_constraint([(e_idx, 1.0), (sneg_idxs[prev_seg], -1.0), (p_idxs[next_seg], -1.0)], lb=-1.0)
                add_constraint([(e_idx, 1.0), (spos_idxs[prev_seg], -1.0), (sneg_idxs[prev_seg], -1.0)], ub=0.0)
                add_constraint([(e_idx, 1.0), (p_idxs[next_seg], -1.0), (n_idxs[next_seg], -1.0)], ub=0.0)

        if shape_cap is not None and len(idx_e):
            add_constraint([(int(e_idx), 1.0) for e_idx in idx_e], lb=-np.inf, ub=float(shape_cap))

        has_strict_preferences = len(self.reference_order_) > 1

        if self.normalize and has_strict_preferences:
            self._add_utanm_normalization_constraints(
                add_constraint=add_constraint,
                idx_u=idx_u,
                crit_mprog_idxs=crit_mprog_idxs,
                crit_msel_idxs=crit_msel_idxs,
                crit_zmin_idxs=crit_zmin_idxs,
            )
        elif not has_strict_preferences:
            for criterion in self.criteria_:
                for u_idx in self._u_var_index_[criterion.name]:
                    add_constraint([(u_idx, 1.0)], lb=0.0, ub=0.0)

        for group in self.reference_order_:
            if len(group) > 1:
                anchor = group[0]
                for alt in group[1:]:
                    pairs = []
                    for var_idx, val in enumerate(self._alt_coeff_[alt] - self._alt_coeff_[anchor]):
                        if abs(val) > 1e-12:
                            pairs.append((var_idx, float(val)))
                    pairs.extend(
                        [
                            (idx_sp[alt], 1.0),
                            (idx_sm[alt], -1.0),
                            (idx_sp[anchor], -1.0),
                            (idx_sm[anchor], 1.0),
                        ]
                    )
                    add_constraint(pairs, lb=0.0, ub=0.0)

        n_reference_constraints = 0
        for g_idx in range(len(self.reference_order_) - 1):
            better_group = self.reference_order_[g_idx]
            worse_group = self.reference_order_[g_idx + 1]
            for a in better_group:
                for b in worse_group:
                    pairs = []
                    for var_idx, val in enumerate(self._alt_coeff_[a] - self._alt_coeff_[b]):
                        if abs(val) > 1e-12:
                            pairs.append((var_idx, float(val)))
                    pairs.extend(
                        [
                            (idx_sp[a], 1.0),
                            (idx_sm[a], -1.0),
                            (idx_sp[b], -1.0),
                            (idx_sm[b], 1.0),
                        ]
                    )
                    add_constraint(pairs, lb=self.delta)
                    n_reference_constraints += 1

        integrality = np.zeros(n_vars, dtype=int)
        binary_blocks = [idx_p, idx_n, idx_z, idx_spos, idx_sneg, idx_tpos, idx_tneg, idx_e, idx_msel, idx_zmin]
        for block in binary_blocks:
            if len(block):
                integrality[block] = 1

        lb = np.full(n_vars, -np.inf)
        ub = np.full(n_vars, np.inf)

        lb[idx_u] = 0.0
        ub[idx_u] = 1.0

        if len(idx_w):
            lb[idx_w] = -1.0
            ub[idx_w] = 1.0

        lb[idx_sp] = 0.0
        lb[idx_sm] = 0.0

        for block in binary_blocks:
            if len(block):
                lb[block] = 0.0
                ub[block] = 1.0

        if len(idx_mprog):
            lb[idx_mprog] = 0.0
            ub[idx_mprog] = 1.0

        A = np.vstack(rows) if rows else np.empty((0, n_vars))
        constraints = LinearConstraint(A, np.array(lbs), np.array(ubs))

        options: Dict[str, Any] = {}
        if self.time_limit is not None:
            options["time_limit"] = float(self.time_limit)
        if self.mip_rel_gap is not None:
            options["mip_rel_gap"] = float(self.mip_rel_gap)

        result = _solve_with_scipy_milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            constraints=constraints,
            options=options,
        )
        if not result.success:
            raise RuntimeError(f"UTA-NM solver failed ({result.status}): {result.message}")

        x = result.primal_values
        u = x[idx_u]
        sp = x[idx_sp]
        sm = x[idx_sm]

        shape_flags = {}
        for criterion in self.criteria_:
            internal = crit_internal_idxs.get(criterion.name, np.array([], dtype=int))
            shape_flags[criterion.name] = np.rint(x[internal]).astype(int) if len(internal) else np.array([], dtype=int)
        self.shape_change_flags_ = shape_flags
        self.shape_penalty_value_ = float(np.dot(c[idx_e], x[idx_e])) if len(idx_e) else 0.0

        criterion_extrema = {}
        for criterion in self.criteria_:
            u_idxs = self._u_var_index_[criterion.name]
            vals = u[u_idxs]
            mprog_idxs = crit_mprog_idxs[criterion.name]
            m_max = float(x[mprog_idxs[-1]]) if len(mprog_idxs) else float(np.max(vals))
            criterion_extrema[criterion.name] = {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "m_max": m_max,
            }
        return {
            "solver": "scipy_milp",
            "solver_status": result.status,
            "solver_message": result.message,
            "mip_gap": result.mip_gap,
            "objective_value": float(result.objective_value),
            "u": u,
            "sp": sp,
            "sm": sm,
            "shape_change_flags": shape_flags,
            "shape_penalty_value": float(np.dot(c[idx_e], x[idx_e])) if len(idx_e) else 0.0,
            "criterion_extrema": criterion_extrema,
            "n_reference_constraints": n_reference_constraints,
        }

    def _add_utanm_normalization_constraints(
        self,
        *,
        add_constraint: Callable[[Sequence[Tuple[int, float]], float, float], None],
        idx_u: np.ndarray,
        crit_mprog_idxs: Dict[str, np.ndarray],
        crit_msel_idxs: Dict[str, np.ndarray],
        crit_zmin_idxs: Dict[str, np.ndarray],
    ) -> None:
        max_terms: List[Tuple[int, float]] = []
        for criterion in self.criteria_:
            u_idxs = self._u_var_index_[criterion.name]
            mprog = crit_mprog_idxs[criterion.name]
            msel = crit_msel_idxs[criterion.name]
            zmins = crit_zmin_idxs[criterion.name]

            add_constraint([(mprog[0], 1.0), (u_idxs[0], -1.0)], 0.0, 0.0)

            for j in range(1, len(u_idxs)):
                m_prev = mprog[j - 1]
                m_curr = mprog[j]
                u_curr = u_idxs[j]
                sel = msel[j - 1]

                add_constraint([(m_curr, 1.0), (m_prev, -1.0)], 0.0, np.inf)
                add_constraint([(m_curr, 1.0), (u_curr, -1.0)], 0.0, np.inf)

                add_constraint([(m_curr, 1.0), (m_prev, -1.0), (sel, -self.big_m)], -np.inf, 0.0)
                add_constraint([(m_curr, 1.0), (u_curr, -1.0), (sel, self.big_m)], -np.inf, self.big_m)

            max_terms.append((mprog[-1], 1.0))

            add_constraint([(int(z), 1.0) for z in zmins], 1.0, np.inf)
            for local_j, u_idx in enumerate(u_idxs):
                z_idx = zmins[local_j]
                add_constraint([(u_idx, 1.0), (z_idx, 1.0)], -np.inf, 1.0)

        add_constraint(max_terms, 1.0, 1.0)


class UTAEstimator(BaseEstimator, RegressorMixin):
    """Wrapper that dispatches to UTA-Star or UTA-NM."""

    def __init__(
        self,
        criteria: Optional[List[Criterion]] = None,
        criterion_types: Optional[Union[List[str], Dict[str, str]]] = None,
        n_segments: int = 4,
        algorithm: str = "UTASTAR",
        sigma: float = 0.001,
        theta: float = 1.0,
        breakpoints: Union[str, Dict[str, Sequence[float]]] = "quantile",
        handle_unknown: str = "error",
        extrapolation: str = "clip",
        big_m: float = 10.0,
        epsilon_sign: float = 1e-6,
        max_nonmonotonicity_degree: int = 2,
        objective_threshold: float = 0.01,
        minimum_improvement: float = 0.0,
        time_limit: Optional[float] = None,
        mip_rel_gap: Optional[float] = None,
    ):
        self.criteria = criteria
        self.criterion_types = criterion_types
        self.n_segments = n_segments
        self.algorithm = algorithm
        self.sigma = sigma
        self.theta = theta
        self.breakpoints = breakpoints
        self.handle_unknown = handle_unknown
        self.extrapolation = extrapolation
        self.big_m = big_m
        self.epsilon_sign = epsilon_sign
        self.max_nonmonotonicity_degree = max_nonmonotonicity_degree
        self.objective_threshold = objective_threshold
        self.minimum_improvement = minimum_improvement
        self.time_limit = time_limit
        self.mip_rel_gap = mip_rel_gap

    def _resolve_algorithm(self) -> str:
        algo = str(self.algorithm).upper()
        if algo not in {"UTASTAR", "UTANM"}:
            raise ValueError(f"Unknown algorithm '{self.algorithm}'. Must be one of: 'UTASTAR', 'UTANM'.")
        return algo

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: ArrayLike,
        reference_mask: Optional[Union[np.ndarray, Sequence[bool], pd.Series]] = None,
    ):
        algo = self._resolve_algorithm()

        if algo == "UTASTAR":
            model = UTAStarRegressor(
                criteria=self.criteria,
                criterion_types=self.criterion_types,
                n_segments=self.n_segments,
                breakpoints=self.breakpoints,
                delta=self.sigma,
                extrapolation=self.extrapolation,
                handle_unknown=self.handle_unknown,
            )
        else:
            model = UTANMRegressor(
                criteria=self.criteria,
                criterion_types=self.criterion_types,
                n_segments=self.n_segments,
                breakpoints=self.breakpoints,
                delta=self.sigma,
                theta=self.theta,
                big_m=self.big_m,
                epsilon_sign=self.epsilon_sign,
                max_nonmonotonicity_degree=self.max_nonmonotonicity_degree,
                objective_threshold=self.objective_threshold,
                minimum_improvement=self.minimum_improvement,
                extrapolation=self.extrapolation,
                handle_unknown=self.handle_unknown,
                time_limit=self.time_limit,
                mip_rel_gap=self.mip_rel_gap,
            )

        model.fit(X, y, reference_mask=reference_mask)
        self.model_ = model

        passthrough_attrs = [
            "_u_values_",
            "criteria_",
            "criterion_types_",
            "reference_indices_",
            "reference_X_",
            "breakpoints_",
            "partial_values_",
            "marginal_values_",
            "marginal_increments_",
            "breakpoint_utilities_",
            "marginal_utilities_",
            "utilities_",
            "errors_plus_",
            "errors_minus_",
            "objective_value_",
            "ranking_fit_",
            "pairwise_ranking_accuracy_",
            "kendall_tau_",
            "shape_change_flags_",
            "shape_penalty_value_",
            "criterion_extrema_",
            "reference_order_",
            "n_features_in_",
            "feature_names_in_",
            "solver_",
            "solver_status_",
            "solver_message_",
            "mip_gap_",
            "n_reference_constraints_",
            "selected_nonmonotonicity_degree_",
            "utanm_iteration_trace_",
            "is_fitted_",
        ]
        for attr in passthrough_attrs:
            if hasattr(model, attr):
                setattr(self, attr, getattr(model, attr))
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        return self.model_.predict(X)

    def rank(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        return self.model_.rank(X)

    def predict_rank(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.rank(X)

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: ArrayLike) -> float:
        check_is_fitted(self, ["model_"])
        return self.model_.score(X, y)

    def get_partial_value_functions(self) -> Dict[str, dict]:
        check_is_fitted(self, ["model_"])
        return self.model_.get_partial_value_functions()

    def get_utility_decomposition(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        check_is_fitted(self, ["model_"])
        return self.model_.get_utility_decomposition(X)
