"""Example: Extended apartment selection with richer criteria and 14 alternatives."""

import pandas as pd
import numpy as np
from uta_solver import UTAEstimator
from uta_solver.criteria import CardinalCriterion, OrdinalCriterion


def main():
    # --- 1. Define alternatives (apartments) ---
    apartments = pd.DataFrame(
        {
            "price": [210.0, 185.0, 160.0, 145.0, 130.0, 120.0, 175.0, 155.0, 140.0, 200.0, 115.0, 170.0, 150.0, 230.0],
            "size": [95.0, 88.0, 80.0, 72.0, 68.0, 60.0, 90.0, 78.0, 74.0, 85.0, 58.0, 82.0, 70.0, 100.0],
            "commute": [20.0, 24.0, 30.0, 18.0, 35.0, 22.0, 28.0, 16.0, 26.0, 14.0, 40.0, 19.0, 12.0, 33.0],
            "condition": [
                "excellent",
                "good",
                "good",
                "fair",
                "fair",
                "poor",
                "excellent",
                "good",
                "fair",
                "excellent",
                "poor",
                "good",
                "good",
                "excellent",
            ],
        }
    )
    print("Apartments:")
    print(apartments)
    print()

    # --- 2. Provide preference ranking (1 = best) ---
    rankings = np.array([2, 8, 9, 10, 12, 13, 3, 6, 11, 1, 14, 7, 5, 4])

    # --- 3. Fit the UTA model ---
    model = UTAEstimator(
        criteria=[
            CardinalCriterion("price", n_segments=4, shape="cost"),
            CardinalCriterion("size", n_segments=4, shape="gain"),
            CardinalCriterion("commute", n_segments=3, shape="cost"),
            OrdinalCriterion("condition", categories=["poor", "fair", "good", "excellent"]),
        ],
        sigma=0.001,
    )
    model.fit(apartments, rankings)

    # --- 4. Predict utilities ---
    utilities = model.predict(apartments)

    apartments_with_results = apartments.copy()
    apartments_with_results["ranking"] = rankings
    apartments_with_results["utility"] = np.round(utilities, 4)
    apartments_with_results = apartments_with_results.sort_values("utility", ascending=False)

    print("Results (sorted by utility):")
    print(apartments_with_results)
    print()

    # --- 5. Score the model ---
    tau = model.score(apartments[["price", "size", "commute", "condition"]], rankings)
    print(f"Kendall's tau: {tau:.4f}")
    print(f"Objective value (total error): {model.objective_value_:.6f}")

    # --- 6. Predict on a new apartment ---
    new = pd.DataFrame({
        "price": [162.0],
        "size": [79.0],
        "commute": [17.0],
        "condition": ["good"],
    })
    new_utility = model.predict(new)
    print(f"\nNew apartment utility: {new_utility[0]:.4f}")


if __name__ == "__main__":
    main()
