"""Example: Extended car selection with mixed criteria and 14 alternatives."""

import pandas as pd
import numpy as np
from uta_solver import UTAEstimator
from uta_solver.criteria import CardinalCriterion, OrdinalCriterion, NominalCriterion


def main():
    # --- 1. Define alternatives (cars) ---
    cars = pd.DataFrame(
        {
            "power": [220.0, 180.0, 140.0, 200.0, 105.0, 165.0, 155.0, 240.0, 130.0, 175.0, 115.0, 210.0, 150.0, 190.0],
            "fuel_cost": [9.5, 6.2, 5.3, 8.7, 4.7, 5.8, 5.1, 10.2, 6.8, 5.6, 4.9, 7.9, 5.0, 6.9],
            "maintenance_cost": [8.0, 5.5, 4.8, 7.2, 4.2, 5.0, 4.7, 8.5, 4.9, 5.3, 4.4, 6.8, 4.6, 6.0],
            "safety": [
                "premium",
                "advanced",
                "standard",
                "premium",
                "basic",
                "advanced",
                "advanced",
                "premium",
                "standard",
                "advanced",
                "standard",
                "premium",
                "advanced",
                "premium",
            ],
            "comfort": [
                "excellent",
                "good",
                "fair",
                "good",
                "poor",
                "good",
                "fair",
                "excellent",
                "good",
                "good",
                "fair",
                "good",
                "good",
                "excellent",
            ],
            "brand": [
                "BMW",
                "Toyota",
                "Skoda",
                "BMW",
                "Dacia",
                "Hyundai",
                "Skoda",
                "Audi",
                "Peugeot",
                "Toyota",
                "Dacia",
                "Mercedes",
                "Kia",
                "Volvo",
            ],
        }
    )
    print("Cars:")
    print(cars)
    print()

    # --- 2. Decision maker's ranking ---
    rankings = np.array([7, 6, 11, 10, 14, 5, 8, 9, 13, 2, 12, 4, 3, 1])

    # --- 3. Fit with explicit criterion types ---
    model = UTAEstimator(
        criteria=[
            CardinalCriterion("power", n_segments=4, shape="gain"),
            CardinalCriterion("fuel_cost", n_segments=4, shape="cost"),
            CardinalCriterion("maintenance_cost", n_segments=3, shape="cost"),
            OrdinalCriterion("safety", categories=["basic", "standard", "advanced", "premium"]),
            OrdinalCriterion("comfort", categories=["poor", "fair", "good", "excellent"]),
            NominalCriterion(
                "brand",
                categories=[
                    "Dacia",
                    "Skoda",
                    "Kia",
                    "Hyundai",
                    "Peugeot",
                    "Toyota",
                    "Volvo",
                    "BMW",
                    "Mercedes",
                    "Audi",
                ],
            ),
        ],
    )
    model.fit(cars, rankings)

    # --- 4. Results ---
    utilities = model.predict(cars)
    cars["ranking"] = rankings
    cars["utility"] = np.round(utilities, 4)

    print("Results:")
    print(cars.sort_values("utility", ascending=False))
    print()

    tau = model.score(cars[["power", "fuel_cost", "maintenance_cost", "safety", "comfort", "brand"]], rankings)
    print(f"Kendall's tau: {tau:.4f}")
    print(f"Learned variables: {len(model.marginal_utilities_)}")


if __name__ == "__main__":
    main()
