"""Example dataset loaders for frontend and external usage."""

import numpy as np
import pandas as pd


def load_apartments():
    X = pd.DataFrame(
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
        },
        index=[f"A{i}" for i in range(1, 15)],
    )
    y = np.array([2, 8, 9, 10, 12, 13, 3, 6, 11, 1, 14, 7, 5, 4])
    criteria_defs = {
        "price": {"type": "cardinal", "shape": "cost", "n_segments": 4},
        "size": {"type": "cardinal", "shape": "gain", "n_segments": 4},
        "commute": {"type": "cardinal", "shape": "cost", "n_segments": 3},
        "condition": {
            "type": "ordinal",
            "categories": ["poor", "fair", "good", "excellent"],
        },
    }
    return X, y, criteria_defs


def load_cars():
    X = pd.DataFrame(
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
        },
        index=[f"C{i}" for i in range(1, 15)],
    )
    y = np.array([7, 6, 11, 10, 14, 5, 8, 9, 13, 2, 12, 4, 3, 1])
    criteria_defs = {
        "power": {"type": "cardinal", "shape": "gain", "n_segments": 4},
        "fuel_cost": {"type": "cardinal", "shape": "cost", "n_segments": 4},
        "maintenance_cost": {"type": "cardinal", "shape": "cost", "n_segments": 3},
        "safety": {
            "type": "ordinal",
            "categories": ["basic", "standard", "advanced", "premium"],
        },
        "comfort": {
            "type": "ordinal",
            "categories": ["poor", "fair", "good", "excellent"],
        },
        "brand": {
            "type": "nominal",
            "categories": [
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
        },
    }
    return X, y, criteria_defs
