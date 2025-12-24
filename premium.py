from pydantic import BaseModel, Field
from typing import Literal, Dict, Tuple

CoverageLevel = Literal["basic", "standard", "premium"]

class PremiumResult(BaseModel):
    monthly_eur: float
    breakdown: Dict[str, float]


def estimate_horsepower_from_engine_size(engine_size_l: float) -> Tuple[int, str]:

    if engine_size_l <= 1.0:
        return (95, "estimated from engine size")
    if engine_size_l <= 1.2:
        return (105, "estimated from engine size")
    if engine_size_l <= 1.4:
        return (120, "estimated from engine size")
    if engine_size_l <= 1.6:
        return (135, "estimated from engine size")
    if engine_size_l <= 1.8:
        return (155, "estimated from engine size")
    if engine_size_l <= 2.0:
        return (180, "estimated from engine size")
    return (200, "estimated from engine size")


def calculate_premium(vehicle_age: int, horsepower: int, city: str, coverage_level: CoverageLevel) -> PremiumResult:
    base = 25.0
    age_factor = 1.0 + max(0, vehicle_age - 3) * 0.05  # +5% per year after 3
    power_factor = 1.0 + max(0, horsepower - 80) * 0.01  # +1% per HP after 80


    city_norm = city.strip().lower()
    city_factor_map = {
        "ljubljana": 1.20,
        "maribor": 1.15,
        "celje": 1.10,
        "koper": 1.05,
    }
    city_factor = city_factor_map.get(city_norm, 1.04)

    coverage_factor_map = {
        "basic": 0.80,
        "standard": 0.90,
        "premium": 1
    }
    coverage_factor = coverage_factor_map[coverage_level]

    monthly = base * age_factor * power_factor * city_factor * coverage_factor

    breakdown = {
        "base": base,
        "age_factor": age_factor,
        "power_factor": power_factor,
        "city_factor": city_factor,
        "coverage_factor": coverage_factor,
    }
    return PremiumResult(monthly_eur=round(monthly, 2), breakdown=breakdown)
