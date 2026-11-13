"""
Polymarket NVIDIA CEO forecast — make.fun bounty polymarket-nvidia-ceo-forecast-01
Strategy:
  - Estimate per-market per-minute volatility from recent price history
  - Use exponential weighted moving average (EWMA) of squared returns for vol estimate
  - Apply mean reversion: prices are bounded [0,1] and tend to drift toward recent mean
  - Build quantile forecasts using asymmetric spread that respects boundaries
  - Score: pinball loss (lower = better); must beat house persistence baseline
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles
Z = [-1.6448536, -0.6744898, 0.0, 0.6744898, 1.6448536]

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def clamp6(x: float) -> float:
    """Clamp to [0, 1] and round to 6 decimals."""
    return round(min(1.0, max(0.0, x)) * 1_000_000) / 1_000_000


def estimate_vol_and_drift(prices: list[float], lookback: int = 240) -> tuple[float, float]:
    """
    Estimate per-minute volatility and drift using EWMA on recent returns.
    Returns (sigma_per_sqrt_min, drift_per_min).
    """
    if len(prices) < 2:
        return 0.0025, 0.0

    tail = prices[-lookback:] if len(prices) >= lookback else prices
    returns = [tail[i] - tail[i - 1] for i in range(1, len(tail))]

    if not returns:
        return 0.0025, 0.0

    # EWMA variance with decay lambda = 0.94 (RiskMetrics style)
    lam = 0.94
    var_ewma = returns[0] ** 2
    for r in returns[1:]:
        var_ewma = lam * var_ewma + (1 - lam) * r ** 2

    sigma = math.sqrt(var_ewma)  # per-minute std dev

    # Drift: short-term (last 60 min) minus long-term (last 240 min) mean price
    # Signals mean reversion tendency
    short = sum(tail[-60:]) / len(tail[-60:]) if len(tail) >= 60 else tail[-1]
    long_ = sum(tail) / len(tail)
    # Small mean-reversion pull toward long-term mean
    drift = (long_ - short) * 0.002  # very mild pull

    return sigma, drift


def forecast_market(prices: list[float], horizon: int) -> list[list[float]]:
    """
    Produce horizon rows of [q05, q25, q50, q75, q95] forecasts.
    Uses EWMA vol + mild mean reversion + boundary-aware spread.
    """
    if not prices:
        # Fallback: everything at 0.5
        return [[0.5] * 5 for _ in range(horizon)]

    x = prices[-1]  # last known price (at checkpoint)
    sigma, drift = estimate_vol_and_drift(prices)

    # Minimum sigma floor (don't go lower than house baseline's 0.0025)
    sigma = max(sigma, 0.0025)

    rows = []
    mu = x  # running median estimate
    for h in range(1, horizon + 1):
        # Drift accumulates (mean reversion)
        mu_h = x + drift * h
        mu_h = min(1.0, max(0.0, mu_h))

        # Spread grows with sqrt(h) — Brownian motion scaling
        spread_h = sigma * math.sqrt(h)

        row = []
        for z in Z:
            q = mu_h + z * spread_h
            # Reflect at boundaries: prices near 0 or 1 have asymmetric distributions
            # Use a simple logit-space approach for boundary correction
            if 0.001 < x < 0.999:
                # mild boundary correction: compress quantiles near bounds
                scale = min(x, 1.0 - x) / 0.5  # 1.0 at x=0.5, near 0 at boundaries
                scale = max(0.1, scale)
                q = mu_h + z * spread_h * scale
            row.append(clamp6(q))

        # Ensure non-decreasing
        for i in range(1, len(row)):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]

        rows.append(row)

    return rows


def process_case(in_path: str) -> dict:
    with open(in_path, "r") as f:
        case = json.load(f)

    horizon = case["horizon_minutes"]
    forecasts = []
    for m in case["markets"]:
        prices = m["prices"]
        q_rows = forecast_market(prices, horizon)
        forecasts.append({"id": m["id"], "q": q_rows})

    return {"forecasts": forecasts}


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(f for f in in_dir.iterdir() if f.suffix == ".in")
    for in_file in input_files:
        case_id = in_file.stem
        result = process_case(str(in_file))
        out_path = out_dir / f"{case_id}.out"
        with open(out_path, "w") as f:
            f.write(json.dumps(result) + "\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
