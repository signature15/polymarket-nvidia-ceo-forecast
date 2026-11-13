# Polymarket NVIDIA CEO Forecast

Entry for the [make.fun bounty](https://make.fun/bounty/polymarket-nvidia-ceo-forecast-01): **Forecast Polymarket's next NVIDIA CEO prices minute by minute, 4 hours ahead**.

## Approach

- **EWMA volatility**: estimates per-market per-minute volatility using exponentially weighted moving average of squared returns (λ=0.94, RiskMetrics style)
- **Mean reversion**: mild drift toward long-term mean price
- **Boundary-aware spreads**: quantile spreads scale down near 0 and 1 to reflect the bounded nature of prediction market prices
- **Non-decreasing quantiles**: enforced row-wise

## Usage

```bash
# Run against input cases
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py
```

## Requirements

Python 3.13 standard library only.
