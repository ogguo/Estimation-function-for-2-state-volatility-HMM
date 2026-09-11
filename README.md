# Estimation function for a 2-state volatility HMM

Code accompanying the thesis "Stochastic Regimes and Numerical Methods in Retail-Based IPO Market Dynamics", BIEF Bsc program, Bocconi University, 2026.

This repository contains the estimation function used to fit the volatility
regimes reported in the thesis. It is run once per listing. The sample
construction, the basket comparison and the cross-sectional tests are not
included here.

## What `analizza_titolo` does

Given a ticker and an IPO date, it:

- downloads daily prices with `yfinance` and keeps the first 180 trading days
- builds an annualized rolling realized volatility series (standard deviation of
  log returns over a 10-day window, scaled by sqrt(252))
- fits a two-state Gaussian HMM to that series, refitting under 30 different
  seeds and keeping the highest-likelihood fit
- labels the state with the higher mean as the high-volatility one, so the two
  regimes mean the same thing across listings
- computes the transition matrix, expected regime durations, the stationary and
  empirical share of high-volatility days, and the smoothed probability of the
  high-volatility state on each date
- fits a single-Gaussian benchmark and compares the two by AIC and BIC

It returns those quantities as a dictionary and writes a two-panel figure,
`<TICKER>_hmm.png`: the volatility series with high-volatility periods shaded,
and the smoothed probabilities below it.

## Running it

Install the dependencies with `pip install -r requirements.txt`, then import the
function and call it:

```python
from <module> import analizza_titolo

res = analizza_titolo("HOOD", "2021-07-29", seed=0)
print(res["delta_bic"])   # positive means the two-state model is preferred
```

Tested on Python 3.12.

## A note on the restarts

The likelihood surface is multimodal and the fit depends on where the optimizer
starts, so each listing is estimated 30 times and only the highest-likelihood
fit is kept. No reported estimate comes from a single fit.
