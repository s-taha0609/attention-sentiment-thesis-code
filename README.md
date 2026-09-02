# Attention and Sentiment Thesis Code

This is the code-only public reproducibility package for the master's thesis
“Do Investor Attention and News Sentiment Have Distinct Conditional Associations
with Stock Returns?” by Shintaro Tahara.

## Research purpose

The package implements the final empirical specification used to distinguish the
conditional associations of investor attention and sentiment with stock returns.
It covers final preprocessing, return and panel construction, provider-neutral
sentiment algorithms, WPV and ATV, fixed-effects regressions, stability tests,
rolling out-of-sample forecasts, forecast evaluation, sensitivities, and minimal
table/figure helpers.

## Scope and limitations

This repository contains no real data. It contains no data-acquisition or
scraping code, no article headings or bodies, and no provider-specific access
logic. It also omits the real-news-derived 88-expression vocabulary, estimated
lexicons, and firm-day sentiment. Vocabulary identifiers and annual polarities
are user-provided inputs.

Consequently, this code alone cannot reproduce the numerical results in the
thesis. Reproduction requires input data that the user has obtained lawfully and
converted to the documented schemas. Users are responsible for checking each
data provider's licence, access terms, and redistribution restrictions.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The optional figure dependency is installed with
`python -m pip install -e '.[figures]'`.

## Required inputs

The complete schemas are in [docs/data_schema.md](docs/data_schema.md). At
minimum, callers provide:

- a firm/trading-date return panel;
- adjusted volume and page-view observations;
- release-aware market and macroeconomic controls;
- provider-neutral expression events containing firm identifier, publication
  timestamp, effective trading date, expression-presence identifiers, and a
  daily-score eligibility flag;
- an externally supplied fixed vocabulary of exactly 88 expression identifiers.

No text field is accepted by the sentiment API.

## Execution order

1. Validate inputs and construct log returns.
2. Construct raw WPV and ATV; merge the panel; lag WPV by one ticker trading
   observation.
3. Align monthly controls on the 40th observed trading date after month-end,
   then forward-fill.
4. Compute rolling market beta and premiums; build expression presence;
   recursively select expressions and estimate annual polarity using data
   through the preceding year.
5. Aggregate eligible events to unscaled firm-day sentiment.
6. Fit 1/99 limits using only the designated IS sample and apply those fixed
   limits unchanged to IS and OOS; estimate the four rolling within-firm models using the 504-date
   baseline window (and 378/630-date sensitivities).
7. Build the strict common evaluation sample, historical-mean benchmark,
   OOS R-squared, Clark--West comparisons, and daily-mean-adjusted metrics.
8. Run headline and period-interaction fixed-effects regressions and the
   common-control winsorization sensitivity; export final tables/figures.

Inputs and outputs of every stage and their thesis mapping are detailed in
[docs/reproducibility.md](docs/reproducibility.md).

## Synthetic tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

All fixtures are deliberately synthetic and contain expression identifiers,
not article text. The test suite checks schemas, information timing, feature
construction, predictor sets, rolling windows, and mean adjustment.

## Licence

This software is released under the [MIT License](LICENSE).
