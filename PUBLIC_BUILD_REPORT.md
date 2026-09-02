# Public Build Report

## Build boundary

This package was assembled in a clean directory. The private source repository was
used read-only. Its Git metadata and audit report were not copied. No Git
repository was initialized, and no commit, push, repository creation, or external
publication was performed.

The release is intentionally code-only: it contains schemas and wholly synthetic
tests, but no empirical or third-party data and no acquisition layer.

## 1. Files copied

No source file was copied verbatim. The allowlisted final specification and
rolling-window utility were used as numerical references, then rewritten into the
public package namespace to remove private paths, output locations, internal
stage labels, and import-time assumptions. This conservative choice also avoids
transferring incidental private-repository context.

## 2. Files rewritten

| Public file | Private implementation used as reference | Public rewrite |
| --- | --- | --- |
| `src/attention_sentiment_thesis/spec.py` | final shared specification | Keeps final variable counts, predictor sets, windows, thresholds, and sample dates; removes local paths and expected empirical row counts |
| `oos/window_utils.py` | final rolling schedule helper | Pure daily schedule with the forecast origin structurally excluded |
| `sentiment/clean_headlines.py` | final preprocessing | Accepts only expression identifiers and timing fields; text and source-specific logic are absent |
| `sentiment/build_dictionary.py` | dictionary construction | Requires an external fixed 88-identifier vocabulary; builds presence indicators only |
| `sentiment/compute_beta.py` | rolling beta logic | Provider-neutral columns and preceding-observation window |
| `sentiment/estimate_polarity.py` | recursive polarity estimation | Preserves annual y-1 history, 0.0015 selection, joint no-intercept estimation |
| `sentiment/compute_sentiment.py` | daily scoring | Preserves presence-only polarity sum and no additional scaling |
| `attention/*.py`, `preprocessing/*.py` | final panel pipeline | Pure functions for returns, WPV, ATV, lagging, release-aware controls, and IS-fitted limits |
| `models/*.py` | final regression and sensitivity scripts | Final fixed effects and corrected two-way clustered covariance without private I/O |
| `oos/*.py` | final OOS scripts | In-memory rolling estimation, prediction, benchmark, strict sample, and evaluation |
| `reporting.py` | final output scripts | Minimal caller-directed table and figure output without embedded results |

## 3. Files newly created

New project metadata and documentation are `README.md`, `.gitignore`,
`CITATION.cff`, `pyproject.toml`, `config.example`,
`docs/data_schema.md`, `docs/reproducibility.md`, and `data/README.md`.
The package initializers, schema validator, reporting map, synthetic fixture, and
all test modules are also new. This report is new. A standard MIT `LICENSE`
was added after the author selected the licence.

## 4. Excluded material and reason

| Excluded source scope | Reason |
| --- | --- |
| `src/01_data_collection/**` | The first public release intentionally has no acquisition layer, including otherwise publishable non-news acquisition code |
| All article acquisition, coverage, scraping, site-access, and diagnosis implementations | Confidentiality, copyright, and release-scope prohibition |
| All raw article material, text, markup, response caches, acquisition logs, sessions, and profiles | Confidentiality, copyright, credentials, or non-code artifact |
| Real-corpus vocabulary, estimated lexicon, and firm-day sentiment | Could disclose or reconstruct protected source material; user input in public design |
| `data/**` from the private repository, final panels, and generated empirical results | Code-only release; redistribution rights not established |
| Old specifications, duplicate versions, diagnostics, exploratory analyses, sector models, penalized models, unused attention proxies, and unused specification tests | Not used in the final thesis specification |
| Notebooks, logs, caches, checkpoints, environments, bundles, conversations, local configuration, and secrets | Non-source, local, private, or security-sensitive |
| Private Git metadata and the private audit artifact | Clean-history requirement and explicit exclusion |

## 5. Final-thesis specification mapping

| Final specification | Public implementation |
| --- | --- |
| Fixed 88-expression candidate vocabulary | External input validated by `validate_candidate_vocabulary` |
| Annual recursive selection/polarity through y-1 | `estimate_annual_polarities` |
| Absolute premium threshold 0.0015; joint no-intercept polarity | `estimate_annual_polarities` |
| Unscaled sentiment; zero means no measured non-zero net signal | `aggregate_daily_sentiment` and documentation |
| WPV prior-eight median and ticker one-observation lag | `compute_raw_wpv`, `lag_wpv_by_trading_observation` |
| ATV prior-60 construction, minimum 30 | `compute_atv` |
| Monthly log differences/levels and 40-trading-date availability | `construct_monthly_controls`, `align_monthly_controls` |
| No interpolation; forward fill only after availability | backward as-of event alignment in `align_monthly_controls` |
| Five-regressor headline model, firm/date FE, two-way clusters | `fit_headline_regression` using debiased `PanelOLS` covariance |
| Firm-period/date FE stability interactions and Wald tests | `fit_stability_regression`, `wald_test` |
| 18-control four-model rolling OOS design | `CONTROL_COLUMNS`, `MODEL_PREDICTORS`, `rolling_forecasts` |
| 504-date baseline; 378/630 sensitivities | `FinalSpecification`, rolling `window_dates` argument |
| Rolling firm historical-mean benchmark | `rolling_historical_mean` |
| Strict common sample, OOS R2, Clark--West | `strict_common_sample`, `evaluate_forecasts`, Clark--West helpers |
| Six Clark--West pairs and Holm adjustment | `CLARK_WEST_COMPARISONS`, `clark_west_comparisons` |
| Daily cross-sectional mean adjustment | `daily_mean_adjust` |
| Simplified-model check | `SIMPLIFIED_MODEL_PREDICTORS` with the common rolling engine |
| Common-control winsorization sensitivity | `prepare_common_control_sensitivity`, `fit_winsor_sensitivity_regression` |
| Final table/figure generation | `reporting.py` semantic output map and generic writers |

## 6. Synthetic test results

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

Result: **17 tests passed, 0 failed** on 2026-09-02. The tests cover
imports, schema failures, fixed vocabulary size, return construction, sentiment
aggregation, WPV lagging across a non-trading gap, prior-only WPV and ATV
benchmarks, 40-trading-date monthly availability, absence of future monthly
values, IS-only preprocessing limits, exclusion of the forecast origin, the
prior-only rolling benchmark, daily mean removal, exact predictor sets,
fixed-effects/two-way-cluster estimators, stability interactions, and Wald tests.
All test observations and expression identifiers are synthetic.

## 7. Confidentiality and content re-audit

The staged public tree was scanned case-insensitively for news-provider-specific
names, browser automation, acquisition URLs, common secret labels, credentials,
cookies, local absolute user paths, the private audit filename, and bundle-tool
markers. No news-provider-specific names or other prohibited matches were found.
The sole named source is the descriptive WPV label in `attention/wpv.py`;
it identifies the thesis attention measure and is neither news acquisition nor
scraping logic. A filesystem scan found no `.git`, data archives,
serialized empirical frames, markup caches, logs, checkpoints, or bytecode
caches. There is no `.git` entry inside the public directory and no history was
copied. The sole fixture is a small synthetic JSON file with abstract firm and
expression identifiers.

The README states the actual boundary: no empirical data, acquisition layer,
article text, real-corpus vocabulary, real lexicon, or full numerical
reproduction without lawfully obtained inputs.

## 8. Unresolved matters

- Empirical equality with thesis tables and figures cannot be tested because the
  release deliberately contains no real inputs or expected empirical outputs.
- The fixed 88-expression vocabulary and annual estimated polarities must be
  supplied outside this repository.
- The caller must implement the thesis sample membership/exclusion decisions in
  its lawfully obtained input panel; real security identifiers are not bundled.
- Exact ordinal table/figure numbers should be checked against the deposited
  thesis version; the code maps outputs by stable semantic name.
- Third-party input licences and redistribution rights remain outside this build.

## 9. Human confirmation required

The author should confirm the citation metadata and release date; verify that
the public control-name mapping matches the final
data dictionary; confirm the IS/OOS date boundaries; review the semantic
table/figure mapping; and perform an empirical dry run in a private environment.
The author should also confirm that any future example remains wholly synthetic.

## 10. Remaining work before publication

1. Review every file and the final content scan manually.
2. Run the package against schema-conformant lawful inputs in a private
   environment and compare all final thesis outputs.
3. Record the supported Python/dependency lock versions if archival
   bit-for-bit environment reproduction is required.
4. Confirm citation and thesis-deposit metadata.
5. Only after those checks, initialize a new independent Git history inside the
   public directory and publish through the author's normal review process.

No publication action is part of this build.
