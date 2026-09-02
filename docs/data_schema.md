# Data schemas

All dates use ISO 8601. Tables must be unique on `firm_id, trading_date` unless
stated otherwise. Missing sentiment is set to zero; missing non-sentiment
predictors are handled by listwise deletion at each fit and forecast origin.

## Expression events

One row per upstream-prepared event:

| Field | Type | Meaning |
| --- | --- | --- |
| `firm_id` | string | Stable firm identifier |
| `publication_timestamp` | datetime | Publication time used for recursive estimation |
| `effective_trading_date` | date | Trading date to which the event is assigned |
| `expressions` | sequence[string] | De-duplicated candidate expression identifiers present |
| `eligible_for_daily_score` | boolean | Whether the event enters the pre-cutoff daily score |

Article text and provider metadata are neither required nor accepted. The fixed
candidate vocabulary is a separate, user-supplied list of exactly 88 unique
identifiers. Annual polarity input/output has `scoring_year`,
`expression_id`, and `polarity`.

Legacy-to-public conceptual mapping:

| Private concept | Public field |
| --- | --- |
| security code | `firm_id` |
| article time | `publication_timestamp` |
| assigned market date | `effective_trading_date` |
| extracted term collection | `expressions` |
| before-market-cutoff indicator | `eligible_for_daily_score` |

## Prices and returns

Price input: `firm_id`, `trading_date`, `open_price`, `close_price`.
Output adds log `r_cc`, log `r_oc`, and the next ticker observation's
`target_return`.

## Attention

Page-view input adds positive `page_views`; output is `wpv_raw`. The analysis
panel stores its one-ticker-observation lag as `wpv_lag1`. Volume input adds
`volume`; output is `atv`.

## Monthly controls

One row per `reference_month` and value columns. A value becomes available on
the 40th actual trading date after month-end and is then carried forward.
There is no interpolation or backward fill.

## Final analysis panel

Required keys and target are `firm_id`, `trading_date`, and
`target_return`. The focal variables are `sentiment`, `wpv_lag1`, and
`atv`. The 18 controls are defined exactly in
`attention_sentiment_thesis.spec.CONTROL_COLUMNS`.

The public control names map to the final thesis variables as follows:
`market_excess_return`, `smb`, `hml`, `rmw`, `cma`,
`market_volatility`, `jgb_10y`, `jp_3m`, `us_default_spread`,
`usdjpy_return`, `eurjpy_return`, `gbpjpy_return`, `cpi_jp`,
`ip_jp`, `m1_jp`, and `unemployment_jp`, plus the firm-varying
`r_cc` and `r_oc`. CPI, industrial production, and money are log
differences; unemployment and interest rates remain in levels.
