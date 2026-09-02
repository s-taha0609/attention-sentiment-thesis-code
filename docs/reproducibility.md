# Reproducibility and thesis mapping

This is a code-only reproduction layer. Create input files outside version
control, copy `config.example` to a local ignored configuration, and invoke
the pure functions from a driver script. Functions do not create directories,
read files, or access a network on import.

| Thesis method/output | Public implementation | Input → output |
| --- | --- | --- |
| Return construction | `preprocessing.returns` | prices → r_cc, r_oc, target |
| Recursive sentiment | `sentiment.*` | expression events, vocabulary, returns → annual polarity and firm-day score |
| WPV / ATV | `attention.*`, `preprocessing.panel` | views/volume → raw features and lagged WPV |
| Monthly timing | `preprocessing.monthly` | monthly values, trading calendar → release-aware controls |
| Headline regression | `models.fixed_effects` | final panel → five-slope firm/date FE estimates |
| Stability | `models.fixed_effects` | period-labelled panel → interactions and Wald tests |
| Rolling OOS | `oos.window_utils`, `oos.rolling` | panel → forecasts and historical means |
| Evaluation | `oos.evaluation` | strict common forecasts → MSPE, OOS R2, Clark--West |
| Mean adjustment | `oos.evaluation.daily_mean_adjust` | common sample → within-date component |
| Winsor sensitivity | `models.winsor_sensitivity` | capped/uncapped controls → common-control alternative |
| Tables/figures | `reporting` | result frames → caller-selected files |

The baseline OOS window is 504 ordered trading dates. Sensitivities use 378 and
630. All rolling training windows end at the date immediately before the
forecast origin. The benchmark is the firm's historical mean over that same
window. The four main predictor sets use 18 controls and are defined in
`spec.MODEL_PREDICTORS`.

Sentiment is summed without additional scaling. Zero means no measured non-zero
net signal; it does not assert that no event occurred. The selection threshold
is an absolute mean market premium above 0.0015. For scoring year y, selection
and no-intercept joint polarity estimation use observations dated through y-1.

To map final outputs, use `reporting.FINAL_OUTPUT_MAP`. Exact table and figure
numbers may change during thesis typesetting, so the mapping uses semantic names
rather than unstable ordinal labels.

