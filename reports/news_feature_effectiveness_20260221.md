# News Feature Effectiveness (2026-02-21)

- sweep: `scripts/feature_search_news_recent.py`
- period: `2025-10-01` to `2026-02-20`
- universe: `smoke_custom10`

## Summary

1. Top score came from horizon 20d settings.
2. News ON/OFF performance was nearly identical in this window.
3. Effective features were mainly price/liquidity proxies:
   - `ret_1d, ret_3d, ret_5d, ret_10d, ret_20d, ret_1d_raw`
   - `corr_mkt_20d, vol_20d, range_20d, adv20_dollar, volume_z_20d`
4. News features were selected but had zero gain/split in top model:
   - `news_count_1d, news_sentiment_1d, news_title_len_mean_1d, news_market_count_1d`

## Notes

- GPU training was used (`LightGBM GPU trainer` on Tesla T4).
- Added robustness in training/prediction to handle sparse long-lookback features by:
  - finite-ratio feature filtering
  - train-median imputation
