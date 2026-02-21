# News Feature Effectiveness (2026-02-21)

- sweep: `scripts/feature_search_news_recent.py`
- period: `2025-10-01` to `2026-02-20`
- universe: `smoke_custom10`

## Summary

1. Top score came from horizon 20d settings.
2. Root cause of "news features unused" was fixed:
   - before: stock news fetch mostly recent-only, so train/val windows had almost all zeros.
   - after: stock news fetch supports date windows + paging and collects historical range.
3. News features are now actively used by the model (non-zero gain/split).
4. In run `20260221_175240Z_9e2bd028_892aaa87` (`include_news=true`), top contributors included:
   - `news_market_share_20d`
   - `news_sentiment_20d_mean`
   - `news_count_20d`
   - `news_count_5d`
   - `news_sentiment_sum_5d`
5. Coverage is now sufficient in train/val/test:
   - train news non-zero ratio: `0.8143`
   - val news non-zero ratio: `1.0000`
   - test news non-zero ratio: `0.9556`

## Notes

- GPU training was used (`LightGBM GPU trainer` on Tesla T4).
- News feature engineering was expanded:
  - richer text sentiment (title + body)
  - daily aggregated polarity/urgency features
  - rolling/abnormal-news features (`5d/20d`, z-score, buzz ratio, market-share trend)
- News fetch behavior is now universe-size aware by default:
  - `<= 50` symbols: historical fetch enabled
  - `> 50` symbols: conservative mode (recent-oriented), overridable via `data.news_fetch`
