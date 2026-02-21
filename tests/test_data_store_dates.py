from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbdt_agent.data_store import _extract_symbol, _parse_date, compute_dataset_id, resolve_universe, update_data
from gbdt_agent.paths import ProjectPaths


def test_parse_date_rejects_empty_and_invalid_values() -> None:
    assert _parse_date("") is None
    assert _parse_date("   ") is None
    assert _parse_date("not-a-date") is None
    assert _parse_date("2026-02-21") == date(2026, 2, 21)


def test_extract_symbol_rejects_company_names() -> None:
    assert _extract_symbol("AAPL") == "AAPL"
    assert _extract_symbol("BRK.B") == "BRK.B"
    assert _extract_symbol("BLOCK, INC.") is None
    assert _extract_symbol("AIRBNB INC") is None


def test_resolve_universe_uses_today_when_end_date_is_blank(monkeypatch, tmp_path: Path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "universe_custom.yaml").write_text(
        yaml.safe_dump({"symbols": ["AAPL", "MSFT"]}, sort_keys=False)
    )

    cfg = {
        "universe": {
            "provider": "custom_list",
            "fallback_small50": {"enabled": False, "symbols": []},
        },
        "data": {
            "start_date": "2026-02-17",
            "end_date": "",
        },
    }
    monkeypatch.setattr("gbdt_agent.data_store._today_utc_date", lambda: date(2026, 2, 21))
    paths = ProjectPaths.from_project_dir(tmp_path)

    result = resolve_universe(cfg, paths, fmp=object())
    assert result.end_date_effective == "2026-02-21"
    assert result.start_date == "2026-02-17"
    assert result.membership_long.empty is False


def test_update_data_skips_price_fetch_when_no_new_business_day(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "universe_custom.yaml").write_text(yaml.safe_dump({"symbols": ["AAPL"]}, sort_keys=False))

    cfg = {
        "universe": {
            "provider": "custom_list",
            "fallback_small50": {"enabled": False, "symbols": []},
        },
        "data": {
            "start_date": "2026-02-20",
            "end_date": "2026-02-21",  # Saturday
            "adjusted_flag": False,
            "endpoints_version": ["prices_eod"],
        },
    }
    paths = ProjectPaths.from_project_dir(tmp_path)
    paths.ensure_base_dirs()

    dsid = compute_dataset_id(cfg, ["AAPL"], effective_end_date="2026-02-21")
    raw = paths.raw_dir / dsid
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": date(2026, 2, 20),
                "symbol": "AAPL",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 100.0,
            }
        ]
    ).to_parquet(raw / "prices.parquet", index=False)

    class DummyFMP:
        def __init__(self) -> None:
            self.price_calls = 0

        def get_prices(self, *args, **kwargs):
            self.price_calls += 1
            return []

        def get_dividends(self, *args, **kwargs):
            return []

        def get_splits(self, *args, **kwargs):
            return []

        def get_earnings(self, *args, **kwargs):
            return []

        def get_earnings_surprises(self, *args, **kwargs):
            return []

        def get_income_statement(self, *args, **kwargs):
            return []

        def get_balance_sheet(self, *args, **kwargs):
            return []

        def get_cash_flow(self, *args, **kwargs):
            return []

    fmp = DummyFMP()
    _, _, last_data_date = update_data(cfg, paths, fmp, force=False)
    assert fmp.price_calls == 0
    assert last_data_date == "2026-02-20"


def test_compute_dataset_id_changes_when_news_contract_changes() -> None:
    base = {
        "run": {"timezone_assumption": "US/Eastern"},
        "universe": {"name": "custom"},
        "data": {
            "start_date": "2026-01-01",
            "end_date": "2026-02-01",
            "adjusted_flag": False,
            "endpoints_version": ["prices_eod"],
            "include_news": False,
        },
        "fmp": {"endpoints": {"stock_news": "news/stock", "general_news": "news/stock"}},
    }
    with_news = {
        **base,
        "data": {**base["data"], "include_news": True},
    }
    with_news_alt = {
        **with_news,
        "fmp": {"endpoints": {"stock_news": "news/stock-v2", "general_news": "news/stock"}},
    }

    id_base = compute_dataset_id(base, ["AAPL"], effective_end_date="2026-02-01")
    id_news = compute_dataset_id(with_news, ["AAPL"], effective_end_date="2026-02-01")
    id_news_alt = compute_dataset_id(with_news_alt, ["AAPL"], effective_end_date="2026-02-01")

    assert id_base != id_news
    assert id_news != id_news_alt


def test_update_data_skips_general_news_when_same_endpoint(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "universe_custom.yaml").write_text(yaml.safe_dump({"symbols": ["AAPL"]}, sort_keys=False))

    cfg = {
        "universe": {"provider": "custom_list", "fallback_small50": {"enabled": False, "symbols": []}},
        "data": {
            "start_date": "2026-02-20",
            "end_date": "2026-02-20",
            "adjusted_flag": False,
            "endpoints_version": ["prices_eod", "stock_news", "general_news"],
            "include_news": True,
        },
        "fmp": {"endpoints": {"stock_news": "news/stock", "general_news": "news/stock"}},
    }
    paths = ProjectPaths.from_project_dir(tmp_path)
    paths.ensure_base_dirs()

    class DummyFMP:
        def __init__(self) -> None:
            self.general_news_calls = 0

        def endpoint_for(self, endpoint_name: str) -> str:
            if endpoint_name in {"stock_news", "general_news"}:
                return "news/stock"
            return endpoint_name

        def get_prices(self, symbol: str, start: str, end: str):
            return [
                {
                    "date": "2026-02-20",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.5,
                    "close": 10.5,
                    "volume": 1000,
                }
            ]

        def get_dividends(self, *args, **kwargs):
            return []

        def get_splits(self, *args, **kwargs):
            return []

        def get_earnings(self, *args, **kwargs):
            return []

        def get_earnings_surprises(self, *args, **kwargs):
            return []

        def get_income_statement(self, *args, **kwargs):
            return []

        def get_balance_sheet(self, *args, **kwargs):
            return []

        def get_cash_flow(self, *args, **kwargs):
            return []

        def get_stock_news(self, symbol: str, limit: int = 50):
            return [
                {
                    "symbol": symbol,
                    "publishedDate": "2026-02-20 10:00:00",
                    "title": f"{symbol} news",
                    "url": f"https://example.com/{symbol}",
                }
            ]

        def get_general_news(self, limit: int = 100):
            self.general_news_calls += 1
            return [{"symbol": "GENERAL", "publishedDate": "2026-02-20 11:00:00", "title": "market", "url": "x"}]

    fmp = DummyFMP()
    dataset_id, _, _ = update_data(cfg, paths, fmp, force=False)
    assert fmp.general_news_calls == 0

    news = pd.read_parquet(paths.raw_dir / dataset_id / "news.parquet")
    assert len(news) == 1
    assert news.iloc[0]["symbol"] == "AAPL"
