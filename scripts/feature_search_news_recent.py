#!/usr/bin/env python3
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml

from gbdt_agent.orchestrator import run_pipeline
from gbdt_agent.paths import ProjectPaths


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _score(metrics: Dict[str, Any]) -> float:
    model = ((metrics.get("model_metrics") or {}).get("gbdt") or {}).get("test") or {}
    rank_ic = _safe_float(((model.get("rank_ic") or {}).get("mean")))
    back = ((metrics.get("backtest") or {}).get("summary") or {})
    sharpe = _safe_float(back.get("sharpe"))
    total_ret = _safe_float(back.get("total_return"))
    mdd = _safe_float(back.get("max_drawdown"))
    if rank_ic != rank_ic:
        rank_ic = -1.0
    if sharpe != sharpe:
        sharpe = -10.0
    if total_ret != total_ret:
        total_ret = -10.0
    if mdd != mdd:
        mdd = -1.0
    # predictive quality first, then tradability, and penalize deep drawdown.
    return (rank_ic * 100.0) + (sharpe * 0.2) + (total_ret * 0.002) + (mdd * 0.2)


def main() -> int:
    project_dir = Path("/content/gbdt-stock-agent-news-export-20260221")
    paths = ProjectPaths.from_project_dir(project_dir)
    base_conf_path = project_dir / "conf" / "smoke_news_recent.yaml"
    base_cfg = yaml.safe_load(base_conf_path.read_text())

    candidates: List[Dict[str, Any]] = [
        {"name": "news_on_h20_lb_1_3_5_10_20_60_120", "include_news": True, "lookbacks": [1, 3, 5, 10, 20, 60, 120], "event_shift": 1, "horizon": 20},
        {"name": "news_off_h20_lb_1_3_5_10_20_60_120", "include_news": False, "lookbacks": [1, 3, 5, 10, 20, 60, 120], "event_shift": 1, "horizon": 20},
        {"name": "news_on_h10_lb_1_3_5_10_20", "include_news": True, "lookbacks": [1, 3, 5, 10, 20], "event_shift": 1, "horizon": 10},
        {"name": "news_off_h10_lb_1_3_5_10_20", "include_news": False, "lookbacks": [1, 3, 5, 10, 20], "event_shift": 1, "horizon": 10},
        {"name": "news_on_h5_lb_1_3_5_10_20", "include_news": True, "lookbacks": [1, 3, 5, 10, 20], "event_shift": 1, "horizon": 5},
        {"name": "news_off_h5_lb_1_3_5_10_20", "include_news": False, "lookbacks": [1, 3, 5, 10, 20], "event_shift": 1, "horizon": 5},
        {"name": "news_on_h3_lb_1_3_5_10_20", "include_news": True, "lookbacks": [1, 3, 5, 10, 20], "event_shift": 1, "horizon": 3},
        {"name": "news_off_h3_lb_1_3_5_10_20", "include_news": False, "lookbacks": [1, 3, 5, 10, 20], "event_shift": 1, "horizon": 3},
        {"name": "news_on_h5_lb_1_3_5_10_20_shift2", "include_news": True, "lookbacks": [1, 3, 5, 10, 20], "event_shift": 2, "horizon": 5},
        {"name": "news_off_h5_lb_1_3_5_10_20_shift2", "include_news": False, "lookbacks": [1, 3, 5, 10, 20], "event_shift": 2, "horizon": 5},
    ]

    out_dir = project_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    conf_dir = project_dir / "conf" / "experiments"
    conf_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for idx, c in enumerate(candidates, start=1):
        cfg = deepcopy(base_cfg)
        cfg["data"]["include_news"] = bool(c["include_news"])
        cfg["features"]["lookbacks"] = list(c["lookbacks"])
        cfg["features"]["event_safe_shift_days"] = int(c["event_shift"])
        cfg["labels"]["horizon_trading_days"] = int(c["horizon"])
        # Keep sweep feasible on recent-window smoke data.
        cfg["split"]["walkforward"] = {"train": 50, "val": 10, "test": 10, "step": 10}
        cfg["split"]["purge"] = 1
        cfg["split"]["embargo"] = 1
        cfg["validation"]["thresholds"]["min_obs_per_symbol"] = 40
        cfg.setdefault("models", {}).setdefault("gbdt", {})["min_finite_ratio"] = 0.10
        conf_path = conf_dir / f"news_recent_feature_{idx:02d}_{c['name']}.yaml"
        conf_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        item: Dict[str, Any] = {
            "name": c["name"],
            "conf_path": str(conf_path),
            "include_news": c["include_news"],
            "lookbacks": c["lookbacks"],
            "event_shift": c["event_shift"],
            "horizon": c["horizon"],
            "ok": False,
        }
        try:
            run_id = run_pipeline(
                project_dir=project_dir,
                conf_path=conf_path,
                resume=False,
                force_unlock=True,
            )
            metrics_path = paths.run_dir(run_id) / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            item["run_id"] = run_id
            item["ok"] = bool(metrics.get("status") == "success")
            item["status"] = metrics.get("status")
            mm = ((metrics.get("model_metrics") or {}).get("gbdt") or {}).get("test") or {}
            bt = ((metrics.get("backtest") or {}).get("summary") or {})
            item["rank_ic_test_mean"] = ((mm.get("rank_ic") or {}).get("mean"))
            item["ic_test_mean"] = ((mm.get("ic") or {}).get("mean"))
            item["sharpe"] = bt.get("sharpe")
            item["total_return"] = bt.get("total_return")
            item["max_drawdown"] = bt.get("max_drawdown")
            item["score"] = _score(metrics)
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)

    ranked = sorted(results, key=lambda x: float(x.get("score", -1e9)), reverse=True)
    payload = {"base_conf": str(base_conf_path), "ranked": ranked}
    out_json = out_dir / "feature_search_news_recent_results.json"
    out_md = out_dir / "feature_search_news_recent_results.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True))

    lines = [
        "# Feature Search (News Recent)",
        "",
        f"- base_conf: `{base_conf_path}`",
        f"- trials: `{len(results)}`",
        "",
        "| rank | name | include_news | score | rank_ic_test_mean | sharpe | total_return | max_drawdown | run_id |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(ranked, start=1):
        lines.append(
            f"| {i} | {r.get('name')} | {r.get('include_news')} | {r.get('score')} | {r.get('rank_ic_test_mean')} | {r.get('sharpe')} | {r.get('total_return')} | {r.get('max_drawdown')} | {r.get('run_id','-')} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"results_json": str(out_json), "results_md": str(out_md), "top": ranked[:3]}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
