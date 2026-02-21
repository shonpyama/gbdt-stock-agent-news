#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from gbdt_agent.orchestrator import run_pipeline
from gbdt_agent.paths import ProjectPaths


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


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
    return (rank_ic * 100.0) + (sharpe * 0.2) + (total_ret * 0.002) + (mdd * 0.2)


def _pick_lookbacks(rng: random.Random) -> List[int]:
    pool = [1, 2, 3, 5, 10, 20, 40, 60, 120]
    k = rng.choice([5, 6, 7])
    vals = sorted(set(rng.sample(pool, k=k)))
    if 1 not in vals:
        vals.insert(0, 1)
    if 20 not in vals:
        vals.append(20)
    return sorted(set(vals))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default="/content/gbdt-stock-agent-news-export-20260221")
    ap.add_argument("--base-conf", default="conf/smoke_news_recent.yaml")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    project_dir = Path(args.project_dir)
    paths = ProjectPaths.from_project_dir(project_dir)
    base_conf_path = project_dir / args.base_conf
    base_cfg = yaml.safe_load(base_conf_path.read_text())

    ts = _utc_ts()
    out_dir = project_dir / "reports" / "autonomous"
    out_dir.mkdir(parents=True, exist_ok=True)
    conf_dir = project_dir / "conf" / "experiments" / "autonomous"
    conf_dir.mkdir(parents=True, exist_ok=True)

    candidates: List[Dict[str, Any]] = []
    for i in range(args.trials):
        include_news = bool(rng.random() < 0.75)
        candidates.append(
            {
                "name": f"auto_{i+1:02d}",
                "include_news": include_news,
                "lookbacks": _pick_lookbacks(rng),
                "event_shift": rng.choice([1, 1, 2]),
                "horizon": rng.choice([5, 10, 20]),
                "min_finite_ratio": rng.choice([0.05, 0.1, 0.15]),
                "stock_window_days": rng.choice([14, 21, 31]),
                "stock_limit": rng.choice([100, 200]),
                "stock_max_pages": rng.choice([3, 5]),
            }
        )
    # Always include one clean baseline.
    candidates.append(
        {
            "name": "baseline_news_off",
            "include_news": False,
            "lookbacks": [1, 3, 5, 10, 20, 60, 120],
            "event_shift": 1,
            "horizon": 20,
            "min_finite_ratio": 0.1,
            "stock_window_days": 31,
            "stock_limit": 200,
            "stock_max_pages": 5,
        }
    )

    results: List[Dict[str, Any]] = []
    for idx, c in enumerate(candidates, start=1):
        cfg = deepcopy(base_cfg)
        cfg["data"]["include_news"] = bool(c["include_news"])
        cfg.setdefault("data", {}).setdefault("news_fetch", {})
        cfg["data"]["news_fetch"]["stock_fetch_historical"] = True
        cfg["data"]["news_fetch"]["stock_window_days"] = int(c["stock_window_days"])
        cfg["data"]["news_fetch"]["stock_limit"] = int(c["stock_limit"])
        cfg["data"]["news_fetch"]["stock_max_pages"] = int(c["stock_max_pages"])
        cfg["data"]["news_fetch"]["stock_max_records_per_symbol"] = 2500
        cfg["features"]["lookbacks"] = list(c["lookbacks"])
        cfg["features"]["event_safe_shift_days"] = int(c["event_shift"])
        cfg["labels"]["horizon_trading_days"] = int(c["horizon"])
        cfg["split"]["walkforward"] = {"train": 50, "val": 10, "test": 10, "step": 10}
        cfg["split"]["purge"] = 1
        cfg["split"]["embargo"] = 1
        cfg["validation"]["thresholds"]["min_obs_per_symbol"] = 40
        cfg.setdefault("models", {}).setdefault("gbdt", {})["min_finite_ratio"] = float(c["min_finite_ratio"])

        conf_path = conf_dir / f"{ts}_{idx:02d}_{c['name']}.yaml"
        conf_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        item: Dict[str, Any] = {
            "name": c["name"],
            "conf_path": str(conf_path),
            "include_news": c["include_news"],
            "lookbacks": c["lookbacks"],
            "event_shift": c["event_shift"],
            "horizon": c["horizon"],
            "min_finite_ratio": c["min_finite_ratio"],
            "ok": False,
        }
        try:
            run_id = run_pipeline(
                project_dir=project_dir,
                conf_path=conf_path,
                resume=False,
                force_unlock=True,
            )
            metrics = json.loads((paths.run_dir(run_id) / "metrics.json").read_text())
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
            tinfo = metrics.get("training_info") or {}
            item["n_features"] = tinfo.get("n_features")
            item["feature_cols"] = tinfo.get("feature_cols")
            item["dropped_feature_cols"] = tinfo.get("dropped_feature_cols")
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)

    ranked = sorted(results, key=lambda x: float(x.get("score", -1e9)), reverse=True)
    payload = {
        "generated_at_utc": ts,
        "base_conf": str(base_conf_path),
        "seed": args.seed,
        "ranked": ranked,
    }
    out_json = out_dir / f"news_feature_hunt_{ts}.json"
    out_md = out_dir / f"news_feature_hunt_{ts}.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True))

    lines = [
        f"# News Feature Hunt {ts}",
        "",
        f"- base_conf: `{base_conf_path}`",
        f"- trials: `{len(results)}`",
        f"- seed: `{args.seed}`",
        "",
        "| rank | name | include_news | horizon | score | rank_ic_test_mean | sharpe | total_return | run_id |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(ranked, start=1):
        lines.append(
            f"| {i} | {r.get('name')} | {r.get('include_news')} | {r.get('horizon')} | {r.get('score')} | {r.get('rank_ic_test_mean')} | {r.get('sharpe')} | {r.get('total_return')} | {r.get('run_id','-')} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"results_json": str(out_json), "results_md": str(out_md), "top": ranked[:3]}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
