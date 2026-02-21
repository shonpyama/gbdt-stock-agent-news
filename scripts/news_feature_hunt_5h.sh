#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/content/gbdt-stock-agent-news-export-20260221}"
HOURS="${HOURS:-5}"
TRIALS_PER_CYCLE="${TRIALS_PER_CYCLE:-10}"
DRIVE_PATH="${DRIVE_PATH:-/content/drive/MyDrive/gbdt-stock-agent-news}"

cd "${ROOT}" || exit 1

mkdir -p logs/autonomous reports/autonomous
START_TS="$(date -u +%Y%m%d_%H%M%SZ)"
LOG_FILE="logs/autonomous/news_feature_hunt_5h_${START_TS}.log"

END_EPOCH="$(python - <<'PY'
import time, os
h = float(os.environ.get("HOURS", "5"))
print(int(time.time() + h * 3600))
PY
)"

echo "start_utc=${START_TS} end_epoch=${END_EPOCH} hours=${HOURS}" | tee -a "${LOG_FILE}"

CYCLE=0
while [ "$(date +%s)" -lt "${END_EPOCH}" ]; do
  CYCLE=$((CYCLE + 1))
  CYCLE_TS="$(date -u +%Y%m%d_%H%M%SZ)"
  CYCLE_LOG="logs/autonomous/cycle_${CYCLE_TS}.log"
  SEED="$(( (RANDOM % 100000) + CYCLE * 13 + 42 ))"

  echo "cycle=${CYCLE} ts=${CYCLE_TS} seed=${SEED} status=started" | tee -a "${LOG_FILE}"

  PYTHONPATH=src ./scripts/news_feature_hunt.py --trials "${TRIALS_PER_CYCLE}" --seed "${SEED}" > "${CYCLE_LOG}" 2>&1
  RC=$?
  echo "cycle=${CYCLE} ts=${CYCLE_TS} rc=${RC}" | tee -a "${LOG_FILE}"

  if [ -f reports/feature_search_news_recent_results.json ]; then
    cp reports/feature_search_news_recent_results.json "reports/autonomous/feature_search_news_recent_results_${CYCLE_TS}.json"
  fi
  if [ -f reports/feature_search_news_recent_results.md ]; then
    cp reports/feature_search_news_recent_results.md "reports/autonomous/feature_search_news_recent_results_${CYCLE_TS}.md"
  fi

  PYTHONPATH=src python -m gbdt_agent.cli ops-snapshot --max-age-hours 120 --require-gpu >> "${LOG_FILE}" 2>&1 || true

  git add scripts/news_feature_hunt.py reports/autonomous reports/feature_search_news_recent_results.json reports/feature_search_news_recent_results.md reports/ops_snapshot_*.md conf/experiments/autonomous 2>/dev/null || true
  if ! git diff --cached --quiet; then
    git commit -m "Autonomous news feature hunt cycle ${CYCLE} ${CYCLE_TS}" >> "${LOG_FILE}" 2>&1 || true
    git push origin main >> "${LOG_FILE}" 2>&1 || true
  fi

  PYTHONPATH=src python -m gbdt_agent.cli colab sync --drive-path "${DRIVE_PATH}" >> "${LOG_FILE}" 2>&1 || true
  sleep 20
done

echo "completed_utc=$(date -u +%Y%m%d_%H%M%SZ) cycles=${CYCLE}" | tee -a "${LOG_FILE}"
