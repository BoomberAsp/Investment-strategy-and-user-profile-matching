#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8001}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-3001}"
API_BASE_URL="http://${API_HOST}:${API_PORT}"
FRONTEND_URL="http://${FRONTEND_HOST}:${FRONTEND_PORT}"
PID_FILE="${ROOT_DIR}/.preview_pids"
LOG_DIR="${ROOT_DIR}/logs"

mkdir -p "${LOG_DIR}"

kill_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti "tcp:${port}" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    echo "Stopping existing process on port ${port}: ${pids}"
    kill ${pids} 2>/dev/null || true
    sleep 1
  fi
}

wait_for_url() {
  local url="$1"
  local name="$2"
  local max_attempts="${3:-60}"

  for ((i = 1; i <= max_attempts; i++)); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "${name} is ready: ${url}"
      return 0
    fi
    sleep 1
  done

  echo "${name} did not become ready in time. Check logs in ${LOG_DIR}." >&2
  return 1
}

echo "Starting investment strategy preview..."
kill_port "${FRONTEND_PORT}"
kill_port "${API_PORT}"

: > "${PID_FILE}"

cd "${ROOT_DIR}"
python -m uvicorn api.main:app --host "${API_HOST}" --port "${API_PORT}" \
  > "${LOG_DIR}/api.log" 2>&1 &
API_PID="$!"
echo "API_PID=${API_PID}" >> "${PID_FILE}"
wait_for_url "${API_BASE_URL}/api/health" "Backend"

cd "${ROOT_DIR}/frontend"
NEXT_PUBLIC_API_BASE_URL="${API_BASE_URL}" npm run dev -- --hostname "${FRONTEND_HOST}" --port "${FRONTEND_PORT}" \
  > "${LOG_DIR}/frontend.log" 2>&1 &
FRONTEND_PID="$!"
echo "FRONTEND_PID=${FRONTEND_PID}" >> "${PID_FILE}"
wait_for_url "${FRONTEND_URL}" "Frontend"

cat <<EOF

Preview is running.
Frontend: ${FRONTEND_URL}
Backend:  ${API_BASE_URL}

Logs:
  ${LOG_DIR}/api.log
  ${LOG_DIR}/frontend.log

Stop with:
  scripts/stop_preview.sh
EOF
