#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${ROOT_DIR}/.preview_pids"
API_PORT="${API_PORT:-8001}"
FRONTEND_PORT="${FRONTEND_PORT:-3001}"

kill_pid() {
  local pid="$1"
  local label="$2"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping ${label}: ${pid}"
    kill "${pid}" 2>/dev/null || true
  fi
}

kill_port() {
  local port="$1"
  local label="$2"
  local pids
  pids="$(lsof -ti "tcp:${port}" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    echo "Stopping ${label} on port ${port}: ${pids}"
    kill ${pids} 2>/dev/null || true
  fi
}

if [[ -f "${PID_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${PID_FILE}"
  kill_pid "${FRONTEND_PID:-}" "frontend"
  kill_pid "${API_PID:-}" "backend"
fi

kill_port "${FRONTEND_PORT}" "frontend"
kill_port "${API_PORT}" "backend"
rm -f "${PID_FILE}"

echo "Preview processes stopped."
