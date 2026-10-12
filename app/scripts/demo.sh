#!/bin/sh
# Start the app against the DEMO database (<real database name>_demo) with the example-data banner.
# Usage: sh scripts/demo.sh [port]      (default port 8100)
# Build the demo first:  uv run robby-admin demo load
set -e
cd "$(dirname "$0")/.."
PORT="${1:-8100}"
DEMO_URL="${ROBBY_DEMO_DATABASE_URL:-$(uv run python -c "from robby.config import get_settings; from robby.demo.loader import default_demo_url; print(default_demo_url(get_settings().database_url))")}"
export ROBBY_DATABASE_URL="$DEMO_URL"
export ROBBY_DEMO=true
export ROBBY_SECURE_COOKIES=false
export ROBBY_ARTEFACT_DIR="$(pwd)/artefacts/demo"
echo "Demo mode: database ${DEMO_URL##*/}, http://127.0.0.1:$PORT (password for demo accounts: demo-password-123)"
exec uv run uvicorn robby.main:create_app --factory --host 127.0.0.1 --port "$PORT"
