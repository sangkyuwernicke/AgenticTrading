#!/bin/bash
# Helper script to run the Agentic RL Backtester skill

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../../" && pwd )"

cd "$PROJECT_ROOT" || exit 1

# Find virtual environment python
PYTHON_CMD="python3"
if [ -d ".venv" ]; then
    PYTHON_CMD="./.venv/bin/python"
elif [ -d "venv" ]; then
    PYTHON_CMD="./venv/bin/python"
fi

MODE="backtest"
EPISODES=10

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift ;;
        --episodes) EPISODES="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "🔍 Checking if MCP Data Pool (Port 8011) is listening..."
if ! lsof -i :8011 > /dev/null; then
    echo "⚠️ MCP Data Pool is not running! Starting agent pools..."
    bash tests/start_agent_pools.sh
    sleep 3
else
    echo "✅ MCP Data Pool is active."
fi

echo "🚀 Launching Agentic RL Backtester (Mode: $MODE)..."
if [ "$MODE" = "train" ]; then
    $PYTHON_CMD tests/test_agentic_rl_backtester.py --mode train --episodes "$EPISODES"
else
    $PYTHON_CMD tests/test_agentic_rl_backtester.py --mode backtest
fi
