#!/bin/bash

# 스크립트가 위치한 디렉터리 가져오기 (프로젝트 루트 디렉터리)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AGENTS_DIR="$PROJECT_ROOT/FinAgents"
POOLS_DIR="$AGENTS_DIR/agent_pools"

# PYTHONPATH 설정
# Python이 각 에이전트 모듈을 찾을 수 있도록 각 레벨의 디렉터리 포함
# 여기서 POOLS_DIR/alpha_agent_pool 경로는 시스템에 설치된 qlib 패키지 대신 로컬 qlib 디렉터리를 임포트하게 만듭니다.
# 따라서 이를 마지막에 배치하거나, 로컬 qlib 오버라이드가 필요 없는 경우 직접 제거할 수 있습니다.
export PYTHONPATH="$PROJECT_ROOT:$AGENTS_DIR:$POOLS_DIR:$POOLS_DIR/alpha_agent_demo:$POOLS_DIR/risk_agent_demo:$POOLS_DIR/portfolio_agent_demo:$POOLS_DIR/execution_agent_demo/execution_agent_demo:$POOLS_DIR/backtest_agent:$PYTHONPATH"

# API Key 설정 (로컬에 설정되어 있지 않다면 여기서 임시로 export하거나 .env 파일에 의존)
# export OPENAI_API_KEY="your_key_here"
# export ALPACA_API_KEY="your_key_here"
# export ALPACA_SECRET_KEY="your_key_here"

echo "🚀 Starting Orchestrator Demo..."
echo "📂 Project Root: $PROJECT_ROOT"
echo "🐍 PYTHONPATH configured."

# 오케스트레이터 실행
python3 "$AGENTS_DIR/orchestrator_demo/orchestrator.py"
