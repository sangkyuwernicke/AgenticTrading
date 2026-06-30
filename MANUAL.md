# AgenticTrading (FinAgent Orchestration) — 코드 구조 분석 및 메뉴얼

> 최종 업데이트: 2026-06-30  
> 브랜치: `agents-code-structure-analysis-and-manual`

---

## 1. 프로젝트 개요

**FinAgent Orchestration**은 알고리즘 트레이딩을 "에이전틱 트레이딩"으로 전환하는 멀티-에이전트 프레임워크다.  
각 트레이딩 파이프라인 단계를 자율 에이전트로 구현하며, MCP(Model Context Protocol), A2A(Agent-to-Agent) 프로토콜을 통해 에이전트 간 통신을 표준화한다.  
메모리 에이전트(Neo4j + 벡터 임베딩)가 공유 컨텍스트 허브로 작동하여 전략의 지속 학습을 지원한다.

---

## 2. 전체 디렉토리 구조

```
AgenticTrading/
├── FinAgents/                         # 핵심 프레임워크 모듈
│   ├── __init__.py
│   ├── orchestrator/                  # 오케스트레이터 엔진
│   │   ├── main_orchestrator.py       # 메인 진입점 (CLI 엔트리포인트)
│   │   ├── finagent_cli.py            # CLI 인터페이스
│   │   ├── core_service_manager.py    # 서비스 매니저
│   │   ├── integration_example.py     # 통합 예제
│   │   ├── config/
│   │   │   └── orchestrator_config.yaml  # 전체 시스템 설정 파일
│   │   └── core/                      # 오케스트레이터 핵심 로직
│   │       ├── finagent_orchestrator.py  # 메인 오케스트레이터 클래스
│   │       ├── dag_planner.py            # DAG 기반 태스크 플래너
│   │       ├── rl_policy_engine.py       # 강화학습 정책 엔진
│   │       ├── sandbox_environment.py    # 백테스트/스트레스 테스트 샌드박스
│   │       ├── llm_integration.py        # LLM 연동 (NaturalLanguageProcessor)
│   │       ├── health_server.py          # 헬스체크 HTTP 서버
│   │       ├── agent_pool_monitor.py     # 에이전트 풀 모니터링
│   │       ├── mcp_nl_interface.py       # MCP 자연어 인터페이스
│   │       └── orchestrator.py           # 레거시 오케스트레이터
│   │
│   ├── agent_pools/                   # 모듈형 에이전트 풀
│   │   ├── alpha_agent_pool/          # 알파 신호 생성 풀
│   │   │   ├── core.py                # MCP 서버 (AlphaAgentPoolMCPServer)
│   │   │   ├── enhanced_mcp_server.py # 확장 MCP 서버 (FastAPI + MCP)
│   │   │   ├── alpha_pool_gateway.py  # 외부-내부 게이트웨이
│   │   │   ├── a2a_memory_coordinator.py  # A2A 메모리 코디네이터
│   │   │   ├── alpha_research_agent.py    # 알파 리서치 에이전트
│   │   │   ├── registry.py            # 에이전트 레지스트리
│   │   │   ├── core/                  # 헥사고날 아키텍처
│   │   │   │   ├── domain/models.py   # 도메인 모델
│   │   │   │   ├── ports/             # 인터페이스 포트
│   │   │   │   └── services/         # 비즈니스 서비스
│   │   │   ├── agents/
│   │   │   │   ├── theory_driven/     # 이론 기반 에이전트
│   │   │   │   │   ├── momentum_agent.py       # 모멘텀 전략
│   │   │   │   │   └── mean_reversion_agent.py # 평균 회귀 전략
│   │   │   │   ├── empirical/         # 실증 기반 에이전트
│   │   │   │   │   ├── data_mining_agent.py
│   │   │   │   │   └── ml_pattern_agent.py
│   │   │   │   └── autonomous/        # 자율 에이전트
│   │   │   │       └── autonomous_agent.py
│   │   │   ├── rl_llm/                # RL + LLM 통합
│   │   │   └── tests/                 # 풀 단위 테스트
│   │   │
│   │   ├── data_agent_pool/           # 시장 데이터 수집 풀
│   │   │   ├── core.py                # DataAgentPool MCP 서버
│   │   │   ├── mcp_server.py          # 심플 MCP 서버
│   │   │   ├── registry.py            # 에이전트 레지스트리
│   │   │   ├── schema.py              # 데이터 스키마
│   │   │   ├── memory_bridge.py       # 메모리 브릿지
│   │   │   └── agents/
│   │   │       ├── equity/            # 주식 데이터 에이전트
│   │   │       │   ├── polygon_agent.py
│   │   │       │   ├── yfinance_server.py
│   │   │       │   ├── alpaca_agent.py
│   │   │       │   ├── iex_agent.py
│   │   │       │   └── market_data.py
│   │   │       ├── news/              # 뉴스 데이터 에이전트
│   │   │       │   ├── newsapi_agent.py
│   │   │       │   ├── alphavantage_agent.py
│   │   │       │   └── rss_agent.py
│   │   │       └── crypto/            # 암호화폐 데이터 에이전트
│   │   │           ├── binance_agent.py
│   │   │           ├── coinbase_agent.py
│   │   │           └── coingecko_agent.py
│   │   │
│   │   ├── risk_agent_pool/           # 리스크 관리 풀
│   │   │   ├── core.py
│   │   │   ├── registry.py
│   │   │   ├── memory_bridge.py
│   │   │   └── agents/
│   │   │       ├── market_risk.py     # 시장 리스크
│   │   │       ├── credit_risk.py     # 신용 리스크
│   │   │       ├── liquidity_risk.py  # 유동성 리스크
│   │   │       ├── operational_risk.py
│   │   │       ├── model_risk.py
│   │   │       ├── stress_testing.py  # 스트레스 테스트
│   │   │       ├── var_calculator.py  # VaR 계산기
│   │   │       └── volatility.py      # 변동성 분석
│   │   │
│   │   ├── transaction_cost_agent_pool/  # 거래비용 분석 풀
│   │   │   ├── core.py
│   │   │   ├── registry.py
│   │   │   ├── memory_bridge.py
│   │   │   └── agents/
│   │   │       ├── pre_trade/         # 사전 거래 분석
│   │   │       │   ├── cost_predictor.py
│   │   │       │   ├── impact_estimator.py
│   │   │       │   └── venue_analyzer.py
│   │   │       ├── post_trade/        # 사후 거래 분석
│   │   │       │   ├── attribution_engine.py
│   │   │       │   ├── execution_analyzer.py
│   │   │       │   └── slippage_analyzer.py
│   │   │       └── optimization/      # 실행 최적화
│   │   │           ├── cost_optimizer.py
│   │   │           ├── routing_optimizer.py
│   │   │           └── timing_optimizer.py
│   │   │
│   │   ├── portfolio_construction_agent_pool/  # 포트폴리오 구성 풀
│   │   │   ├── core.py
│   │   │   ├── config.py
│   │   │   └── memory_bridge.py
│   │   │
│   │   └── backtest_agent/            # 백테스트 풀
│   │       ├── backtest_agent.py
│   │       ├── backtest_visualizer.py
│   │       └── local_agents.py
│   │
│   └── memory/                        # 메모리 에이전트
│       ├── mcp_server.py              # MCP 서버 (포트 8001)
│       ├── a2a_server.py              # A2A 프로토콜 서버
│       ├── memory_server.py           # REST API 메모리 서버
│       ├── unified_database_manager.py # Neo4j DB 통합 매니저
│       ├── unified_interface_manager.py # 통합 인터페이스 매니저
│       ├── intelligent_memory_indexer.py # 지능형 메모리 인덱서
│       ├── database_initializer.py    # DB 초기화
│       ├── configuration_manager.py   # 설정 관리
│       ├── llm_research_service.py    # LLM 리서치 서비스
│       ├── realtime_stream_processor.py # 실시간 스트림
│       ├── a2a_health_checker.py
│       └── interface.py               # 레거시 인터페이스
│
├── tests/                             # 통합/기능 테스트
│   ├── comprehensive_integration_test.py
│   ├── simple_integration_test.py
│   ├── test_simple_llm_backtest.py
│   ├── test_alpha_memory_a2a_connection.py
│   ├── conftest.py
│   └── data_error_rate/               # 데이터 에러율 측정 테스트
│
├── examples/                          # 사용 예제
│   ├── 1_backtest_training.py         # 백테스트 학습 예제
│   ├── 2_out_of_sample_inference_test.py  # 아웃오브샘플 추론
│   ├── autonomous_agent_example.py    # 자율 에이전트 예제
│   ├── alpha_agent_pool_memory_integration_examples.py
│   ├── example_momentum_run.py        # 모멘텀 전략 실행
│   ├── example_a2a_momentum_integration.py
│   ├── polygon_batch_fetch_via_mcp.py # Polygon MCP 배치 조회
│   └── external_memory_agent_demo.py  # 메모리 에이전트 데모
│
├── scripts/                           # 유틸리티 스크립트
│   ├── setup_neo4j.py                 # Neo4j 초기화
│   ├── setup_integration.sh           # 환경 설정 스크립트
│   └── reddit_monitor.py              # Reddit 모니터링
│
├── data/                              # 시장 데이터 캐시 및 로그
├── docs/                              # Sphinx 기반 문서
├── pyproject.toml                     # 프로젝트 메타데이터
└── requirements.txt                   # Python 의존성
```

---

## 3. 아키텍처 레이어

```
┌─────────────────────────────────────────────────────┐
│              Users Query / CLI Interface             │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│         DAG Planner (dag_planner.py)                │
│  LLM이 자연어 쿼리를 DAG 실행 계획으로 변환          │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│     FinAgent Orchestrator (finagent_orchestrator.py) │
│  DAG 실행 · 에이전트 풀 조율 · 헬스 모니터링         │
└────┬───────────┬─────────────┬────────────┬─────────┘
     │           │             │            │
  :8001       :8081         :8084        :8085
     │           │             │            │
┌────▼───┐ ┌────▼────┐ ┌──────▼───┐ ┌─────▼──────┐
│ Data   │ │ Alpha   │ │  Risk    │ │Transaction │
│ Agent  │ │ Agent   │ │  Agent   │ │   Cost     │
│ Pool   │ │  Pool   │ │   Pool   │ │   Pool     │
└────────┘ └─────────┘ └──────────┘ └────────────┘
     │           │             │            │
     └───────────┴─────────────┴────────────┘
                       │
              ┌────────▼────────┐
              │  Memory Agent   │ ← Neo4j + Vector
              │  (port 8010)    │    Embeddings
              └─────────────────┘
```

---

## 4. 주요 컴포넌트 상세

### 4.1 Orchestrator (`FinAgents/orchestrator/`)

| 파일 | 역할 |
|------|------|
| `main_orchestrator.py` | CLI 진입점. `--mode` 인수로 production/development/sandbox 모드 선택 |
| `core/finagent_orchestrator.py` | `FinAgentOrchestrator` 클래스. 에이전트 풀 등록, 전략 실행, RL 엔진 초기화 |
| `core/dag_planner.py` | `DAGPlanner`. NetworkX DAG로 트레이딩 태스크 계획 수립, LLM 통합 |
| `core/rl_policy_engine.py` | DDPG/PPO/SAC/TD3 RL 알고리즘 구현. 보상함수: Sharpe/Sortino/MaxDrawdown |
| `core/sandbox_environment.py` | 백테스트, 스트레스 테스트, A/B 테스트 샌드박스 환경 |
| `core/llm_integration.py` | `NaturalLanguageProcessor`. 자연어 → 전략 계획 변환 |
| `config/orchestrator_config.yaml` | 모든 에이전트 풀 URL, RL 하이퍼파라미터, 샌드박스 설정 포함 |

### 4.2 Alpha Agent Pool (`FinAgents/agent_pools/alpha_agent_pool/`)

**포트**: 8081 (Enhanced MCP), 8082 (Gateway)

| 컴포넌트 | 역할 |
|----------|------|
| `core.py` (AlphaAgentPoolMCPServer) | FastMCP 기반 알파 풀 메인 서버 |
| `enhanced_mcp_server.py` | FastAPI + FastMCP 복합 서버. `/tools`, `/health`, `/sse` 엔드포인트 제공 |
| `alpha_pool_gateway.py` | 외부 오케스트레이터를 위한 MCP 서버 겸 내부 에이전트용 MCP 클라이언트 |
| `agents/theory_driven/momentum_agent.py` | 모멘텀 기반 알파 신호 생성 |
| `agents/theory_driven/mean_reversion_agent.py` | 평균 회귀 신호 생성 |
| `agents/empirical/ml_pattern_agent.py` | ML 기반 패턴 인식 |
| `agents/autonomous/autonomous_agent.py` | LLM 자율 에이전트 |
| `core/` | 헥사고날 아키텍처 (domain/ports/services) |

### 4.3 Data Agent Pool (`FinAgents/agent_pools/data_agent_pool/`)

**포트**: 8001 (Pool), 8002 (Polygon), 8003 (Binance)

| 에이전트 | 데이터 소스 |
|----------|-------------|
| `polygon_agent.py` | Polygon.io (미국 주식/옵션) |
| `yfinance_server.py` | Yahoo Finance |
| `alpaca_agent.py` | Alpaca Markets |
| `iex_agent.py` | IEX Cloud |
| `binance_agent.py` | Binance 암호화폐 |
| `coinbase_agent.py` | Coinbase |
| `coingecko_agent.py` | CoinGecko |
| `newsapi_agent.py` | News API |
| `alphavantage_agent.py` | Alpha Vantage |
| `rss_agent.py` | RSS 피드 |

### 4.4 Risk Agent Pool (`FinAgents/agent_pools/risk_agent_pool/`)

**포트**: 8084

| 에이전트 | 기능 |
|----------|------|
| `market_risk.py` | 시장 리스크 모델링 |
| `var_calculator.py` | Value-at-Risk (VaR) 계산 |
| `credit_risk.py` | 신용 리스크 분석 |
| `liquidity_risk.py` | 유동성 리스크 |
| `volatility.py` | 변동성 분석 |
| `stress_testing.py` | 스트레스 테스트 시나리오 |
| `operational_risk.py` | 운영 리스크 |
| `model_risk.py` | 모델 리스크 |

### 4.5 Transaction Cost Agent Pool (`FinAgents/agent_pools/transaction_cost_agent_pool/`)

**포트**: 8085

| 서브 모듈 | 에이전트 |
|-----------|----------|
| `pre_trade/` | `cost_predictor`, `impact_estimator`, `venue_analyzer` |
| `post_trade/` | `attribution_engine`, `execution_analyzer`, `slippage_analyzer` |
| `optimization/` | `cost_optimizer`, `routing_optimizer`, `timing_optimizer` |

### 4.6 Memory Agent (`FinAgents/memory/`)

**포트**: 8010 (REST + MCP + A2A)

| 컴포넌트 | 역할 |
|----------|------|
| `mcp_server.py` | FastMCP 서버. 6개 MCP 도구 제공 |
| `a2a_server.py` | A2A 프로토콜 서버 (에이전트-to-에이전트 직접 통신) |
| `memory_server.py` | REST API 메모리 서버 |
| `unified_database_manager.py` | Neo4j 그래프 DB 통합 관리 |
| `unified_interface_manager.py` | 모든 메모리 작업의 단일 진입점 |
| `intelligent_memory_indexer.py` | 벡터 임베딩 기반 의미론적 인덱싱 |

---

## 5. 통신 프로토콜

| 프로토콜 | 용도 | 구현 위치 |
|----------|------|-----------|
| **MCP** (Model Context Protocol) | 에이전트 도구 호출, DAG 실행 | `mcp_server.py` (각 풀) |
| **A2A** (Agent-to-Agent) | 피어-투-피어 에이전트 직접 통신 | `memory/a2a_server.py` |
| **REST/HTTP** | 헬스체크, 상태 조회 | `/health`, `/status` 엔드포인트 |
| **SSE** (Server-Sent Events) | 실시간 스트리밍 | `alpha_pool_gateway.py` |

---

## 6. 서비스 포트 맵

| 서비스 | 포트 | 프로토콜 |
|--------|------|----------|
| Memory Agent (MCP) | 8001 | MCP/HTTP |
| Memory Agent (REST) | 8010 | REST |
| Data Agent Pool | 8001 | MCP |
| Polygon Agent | 8002 | MCP |
| Binance Agent | 8003 | MCP |
| Alpha Agent Pool (Enhanced) | 8081 | MCP+FastAPI |
| Alpha Pool Gateway | 8082 | MCP |
| Risk Agent Pool | 8084 | MCP |
| Transaction Cost Pool | 8085 | MCP |
| Orchestrator | 9000 | HTTP |

---

## 7. 빠른 시작 가이드

### 7.1 환경 설정

```bash
# 1. 클론
git clone https://github.com/Open-Finance-Lab/AgenticTrading.git
cd AgenticTrading

# 2. 의존성 설치
pip install -r requirements.txt
# 또는 uv 사용
uv sync

# 3. 환경 변수 설정 (.env 파일)
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=finagent123
POLYGON_API_KEY=your_polygon_api_key

# 4. Neo4j 초기화
python scripts/setup_neo4j.py
```

### 7.2 에이전트 풀 시작

```bash
# 전체 에이전트 풀 일괄 시작 (스크립트 활용)
bash tests/start_agent_pools.sh

# 개별 시작
# Data Agent Pool (포트 8001)
cd FinAgents/agent_pools/data_agent_pool
uvicorn mcp_server:app --host 0.0.0.0 --port 8001

# Alpha Agent Pool (포트 8081)
cd FinAgents/agent_pools/alpha_agent_pool
python enhanced_mcp_server.py

# Memory Agent MCP Server (포트 8001)
cd FinAgents/memory
uvicorn mcp_server:app --host 0.0.0.0 --port 8001

# Memory Agent A2A Server (포트 8010)
cd FinAgents/memory
python a2a_server.py --port 8010
```

### 7.3 오케스트레이터 실행

```bash
# Development 모드 (데모 실행)
python FinAgents/orchestrator/main_orchestrator.py --mode development

# Production 모드 (상시 가동)
python FinAgents/orchestrator/main_orchestrator.py \
  --config FinAgents/orchestrator/config/orchestrator_config.yaml \
  --mode production

# Sandbox 모드 (백테스트/스트레스 테스트)
python FinAgents/orchestrator/main_orchestrator.py --mode sandbox
```

### 7.4 백테스트 실행

```bash
# 기본 백테스트 학습
python examples/1_backtest_training.py

# KOSPI 백테스트
python examples/1_backtest_training_kospi.py

# LLM 기반 3년 백테스트 (KOSPI Top 10)
python tests/test_simple_llm_backtest.py

# 아웃오브샘플 추론
python examples/2_out_of_sample_inference_test.py
```

---

## 8. 설정 파일 (`orchestrator_config.yaml`) 주요 항목

```yaml
orchestrator:
  port: 9000
  memory_agent_url: "http://localhost:8010"
  enable_rl: true          # RL 엔진 활성화
  enable_sandbox: true     # 샌드박스 활성화

agent_pools:
  alpha_agent_pool:
    url: "http://localhost:8081"
  risk_agent_pool:
    url: "http://localhost:8084"

rl_engine:
  algorithm: "TD3"         # TD3/SAC/PPO/DDPG
  learning_rate: 0.0003
  batch_size: 256

sandbox:
  initial_capital: 1000000
  commission_rate: 0.001
  benchmark_symbol: "SPY"
```

---

## 9. 테스트 코드 목록 및 사용법

### 9.1 통합 테스트 (`tests/`)

| 파일 | 설명 | 실행 방법 |
|------|------|-----------|
| `simple_integration_test.py` | External Memory Agent 기본 기능 테스트 (이벤트 로깅, 쿼리, 배치) | `python tests/simple_integration_test.py` |
| `comprehensive_integration_test.py` | 메모리 에이전트 종합 테스트 (트랜잭션/최적화/시장데이터 이벤트) | `python tests/comprehensive_integration_test.py` |
| `test_simple_llm_backtest.py` | LLM 기반 KOSPI Top10 3년 백테스트. 실제/합성 데이터 자동 폴백 | `python tests/test_simple_llm_backtest.py` |
| `test_alpha_memory_a2a_connection.py` | Alpha 에이전트 ↔ Memory 에이전트 A2A 연결 테스트 | `python tests/test_alpha_memory_a2a_connection.py` |

```bash
# A2A 연결 테스트 (전용 스크립트 사용)
bash tests/run_a2a_connection_test.sh

# 에이전트 풀 시작 후 테스트
bash tests/start_agent_pools.sh
python tests/test_alpha_memory_a2a_connection.py
bash tests/stop_agent_pools.sh
```

### 9.2 Alpha Agent Pool 테스트 (`FinAgents/agent_pools/alpha_agent_pool/tests/`)

| 파일 | 설명 | 실행 방법 |
|------|------|-----------|
| `test_core_functionality.py` | AlphaAgentPool 핵심 기능 (신호 생성, 팩터 발견) | `python -m pytest FinAgents/agent_pools/alpha_agent_pool/tests/test_core_functionality.py` |
| `test_enhanced_demo_system.py` | EnhancedAlphaPoolDemo 전체 파이프라인 테스트 | `python -m pytest FinAgents/agent_pools/alpha_agent_pool/tests/test_enhanced_demo_system.py` |

### 9.3 Portfolio Construction Pool 테스트

| 파일 | 설명 | 실행 방법 |
|------|------|-----------|
| `test_integration.py` | 포트폴리오 구성 통합 테스트 | `python FinAgents/agent_pools/portfolio_construction_agent_pool/test_integration.py` |
| `test_langgraph.py` | LangGraph 기반 워크플로우 테스트 | `python FinAgents/agent_pools/portfolio_construction_agent_pool/test_langgraph.py` |
| `test_langgraph_demo.py` | LangGraph 데모 테스트 | `python FinAgents/agent_pools/portfolio_construction_agent_pool/test_langgraph_demo.py` |
| `test_memory_unit.py` | 메모리 유닛 저장/조회 테스트 | `python FinAgents/agent_pools/portfolio_construction_agent_pool/test_memory_unit.py` |

### 9.4 Transaction Cost Pool 테스트

| 파일 | 실행 방법 |
|------|-----------|
| `test_integration.py` | `python FinAgents/agent_pools/transaction_cost_agent_pool/test_integration.py` |

### 9.5 Data Agent Pool 테스트

| 파일 | 설명 | 실행 방법 |
|------|------|-----------|
| `test_client.py` | Data Agent Pool MCP 클라이언트 테스트 | `python FinAgents/agent_pools/data_agent_pool/test_client.py` |
| `unified_test_client.py` | 통합 MCP 클라이언트 테스트 | `python FinAgents/agent_pools/data_agent_pool/unified_test_client.py` |
| `unified_test_core_http.py` | Core HTTP 엔드포인트 테스트 | `python FinAgents/agent_pools/data_agent_pool/unified_test_core_http.py` |

### 9.6 Scripts 통합 테스트

| 파일 | 실행 방법 |
|------|-----------|
| `scripts/test_integration_pipeline.py` | 전체 파이프라인 통합 테스트 | `python scripts/test_integration_pipeline.py` |

### 9.7 Examples (사용 예제)

| 파일 | 설명 | 실행 방법 |
|------|------|-----------|
| `autonomous_agent_example.py` | 자율 에이전트 전략 생성 데모 | `python examples/autonomous_agent_example.py` |
| `alpha_agent_pool_memory_integration_examples.py` | Alpha ↔ Memory 통합 예제 | `python examples/alpha_agent_pool_memory_integration_examples.py` |
| `example_momentum_run.py` | 모멘텀 전략 실행 예제 | `python examples/example_momentum_run.py` |
| `example_a2a_momentum_integration.py` | A2A 프로토콜 모멘텀 통합 | `python examples/example_a2a_momentum_integration.py` |
| `polygon_batch_fetch_via_mcp.py` | Polygon MCP 배치 데이터 조회 | `python examples/polygon_batch_fetch_via_mcp.py` |
| `external_memory_agent_demo.py` | 외부 메모리 에이전트 데모 | `python examples/external_memory_agent_demo.py` |
| `openai_agent_test.py` | OpenAI 에이전트 기능 테스트 | `python examples/openai_agent_test.py` |
| `tutorial_step1.py` | 튜토리얼 1단계 | `python examples/tutorial_step1.py` |
| `tutorial_step2.py` | 튜토리얼 2단계 | `python examples/tutorial_step2.py` |

---

## 10. MCP 도구 목록

### 10.1 Memory Agent MCP Server (`FinAgents/memory/mcp_server.py`)

서버명: `FinAgent-MCP-Server` v2.0.0  
연결: `uvicorn mcp_server:app --host 0.0.0.0 --port 8001`

| 도구명 | 설명 | 주요 파라미터 |
|--------|------|---------------|
| `store_memory` | 메모리 레코드 저장 (의미론적 인덱싱 포함) | `query`, `keywords[]`, `summary`, `agent_id`, `event_type`, `session_id` |
| `retrieve_memory` | 메모리 검색 및 조회 | `search_query`, `limit=5` |
| `semantic_search` | AI 기반 의미론적 유사도 검색 | `query`, `limit=10`, `similarity_threshold=0.3` |
| `get_statistics` | 시스템 통계 및 헬스 정보 조회 | (없음) |
| `health_check` | 전체 시스템 컴포넌트 헬스 체크 | (없음) |
| `create_relationship` | 메모리 노드 간 관계 생성 (Neo4j) | `source_memory_id`, `target_memory_id`, `relationship_type` |

**HTTP 엔드포인트**:
- `GET /` — 서버 정보 및 도구 목록
- `GET /health` — 헬스체크

### 10.2 Data Agent Pool MCP Server (`FinAgents/agent_pools/data_agent_pool/mcp_server.py`)

서버명: `Data Agent Pool`  
연결: `uvicorn mcp_server:app --host 0.0.0.0 --port 8001`

| 도구명 | 설명 | 주요 파라미터 |
|--------|------|---------------|
| `agent.execute` | 등록된 에이전트 함수 실행 | `agent_id`, `function`, `input: dict` |

**MCP 리소스**:
- `register://{agent_id}` — 새 에이전트 동적 등록
- `heartbeat://{agent_id}` — 에이전트 하트비트 기록

### 10.3 Alpha Agent Pool Enhanced MCP Server (`FinAgents/agent_pools/alpha_agent_pool/enhanced_mcp_server.py`)

서버명: `AlphaAgentPoolMCPServer`  
연결: `python enhanced_mcp_server.py` (포트 8081)

| 도구명 | 유형 | 설명 | 주요 파라미터 |
|--------|------|------|---------------|
| `generate_alpha_signals` | core | 모멘텀 에이전트 기반 알파 신호 생성 | `symbol`, `symbols[]`, `date`, `lookback_period=20`, `price` |
| `discover_alpha_factors` | core | 알파 팩터 발견 | `factor_categories[]`, `significance_threshold=0.05` |
| `develop_strategy_configuration` | core | 전략 설정 개발 | `risk_level="moderate"`, `target_volatility=0.15` |
| `run_comprehensive_backtest` | core | 종합 백테스트 실행 | `strategy_id`, `start_date="2018-01-01"`, `end_date="2023-12-31"` |
| `submit_strategy_to_memory` | core | 전략을 메모리에 저장 | `strategy_id`, `backtest_id` |
| `run_integrated_backtest` | core | 전체 파이프라인 통합 백테스트 | `strategy_id`, `symbols[]`, `start_date`, `end_date`, `risk_level` |
| `validate_strategy_performance` | core | 전략 성능 검증 | `strategy_id`, `backtest_id` |
| `start_agent` | agent_mgmt | 에이전트 시작 | `agent_name` |
| `list_agents` | agent_mgmt | 에이전트 목록 조회 | (없음) |
| `get_agent_status` | agent_mgmt | 에이전트 상태 조회 | `agent_id` |
| `momentum_health` | agent_mgmt | 모멘텀 에이전트 헬스체크 | (없음) |
| `get_memory` | memory | 메모리 키 조회 | `key` |
| `set_memory` | memory | 메모리 저장 | `key`, `value` |
| `delete_memory` | memory | 메모리 삭제 | `key` |
| `list_memory_keys` | memory | 메모리 키 목록 | (없음) |

**HTTP 엔드포인트**:
- `GET /` — 서버 정보
- `GET /health` — 헬스체크
- `GET /status` — 상세 상태
- `GET /tools` — MCP 도구 목록
- `GET /sse` — SSE MCP 엔드포인트

---

## 11. 강화학습 (RL) 알고리즘

`FinAgents/orchestrator/core/rl_policy_engine.py`에 구현됨.

| 알고리즘 | 클래스명 | 특징 |
|----------|----------|------|
| **TD3** | `TD3Algorithm` | Twin Delayed DDPG. 연속 행동 공간, 정책 노이즈 사용. 기본 권장 알고리즘 |
| **SAC** | `SACAlgorithm` | Soft Actor-Critic. 엔트로피 정규화, 탐색-활용 균형 |
| **PPO** | `PPOAlgorithm` | Proximal Policy Optimization. 클리핑 목적함수, 정책 안정성 |
| **DDPG** | `DDPGAlgorithm` | Deep Deterministic Policy Gradient. 결정론적 정책 |

**보상 함수** (`RewardFunction` Enum):
- `SHARPE_RATIO` — 샤프 비율 최대화 (기본)
- `SORTINO_RATIO` — 소르티노 비율
- `MAX_DRAWDOWN` — 최대 낙폭 최소화
- `RETURNS` — 절대 수익률

---

## 12. 의존성 핵심 라이브러리

```
mcp[cli]>=1.9.1     # Model Context Protocol
a2a-sdk>=0.2.4      # Agent-to-Agent Protocol
neo4j>=6.1.0        # 그래프 데이터베이스
fastapi             # REST API 서버
uvicorn             # ASGI 서버
networkx            # DAG 구현
torch               # RL 신경망
pandas, numpy       # 데이터 처리
langchain/langgraph # LLM 워크플로우
```

---

## 13. 문제 해결 (Troubleshooting)

| 증상 | 원인 | 해결 |
|------|------|------|
| `MCP server not available` | `mcp` 패키지 미설치 | `pip install mcp[cli]` |
| `Interface manager not initialized` | Neo4j 미연결 | Neo4j 실행 확인, `bolt://localhost:7687` 접속 |
| `Unknown agent_id` | 에이전트 미등록 | `preload_default_agents()` 호출 또는 `register://` 리소스로 등록 |
| A2A 연결 실패 | Memory A2A 서버 미실행 | `python FinAgents/memory/a2a_server.py --port 8010` |
| RL 학습 오류 | `torch` 미설치 | `pip install torch` |
