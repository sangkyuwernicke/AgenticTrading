# Tests 실행 방법

## 사전 준비

### 환경 변수 설정
프로젝트 루트의 `.env` 파일에 다음 항목을 설정합니다.

```env
OPENAI_API_KEY=your_openai_api_key
```

---

## Agent Pool 서버 기동

테스트 실행 전, 5개의 Agent Pool 서버를 모두 기동해야 합니다.  
각각 별도 터미널에서 실행하거나 백그라운드(`&`)로 실행합니다.

```bash
cd /path/to/AgenticTrading

# Data Agent Pool (port 8001)
.venv/bin/python FinAgents/agent_pools/data_agent_pool/core.py

# Alpha Agent Pool (port 8081)
# ※ 반드시 -m 모듈 방식으로 실행 (상대 import 의존성)
.venv/bin/python -m FinAgents.agent_pools.alpha_agent_pool.core

# Portfolio Construction Agent Pool (port 8083)
# ※ 반드시 -m 모듈 방식으로 실행
.venv/bin/python -m FinAgents.agent_pools.portfolio_construction_agent_pool.core

# Risk Agent Pool (port 8084)
.venv/bin/python FinAgents/agent_pools/risk_agent_pool/core.py

# Transaction Cost Agent Pool (port 8085)
.venv/bin/python FinAgents/agent_pools/transaction_cost_agent_pool/core.py
```

---

## 서버 상태 확인

모든 서버가 정상 기동됐는지 확인합니다.

```bash
for port in 8001 8081 8083 8084 8085; do
  curl -s --max-time 2 http://localhost:$port/sse | head -1 \
    && echo "✅ $port UP" || echo "❌ $port DOWN"
done
```

5개 포트 모두 `✅ UP` 상태여야 합니다.

---

## 테스트 실행

```bash
cd /path/to/AgenticTrading
.venv/bin/python tests/test_simple_llm_backtest.py
```

### 정상 실행 시 예상 출력

```
✅ Agent pool health verification completed
✅ Retrieved market data: success
✅ Generated alpha signals: success
✅ Portfolio optimization: success
✅ Transaction cost analysis: success
✅ Risk management analysis: success
🎯 Orchestration Efficiency:
    Agent Pools Used: 5/5
    Successful Integrations: 5
    Data Quality: high
✅ Orchestrator-Based Backtest Completed Successfully!
```

---

## 주의사항

| 항목 | 설명 |
|------|------|
| Alpha / Portfolio Pool | 반드시 `-m` 모듈 방식으로 실행 (직접 실행 시 상대 import 오류 발생) |
| PolygonAgent | 미기동 시 synthetic data로 자동 fallback (테스트는 정상 통과) |
| OPENAI_API_KEY | 미설정 시 LLM 호출 불가, 일부 기능은 fallback으로 동작 |
