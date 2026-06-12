from FinAgents.agent_pools.backtest_agent.backtest_agent import BacktestAgent

# 1) Agent 생성
agent = BacktestAgent()

# 2) Qlib 초기화 (환경에 맞게 경로 수정)
init_res = agent.initialize_qlib_system(
    provider_uri="~/.qlib/qlib_data/us_data",  # 예시
    region="US",
    force_reinit=False
)
print("initialize_qlib_system:", init_res.get("status"), init_res)

# 3) 데이터 초기화/자산 스캔
data_res = agent.initialize_qlib_data(
    asset_symbol="AAPL",
    data_source="local"
)
print("initialize_qlib_data:", data_res.get("status"), "assets:", data_res.get("total_assets", len(data_res.get("available_assets", []))))

# 4) 알파 팩터 전략 생성
alpha_factors = {
    "asset": "AAPL",
    "factor_proposals": [
        {"factor_name": "momentum_20d", "factor_type": "momentum"},
        {"factor_name": "mean_reversion_5d", "factor_type": "reversion"},
        {"factor_name": "volume_spike", "factor_type": "volume"}
    ]
}

strategy_params = {
    "rebalancing_frequency": "daily",
    "position_sizing": "equal_weight",
    "max_positions": 10,
    "transaction_cost_rate": 0.001,
    "slippage_rate": 0.0005,
    "leverage": 1.0,
    "risk_budget": 0.02
}

strategy = agent.create_alpha_factor_strategy(
    alpha_factors=alpha_factors,
    strategy_params=strategy_params
)
strategy_id = strategy["strategy_id"]
print("strategy_id:", strategy_id)

# 5) 종합 백테스트 실행
bt = agent.run_comprehensive_backtest(
    strategy_id=strategy_id,
    start_date="2023-01-01",
    end_date="2023-12-31",
    benchmark="SPY"
)
print("run_comprehensive_backtest:", bt.get("status"))

# 6) 거래비용 분석
tc = agent.calculate_transaction_costs(
    strategy_id=strategy_id,
    cost_model="realistic"
)
print("transaction_costs total:", tc["cost_breakdown"]["total_transaction_costs"])

# 7) 팩터 성과 분석
fa = agent.analyze_factor_performance(strategy_id=strategy_id)
print("factor_analysis done, factors:", len(fa.get("individual_factors", {})))

# 8) 포트폴리오 분석
pa = agent.create_portfolio_analysis(
    strategy_id=strategy_id,
    analysis_type="comprehensive"
)
print("portfolio_analysis:", pa.get("status"))

# 9) 고급 리스크 지표
risk_adv = agent.calculate_advanced_risk_metrics(strategy_id=strategy_id)
print("advanced_risk:", risk_adv.get("status"), risk_adv.get("advanced_risk_metrics", {}))

# 10) 파라미터 최적화 (옵션)
opt = agent.optimize_strategy_parameters(
    strategy_id=strategy_id,
    optimization_type="sharpe"
)
print("opt best params:", opt.get("best_parameters"))

# 11) 워크포워드 분석 (옵션)
wf = agent.run_walk_forward_analysis(
    strategy_id=strategy_id,
    window_size=252,
    step_size=21,
    start_date="2020-01-01",
    end_date="2023-12-31"
)
print("walk_forward:", wf.get("status"))

# 12) 최종 리포트
report = agent.generate_detailed_report(
    strategy_id=strategy_id,
    include_charts=True
)

print("\n=== Final Summary ===")
if bt.get("status") == "success":
    pm = bt["results"]["performance_metrics"]
    print("Total Return :", pm.get("total_return"))
    print("Sharpe Ratio :", pm.get("sharpe_ratio"))
    print("Max Drawdown :", pm.get("max_drawdown"))
print("Report keys:", list(report.keys()))