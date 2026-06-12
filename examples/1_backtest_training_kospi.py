import os
import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path to import Orchestrator
project_root = Path("../").resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FinAgents" / "orchestrator_demo"))
sys.path.insert(0, str(project_root / "FinAgents" / "agent_pools"))

# Import Orchestrator
from FinAgents.orchestrator_demo.orchestrator import Orchestrator

# Initialize Orchestrator
orchestrator = Orchestrator()
print("✅ Orchestrator Initialized")

def train_agents_over_period(symbol, start_year, end_year):
    current_prompts = {
        "Alpha": orchestrator.alpha_agent.agent.instructions,
        "Risk": orchestrator.risk_agent.agent.instructions,
        "Portfolio": orchestrator.portfolio_agent.agent.instructions
    }
    
    performance_history = []
    
    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        print(f"\n--- Processing Year: {year} ---")
        
        # Run Pipeline (using the Legacy pipeline method for direct control, or agentic if preferred)
        # Here we use the underlying run_pipeline logic exposed in Orchestrator
        # Note: In a real scenario, we would capture the result object
        try:
            result = orchestrator.run_pipeline(symbol, start_date, end_date, mode="backtest")
            
            if result and result.get('status') == 'success':
                metrics = result.get('performance_metrics', {})
                sharpe = metrics.get('sharpe_ratio', 0.0)
                print(f"📊 Performance for {year}: Sharpe Ratio = {sharpe:.2f}")
                
                performance_history.append({'year': year, 'sharpe': sharpe})
                
                # Optimization Logic: If performance is poor, optimize prompts
                if sharpe < 1.0: # Threshold for optimization
                    print("⚠️ Performance below threshold. Optimizing prompts...")
                    
                    # Call the optimizer (Meta-Agent)
                    # In the demo, this calls OpenAI to rewrite instructions
                    new_instruction = orchestrator.optimize_agent_prompts(
                        agent_name="Alpha", 
                        performance_metric="Sharpe Ratio", 
                        current_value=sharpe, 
                        target_value=1.5
                    )
                    
                    if new_instruction and "Optimization failed" not in new_instruction:
                         current_prompts["Alpha"] = new_instruction
                         print("✅ Alpha Agent prompt updated.")
            else:
                print(f"❌ Backtest failed for {year}: {result.get('message') if result else 'Unknown error'}")
                
        except Exception as e:
            print(f"❌ Error during execution: {e}")
            
    return current_prompts, performance_history

# Run the Training Loop
# KOSPI Top 10 by market cap (Yahoo Finance tickers)
symbol = [
    '005930.KS',  # 삼성전자
    '000660.KS',  # SK하이닉스
    '373220.KS',  # LG에너지솔루션
    '207940.KS',  # 삼성바이오로직스
    '005380.KS',  # 현대차
    '000270.KS',  # 기아
    '005490.KS',  # POSCO홀딩스
    '006400.KS',  # 삼성SDI
    '105560.KS',  # KB금융
    '068270.KS',  # 셀트리온
]
optimized_prompts, history = train_agents_over_period(symbol, 2025, 2025)

output_path = "optimized_prompts.json"
with open(output_path, "w") as f:
    json.dump(optimized_prompts, f, indent=2)
    
print(f"💾 Optimized prompts saved to {output_path}")
print("History:", history)