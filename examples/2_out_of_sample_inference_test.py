import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path("../").resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FinAgents" / "orchestrator_demo"))
sys.path.insert(0, str(project_root / "FinAgents" / "agent_pools"))

from FinAgents.orchestrator_demo.orchestrator import Orchestrator

orchestrator = Orchestrator()
print("✅ Orchestrator Initialized")

prompts_path = "optimized_prompts.json"

if os.path.exists(prompts_path):
    with open(prompts_path, "r") as f:
        optimized_prompts = json.load(f)
    
    # Apply prompts to agents
    orchestrator.alpha_agent.agent.instructions = optimized_prompts.get("Alpha", "")
    orchestrator.risk_agent.agent.instructions = optimized_prompts.get("Risk", "")
    orchestrator.portfolio_agent.agent.instructions = optimized_prompts.get("Portfolio", "")
    
    print("✅ Optimized prompts loaded and applied.")
else:
    print("⚠️ Optimized prompts file not found. Using default instructions.")

symbol = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'WMT']
# start_date = "2024-09-01"
start_date = "2025-03-01"
end_date = "2025-05-31" # Or today's date

print(f"🚀 Running Out-of-Sample Test for {symbol} ({start_date} to {end_date})...")

# Run Event-Driven World Model Inference (Step-by-Step)
print("🌍 Starting World Model Inference (Day-by-Day Simulation)...")
result = orchestrator.run_inference_rolling_week(symbol, start_date, end_date)

if result and result.get('status') == 'success':
    metrics = result.get('performance_metrics', {})
    print("\n📊 Out-of-Sample Performance Results:")
    print(f"   Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
else:
    print("❌ Test failed.")

# Optional: Visualize if available
print("Result Details:", json.dumps(result, indent=2, default=str))