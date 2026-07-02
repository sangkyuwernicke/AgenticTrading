---
name: agentic-rl-backtester
description: Run and manage PyTorch-based RL (TD3) training and evaluation with MCP-connected agent pools and LLM meta-prompt optimization.
---

# Agentic RL Backtester Skill

This skill allows the agent to run and manage the custom reinforcement learning (TD3) and backtesting pipeline for the KOSPI universe. It interacts with the external MCP agent pools (Port 8011 for data, etc.) and integrates LLM-based prompt optimization.

## Guidelines for the Agent

When a user requests to run training or backtesting, follow these operational rules:

### 1. Verification of Agent Pools
Before running the backtester script, verify that the external MCP agent pools are running (especially the data pool on Port 8011).
You can check port status or launch the helper script:
```bash
bash tests/start_agent_pools.sh
```

### 2. Running Training Mode
Use training mode to optimize model weights and prompt templates.
```bash
# Run training with N episodes (default 10)
python tests/test_agentic_rl_backtester.py --mode train --episodes <N>
```
* **Output files created/modified**:
  * Weights: `tests/checkpoint_rl.pt`
  * Prompts: `examples/optimized_prompts.json`

### 3. Running Backtest (Evaluation) Mode
Use backtest mode to evaluate model performance without modifying weights or prompts.
```bash
python tests/test_agentic_rl_backtester.py --mode backtest
```
* **Output generated**:
  * Accumulative reward statistics
  * Visual Performance Chart: `tests/agentic_rl_backtest_chart.png`

### 4. Interactive Reporting
After running a backtest, read the final statistics from the command output and tell the user the Initial/Final capital, return percentage, and total trades. Proactively link to the generated chart: [agentic_rl_backtest_chart.png](file:///Users/sangkyu/Work/tutorials/AgenticTrading/tests/agentic_rl_backtest_chart.png).
