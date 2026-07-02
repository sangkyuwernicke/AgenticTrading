import asyncio
import pandas as pd
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from FinAgents.orchestrator.core.rl_policy_engine import RLPolicyEngine, RLConfiguration, RLAlgorithm, RewardFunction

async def test_rl_training():
    print("==================================================================")
    print("🚀 Starting RL Policy Engine Unit Training Test")
    print("==================================================================")

    # 1. Generate Mock Stock Market Data (3 symbols, 100 trading days)
    symbols = ["005930", "000660", "373220"]
    dates = pd.date_range(start="2023-01-01", periods=100)
    market_data = {}
    
    np.random.seed(42)  # For reproducibility
    for symbol in symbols:
        # Generate random walk stock prices
        prices = 100.0 + np.cumsum(np.random.normal(0, 1.5, 100))
        prices = np.clip(prices, 10.0, 1000.0)
        
        market_data[symbol] = pd.DataFrame({
            "close": prices,
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "volume": np.random.randint(1000, 10000, 100)
        }, index=dates)

    # 2. Configure RL settings
    config = RLConfiguration(
        algorithm=RLAlgorithm.TD3,
        reward_function=RewardFunction.SHARPE_RATIO,
        state_features=["returns", "volatility", "rsi", "macd"],
        action_space_dim=3,  # 3 stocks
        learning_rate=1e-3,
        batch_size=32,      # Smaller batch size for fast execution
        memory_size=1000,
        discount_factor=0.95
    )

    # 3. Initialize RL Policy Engine and Environment
    print("🧠 Initializing RL Policy Engine...")
    rl_engine = RLPolicyEngine(config)
    
    print("🌍 Creating Trading Environment...")
    rl_engine.create_environment("test_env", market_data)
    
    # State dimension = 8 features * 3 symbols = 24 dimensions
    print("🤖 Initializing TD3 Agent...")
    rl_engine.create_agent("test_agent", state_dim=24, action_dim=3)

    # 4. Run Training Loop
    print("\n⏳ Running RL training loop for 15 episodes...")
    training_results = await rl_engine.train_agent("test_agent", "test_env", episodes=15)
    
    # 5. Output Summary
    print("\n==================================================================")
    print("✅ RL Policy Engine Training completed successfully!")
    print("==================================================================")
    episode_rewards = training_results.get("episode_rewards", [])
    print(f"📊 Training Summary:")
    print(f"   • Total Episodes Trained: {len(episode_rewards)}")
    print(f"   • Initial Episode Reward: {episode_rewards[0]:.4f}" if episode_rewards else "N/A")
    print(f"   • Final Episode Reward  : {episode_rewards[-1]:.4f}" if episode_rewards else "N/A")
    print(f"   • Average Reward        : {np.mean(episode_rewards):.4f}" if episode_rewards else "N/A")
    
    loss_history = training_results.get("loss_history", [])
    if loss_history:
        print(f"   • Final Training Loss   : {loss_history[-1]:.6f}")
    print("==================================================================")

if __name__ == "__main__":
    # Run async training test
    asyncio.run(test_rl_training())
