import argparse
import asyncio
import logging
import sys
import os
import numpy as np
import pandas as pd
import random
import json
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AgenticRLBacktester")

# Import FinAgent components
from FinAgents.orchestrator.core.finagent_orchestrator import FinAgentOrchestrator
from FinAgents.orchestrator.core.rl_policy_engine import RLPolicyEngine, RLConfiguration, RLAlgorithm, RewardFunction, TD3Agent, TradingEnvironment

# Import MCP Client dependencies
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ MCP client not available. Will use fallback data generation.")
    MCP_AVAILABLE = False

try:
    from openai import AsyncOpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    AsyncOpenAI = None

# KOSPI universe details
KOSPI_UNIVERSE = {
    "005930": {"name": "삼성전자",         "shares": 5_969_782_550},
    "373220": {"name": "LG에너지솔루션",   "shares":   234_000_000},
    "000660": {"name": "SK하이닉스",       "shares":   728_002_365},
    "207940": {"name": "삼성바이오로직스", "shares":    71_174_000},
    "005935": {"name": "삼성전자우",       "shares":   822_886_700},
    "005380": {"name": "현대차",           "shares":   213_668_187},
    "035420": {"name": "NAVER",            "shares":   164_263_395},
    "006400": {"name": "삼성SDI",          "shares":    66_978_000},
    "035720": {"name": "카카오",           "shares":   887_967_808},
    "000270": {"name": "기아",             "shares":   399_429_050},
}
KOSPI_BASE_PRICES = {
    "005930": 52000,   "373220": 446000, "000660": 74000,
    "207940": 772000,  "005935": 47000,  "005380": 136000,
    "035420": 176000,  "006400": 598000, "035720": 52500,
    "000270": 53000,
}

class AgenticRLBacktester:
    def __init__(self, mode: str, episodes: int, model_path: str, prompts_path: str):
        self.mode = mode
        self.episodes = episodes
        self.model_path = model_path
        self.prompts_path = prompts_path
        self.orchestrator = None
        self.rl_engine = None
        self.llm_client = None
        
        # Load environment variables
        self._load_env()
        
        # Initialize OpenAI Client
        if LLM_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("✅ LLM client initialized")
            
    def _load_env(self):
        try:
            from dotenv import load_dotenv
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            env_path = os.path.join(project_root, '.env')
            load_dotenv(env_path)
            logger.info(f"✅ Loaded .env from: {env_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load .env: {e}")

    async def initialize(self):
        logger.info("🔧 Initializing Orchestrator & Connection to MCP Pools...")
        self.orchestrator = FinAgentOrchestrator(
            host="localhost",
            port=9000,
            enable_memory=True,
            enable_rl=True,
            enable_monitoring=True
        )
        await self.orchestrator.initialize()
        logger.info("✅ Orchestrator initialized")

    async def retrieve_market_data(self) -> Dict[str, Any]:
        """Fetch market data via Data Agent Pool MCP Server (Port 8011)"""
        logger.info("📊 Fetching market data via Data Agent Pool (Port 8011)...")
        if MCP_AVAILABLE:
            data_pool_url = "http://localhost:8011/sse"
            symbols_str = ", ".join(list(KOSPI_UNIVERSE.keys()))
            query = f"Get daily price data for KOSPI universe stocks ({symbols_str}) from 2023-01-01 to 2024-12-31"

            try:
                async with sse_client(data_pool_url, timeout=60) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool("process_market_query", {"query": query})
                        
                        if result.content and len(result.content) > 0:
                            content_item = result.content[0]
                            if hasattr(content_item, 'text'):
                                data = json.loads(content_item.text)
                                logger.info("✅ Successfully retrieved data from MCP")
                                return data
            except Exception as e:
                logger.error(f"❌ Failed to query Data Agent Pool MCP: {e}. Falling back to mock data.")
                
        # Fallback to generating mock data
        return self._generate_mock_market_data()

    def _generate_mock_market_data(self) -> Dict[str, Any]:
        logger.info("⚠️ Generating mock market data as fallback...")
        dates = pd.date_range(start="2023-01-01", end="2024-12-31").strftime("%Y-%m-%d").tolist()
        data_dict = {}
        
        np.random.seed(42)
        for sym, info in KOSPI_UNIVERSE.items():
            base_price = KOSPI_BASE_PRICES.get(sym, 50000)
            prices = base_price + np.cumsum(np.random.normal(0, base_price * 0.015, len(dates)))
            prices = np.clip(prices, base_price * 0.1, base_price * 10.0)
            
            data_dict[sym] = [
                {"date": d, "close": float(p), "open": float(p * 0.99), "high": float(p * 1.01), "low": float(p * 0.98), "volume": 10000.0}
                for d, p in zip(dates, prices)
            ]
            
        return {
            "status": "success",
            "source": "yfinance",
            "data": data_dict
        }

    def _prepare_env_data(self, market_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Convert MCP/Mock market data format to Dict[str, pd.DataFrame]"""
        logger.info("📦 Reformatting market data for TradingEnvironment...")
        env_market_data = {}
        data_dict = market_data.get("data", {})
        
        for symbol, points in data_dict.items():
            df = pd.DataFrame(points)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            env_market_data[symbol] = df
            
        return env_market_data

    async def run(self):
        # 1. Retrieve market data via MCP
        raw_data = await self.retrieve_market_data()
        env_data = self._prepare_env_data(raw_data)
        
        # 2. Define RL configurations
        symbols = list(env_data.keys())
        config = RLConfiguration(
            algorithm=RLAlgorithm.TD3,
            reward_function=RewardFunction.SHARPE_RATIO,
            state_features=["returns", "volatility", "rsi", "macd"],
            action_space_dim=len(symbols),
            learning_rate=1e-3,
            batch_size=32,
            memory_size=10000,
            discount_factor=0.99
        )
        
        self.rl_engine = RLPolicyEngine(config)
        env = self.rl_engine.create_environment("agentic_env", env_data)
        
        # Dim = 8 features * symbols count
        state_dim = len(env.state_features) * len(symbols)
        agent = self.rl_engine.create_agent("agentic_agent", state_dim=state_dim, action_dim=len(symbols))
        
        # 3. Determine Execution Mode
        if self.mode == "train":
            await self._run_training_loop(agent, env)
        elif self.mode == "backtest":
            await self._run_backtest_only(agent, env)

    async def _run_training_loop(self, agent: TD3Agent, env: TradingEnvironment):
        logger.info(f"🏋️ Starting AGENTIC TRAINING LOOP ({self.episodes} episodes)...")
        
        best_reward = -float('inf')
        current_prompts = self._load_initial_prompts()
        
        for ep in range(1, self.episodes + 1):
            logger.info(f"\n--- 🎬 Episode {ep}/{self.episodes} ---")
            state = env.reset()
            episode_reward = 0
            done = False
            
            # Step loop (Day-by-day simulation)
            while not done:
                action = agent.select_action(state, add_noise=True)
                next_state, reward, done, info = env.step(action)
                
                agent.store_experience(state, action, reward, next_state, done)
                agent.train()
                
                state = next_state
                episode_reward += reward
                
            logger.info(f"🏆 Episode {ep} completed. Total Reward: {episode_reward:.4f}, Portfolio Value: ${env.portfolio_value:,.2f}")
            
            # Save the best model weights
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save_model(self.model_path)
                logger.info(f"💾 Saved new best model checkpoint to {self.model_path}")
                
            # --- Agentic Loop: LLM Prompt Optimization & Metaparameter feedback ---
            # Optimize prompt when rewards are poor (e.g. below target) or at specific checkpoints
            if LLM_AVAILABLE and self.llm_client and (episode_reward < 0.0 or ep % 5 == 0):
                logger.info("⚠️ Optimizing prompts via LLM Meta-Optimizer...")
                optimized_instruction = await self._optimize_instructions_via_llm(
                    current_prompts.get("Alpha", ""),
                    episode_reward,
                    env.portfolio_value
                )
                if optimized_instruction:
                    current_prompts["Alpha"] = optimized_instruction
                    self._save_prompts(current_prompts)
                    logger.info(f"💾 Optimized prompts saved to {self.prompts_path}")

        logger.info("\n🎉 Agentic RL Training Loop finished!")

    async def _run_backtest_only(self, agent: TD3Agent, env: TradingEnvironment):
        logger.info("📊 Running Pure BACKTEST Mode (No training, evaluation only)...")
        
        # Load weights
        if os.path.exists(self.model_path):
            agent.load_model(self.model_path)
            logger.info(f"✅ Loaded trained model weights from {self.model_path}")
        else:
            logger.warning(f"⚠️ No model checkpoint found at {self.model_path}. Running with random weights.")
            
        # Load optimized prompts
        prompts = self._load_initial_prompts()
        logger.info(f"✅ Active Alpha Prompt instruction size: {len(prompts.get('Alpha', ''))} characters")

        # Run 1 evaluation episode
        state = env.reset()
        done = False
        episode_reward = 0
        portfolio_values = []
        dates = []
        
        while not done:
            action = agent.select_action(state, add_noise=False) # No noise in backtest
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
            
            portfolio_values.append(env.portfolio_value)
            dates.append(env.current_step)
            
        logger.info(f"\n==================================================================")
        logger.info(f"📊 BACKTEST RESULTS:")
        logger.info(f"==================================================================")
        logger.info(f"   • Initial Capital : ${env.initial_capital:,.2f}")
        logger.info(f"   • Final Value     : ${env.portfolio_value:,.2f}")
        logger.info(f"   • Total Return    : {((env.portfolio_value - env.initial_capital) / env.initial_capital) * 100:.2f}%")
        logger.info(f"   • Total Trades     : {len(env.trade_history)}")
        logger.info(f"   • Accumulated Reward: {episode_reward:.4f}")
        logger.info(f"==================================================================")
        
        # Generate chart
        self._plot_results(portfolio_values)

    async def _optimize_instructions_via_llm(self, current_instruction: str, reward: float, final_val: float) -> Optional[str]:
        meta_prompt = f"""
        You are a Meta-Agent optimizing trading instructions.
        The current Trading Agent underperformed during training.
        Performance Metrics:
        - Episode Reward: {reward:.4f}
        - Final Portfolio Value: ${final_val:,.2f} (Initial: $100,000)
        
        Current Instructions:
        {current_instruction}
        
        Please rewrite the instructions to improve the agent's strategy.
        Focus on:
        1. Safer entry/exit criteria based on RSI and MACD.
        2. Stricter drawdown controls to prevent large negative rewards.
        3. Adaptive position sizing (more conservative under high volatility).
        
        Return ONLY the rewritten instruction text, without markdown formatting or headers.
        """
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": meta_prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ Failed to run LLM Prompt Optimization: {e}")
            return None

    def _load_initial_prompts(self) -> Dict[str, str]:
        if os.path.exists(self.prompts_path):
            try:
                with open(self.prompts_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load prompts from {self.prompts_path}: {e}")
        
        # Default prompt templates
        return {
            "Alpha": """
            You are an Alpha Signal Agent. Analyze market data carefully.
            Generate buy/sell signals based on MACD and RSI indicators.
            Use conservative volume targets when volatility is high to prevent drawdowns.
            """,
            "Risk": "Monitor drawdown limits and apply position sizing cuts when overall volatility exceeds 2.5%."
        }

    def _save_prompts(self, prompts: Dict[str, str]):
        try:
            with open(self.prompts_path, "w") as f:
                json.dump(prompts, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save prompts to {self.prompts_path}: {e}")

    def _plot_results(self, portfolio_values: List[float]):
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(portfolio_values, label="Orchestrator RL Portfolio", color="royalblue", linewidth=2)
            plt.title("Agentic RL Backtest Performance", fontsize=14, fontweight='bold')
            plt.xlabel("Trading Steps (Days)")
            plt.ylabel("Portfolio Value ($)")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            
            # Save chart to disk
            chart_path = "tests/agentic_rl_backtest_chart.png"
            plt.savefig(chart_path)
            logger.info(f"📈 Performance chart saved successfully to {chart_path}")
        except Exception as e:
            logger.error(f"❌ Failed to plot visualizations: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic RL Training & Backtest Loop")
    parser.add_argument("--mode", type=str, default="backtest", choices=["train", "backtest"], help="Running mode")
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes")
    parser.add_argument("--model-path", type=str, default="tests/checkpoint_rl.pt", help="Path to RL model weights")
    parser.add_argument("--prompts-path", type=str, default="examples/optimized_prompts.json", help="Path to prompts config")
    args = parser.parse_args()

    backtester = AgenticRLBacktester(
        mode=args.mode,
        episodes=args.episodes,
        model_path=args.model_path,
        prompts_path=args.prompts_path
    )
    
    # Run async main loop
    asyncio.run(backtester.initialize())
    asyncio.run(backtester.run())
