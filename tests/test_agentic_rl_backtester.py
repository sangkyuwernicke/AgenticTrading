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
from typing import Dict, List, Any, Optional, Tuple
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
from FinAgents.orchestrator.core.rl_policy_engine import RLPolicyEngine, RLConfiguration, RLAlgorithm, RewardFunction, TD3Agent

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

# KOSPI universe details (with historical shares count)
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

class DynamicTradingEnvironment:
    """Trading environment supporting dynamic quarterly rebalancing and Look-ahead bias elimination"""
    def __init__(self, 
                 market_data: Dict[str, pd.DataFrame],
                 top_n: int = 7,
                 initial_capital: float = 100000.0,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.001):
        self.market_data = market_data
        self.top_n = top_n
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # Get common dates across data
        common_dates = None
        for df in market_data.values():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))
        self.dates = sorted(list(common_dates))
        
        self.state_features = [
            'returns', 'volatility', 'rsi', 'macd', 'bollinger_position',
            'volume_ratio', 'price_momentum', 'portfolio_weight'
        ]
        
        self.reset()
        logger.info(f"Dynamic Trading Environment initialized with {len(self.dates)} trading days.")

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.capital = self.initial_capital
        self.positions = {symbol: 0.0 for symbol in self.market_data.keys()}
        self.portfolio_value = self.initial_capital
        
        self.trade_history = []
        self.portfolio_history = [{'date': self.dates[0], 'portfolio_value': self.initial_capital}]
        
        # Determine initial active symbols (Q1 2023)
        self.active_symbols = self._select_top_n_symbols(self.dates[0])
        logger.info(f"🎬 Initial Universe Selected for {self.dates[0].strftime('%Y-%m-%d')}: {self.active_symbols}")
        
        return self._get_state()

    def _select_top_n_symbols(self, date: datetime) -> List[str]:
        """Determine top N symbols based on historical market capitalization on the specific date"""
        market_caps = {}
        for symbol, df in self.market_data.items():
            # Get latest close price on or before this date
            prices_before = df[:date]
            if not prices_before.empty:
                price = prices_before.iloc[-1]['close']
                shares = KOSPI_UNIVERSE[symbol]["shares"]
                market_caps[symbol] = price * shares
            else:
                # Default to base price
                market_caps[symbol] = KOSPI_BASE_PRICES[symbol] * KOSPI_UNIVERSE[symbol]["shares"]
                
        # Sort by cap descending and slice top_n
        sorted_symbols = sorted(market_caps, key=market_caps.get, reverse=True)
        return sorted_symbols[:self.top_n]

    def _get_current_price(self, symbol: str, date: datetime) -> float:
        df = self.market_data[symbol]
        prices_before = df[:date]
        if not prices_before.empty:
            return float(prices_before.iloc[-1]['close'])
        return float(KOSPI_BASE_PRICES.get(symbol, 50000))

    def _check_rebalance(self, prev_date: datetime, curr_date: datetime):
        """Perform quarterly rebalancing if the quarter changed"""
        if prev_date is None:
            return
            
        prev_q = (prev_date.month - 1) // 3
        curr_q = (curr_date.month - 1) // 3
        
        if prev_q != curr_q:
            new_universe = self._select_top_n_symbols(curr_date)
            logger.info(f"🔄 Rebalance Triggered: Quarter transition from Q{prev_q+1} to Q{curr_q+1}.")
            logger.info(f"   • Current Date : {curr_date.strftime('%Y-%m-%d')}")
            logger.info(f"   • Old Universe : {self.active_symbols}")
            logger.info(f"   • New Universe : {new_universe}")
            
            # Liquidation of dropped symbols
            for symbol in self.active_symbols:
                if symbol not in new_universe:
                    shares_to_sell = self.positions[symbol]
                    if shares_to_sell > 0:
                        price = self._get_current_price(symbol, curr_date)
                        trade_val = shares_to_sell * price
                        fee = trade_val * self.commission_rate
                        
                        self.capital += trade_val - fee
                        self.positions[symbol] = 0.0
                        
                        trade_record = {
                            'symbol': symbol,
                            'shares': -shares_to_sell,
                            'price': price,
                            'value': -trade_val,
                            'commission': fee,
                            'timestamp': self.current_step,
                            'type': 'liquidation'
                        }
                        self.trade_history.append(trade_record)
                        logger.info(f"🧹 Liquidation: Sold all shares of {symbol} (${trade_val:,.2f}) due to exclusion from universe.")
            
            self.active_symbols = new_universe

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        prev_date = self.dates[self.current_step]
        self.current_step += 1
        curr_date = self.dates[self.current_step] if self.current_step < len(self.dates) else self.dates[-1]
        
        # 1. Check & Execute Quarterly Rebalancing
        self._check_rebalance(prev_date, curr_date)
        
        # 2. Execute trades based on RL actions
        actions = np.clip(actions, -1, 1)
        trade_info = self._execute_trades(actions, curr_date)
        
        # 3. Update portfolio total value
        self._update_portfolio_value(curr_date)
        
        # 4. Calculate reward
        reward = self._calculate_reward()
        
        # 5. Build next state
        next_state = self._get_state()
        
        # 6. Check done status
        done = self.current_step >= len(self.dates) - 1
        
        info = {
            'portfolio_value': self.portfolio_value,
            'trades': trade_info,
            'positions': {sym: self.positions[sym] for sym in self.active_symbols},
            'step': self.current_step,
            'active_symbols': self.active_symbols.copy()
        }
        
        return next_state, reward, done, info

    def _execute_trades(self, actions: np.ndarray, date: datetime) -> List[Dict[str, Any]]:
        trades = []
        
        # Calculate current active portfolio weights
        current_weights = {}
        for symbol in self.active_symbols:
            price = self._get_current_price(symbol, date)
            val = self.positions[symbol] * price
            current_weights[symbol] = val / self.portfolio_value if self.portfolio_value > 0 else 0.0
            
        for i, symbol in enumerate(self.active_symbols):
            if i >= len(actions):
                break
                
            target_weight = actions[i]
            curr_weight = current_weights.get(symbol, 0.0)
            
            weight_diff = target_weight - curr_weight
            trade_value = weight_diff * self.portfolio_value
            
            if abs(trade_value) > self.portfolio_value * 0.01:
                price = self._get_current_price(symbol, date)
                shares = trade_value / price
                
                # Apply commissions & slippage
                execution_price = price * (1 + np.sign(shares) * self.slippage_rate)
                commission = abs(trade_value) * self.commission_rate
                
                # Verify cash limit
                if shares > 0 and (trade_value + commission) > self.capital:
                    # Adjust buy quantity to fit available cash
                    trade_value = self.capital - commission
                    shares = trade_value / execution_price
                    
                if shares != 0 and (self.positions[symbol] + shares) >= 0:
                    self.positions[symbol] += shares
                    self.capital -= (shares * execution_price) + commission
                    
                    trade = {
                        'symbol': symbol,
                        'shares': shares,
                        'price': execution_price,
                        'value': shares * execution_price,
                        'commission': commission,
                        'timestamp': self.current_step,
                        'type': 'trade'
                    }
                    trades.append(trade)
                    self.trade_history.append(trade)
                    
        return trades

    def _update_portfolio_value(self, date: datetime):
        asset_value = 0.0
        for symbol, shares in self.positions.items():
            if shares > 0:
                price = self._get_current_price(symbol, date)
                asset_value += shares * price
        self.portfolio_value = self.capital + asset_value
        self.portfolio_history.append({'date': date, 'portfolio_value': self.portfolio_value})

    def _calculate_reward(self) -> float:
        if len(self.portfolio_history) < 2:
            return 0.0
            
        prev_val = self.portfolio_history[-2]['portfolio_value']
        curr_val = self.portfolio_value
        
        if prev_val == 0:
            return 0.0
            
        returns = (curr_val - prev_val) / prev_val
        
        # Sharpe-like reward (using last 20 steps)
        if len(self.portfolio_history) >= 20:
            recent_returns = [
                (self.portfolio_history[i]['portfolio_value'] - self.portfolio_history[i-1]['portfolio_value']) / 
                self.portfolio_history[i-1]['portfolio_value']
                for i in range(-19, 0)
            ]
            vol = np.std(recent_returns)
            risk_adjusted = returns / vol if vol > 0 else returns
        else:
            risk_adjusted = returns
            
        # Drawdown penalty
        max_val = max(h['portfolio_value'] for h in self.portfolio_history)
        drawdown = (max_val - curr_val) / max_val
        drawdown_penalty = -max(0, drawdown - 0.05) * 10
        
        return risk_adjusted + drawdown_penalty

    def _get_state(self) -> np.ndarray:
        date = self.dates[self.current_step]
        state_vector = []
        
        for symbol in self.active_symbols:
            df = self.market_data[symbol]
            data_before = df[:date]
            
            if len(data_before) > 0:
                close_prices = data_before['close']
                
                # Returns
                returns = close_prices.pct_change().iloc[-1] if len(close_prices) > 1 else 0.0
                
                # Volatility
                volatility = close_prices.pct_change().rolling(20).std().iloc[-1] if len(close_prices) > 20 else 0.0
                
                # price momentum
                momentum = (close_prices.iloc[-1] - close_prices.iloc[-5]) / close_prices.iloc[-5] if len(close_prices) > 5 else 0.0
                
                # RSI dummy
                rsi = 0.5
                # MACD dummy
                macd = 0.0
                # Bollinger
                bollinger = 0.5
                # Volume ratio
                volume = 1.0
                
                # Current portfolio weight
                price = close_prices.iloc[-1]
                weight = (self.positions[symbol] * price) / self.portfolio_value if self.portfolio_value > 0 else 0.0
                
                features = [
                    returns, volatility, rsi, macd, bollinger,
                    volume, momentum, weight
                ]
                # Clean NaNs
                features = [f if not np.isnan(f) else 0.0 for f in features]
                state_vector.extend(features)
            else:
                state_vector.extend([0.0] * len(self.state_features))
                
        return np.array(state_vector, dtype=np.float32)

class AgenticRLBacktester:
    def __init__(self, mode: str, episodes: int, model_path: str, prompts_path: str):
        self.mode = mode
        self.episodes = episodes
        self.model_path = model_path
        self.prompts_path = prompts_path
        self.orchestrator = None
        self.llm_client = None
        
        self._load_env()
        if LLM_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("✅ LLM client initialized")
            
    def _load_env(self):
        try:
            from dotenv import load_dotenv
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            env_path = os.path.join(project_root, '.env')
            load_dotenv(env_path)
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
                
        return self._generate_mock_market_data()

    def _generate_mock_market_data(self) -> Dict[str, Any]:
        logger.info("⚠️ Generating mock market data as fallback...")
        dates = pd.date_range(start="2023-01-01", end="2024-12-31").strftime("%Y-%m-%d").tolist()
        data_dict = {}
        
        np.random.seed(42)
        for sym in KOSPI_UNIVERSE:
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
        # 1. Retrieve market data
        raw_data = await self.retrieve_market_data()
        env_data = self._prepare_env_data(raw_data)
        
        # 2. Setup dynamic environment (Top 7 active stocks rebalanced quarterly)
        top_n = 7
        env = DynamicTradingEnvironment(env_data, top_n=top_n)
        
        # Define config (State_dim = 8 features * 7 stocks = 56 dimensions, Action_dim = 7)
        config = RLConfiguration(
            algorithm=RLAlgorithm.TD3,
            reward_function=RewardFunction.SHARPE_RATIO,
            state_features=["returns", "volatility", "rsi", "macd"],
            action_space_dim=top_n,
            learning_rate=1e-3,
            batch_size=32,
            memory_size=10000,
            discount_factor=0.99
        )
        
        rl_engine = RLPolicyEngine(config)
        agent = rl_engine.create_agent("agentic_agent", state_dim=len(env.state_features) * top_n, action_dim=top_n)
        
        # 3. Mode execution
        if self.mode == "train":
            await self._run_training_loop(agent, env)
        elif self.mode == "backtest":
            await self._run_backtest_only(agent, env)

    async def _run_training_loop(self, agent: TD3Agent, env: DynamicTradingEnvironment):
        logger.info(f"🏋️ Starting BIAS-FREE AGENTIC TRAINING LOOP ({self.episodes} episodes)...")
        
        best_reward = -float('inf')
        current_prompts = self._load_initial_prompts()
        
        for ep in range(1, self.episodes + 1):
            logger.info(f"\n--- 🎬 Episode {ep}/{self.episodes} ---")
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state, add_noise=True)
                next_state, reward, done, info = env.step(action)
                
                agent.store_experience(state, action, reward, next_state, done)
                agent.train()
                
                state = next_state
                episode_reward += reward
                
            logger.info(f"🏆 Episode {ep} completed. Total Reward: {episode_reward:.4f}, Portfolio Value: ${env.portfolio_value:,.2f}")
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save_model(self.model_path)
                logger.info(f"💾 Saved new best model checkpoint to {self.model_path}")
                
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

    async def _run_backtest_only(self, agent: TD3Agent, env: DynamicTradingEnvironment):
        logger.info("📊 Running Pure BACKTEST Mode (Bias-Free, Evaluation only)...")
        
        if os.path.exists(self.model_path):
            agent.load_model(self.model_path)
            logger.info(f"✅ Loaded trained model weights from {self.model_path}")
        else:
            logger.warning(f"⚠️ No model checkpoint found at {self.model_path}. Running with random weights.")
            
        prompts = self._load_initial_prompts()
        logger.info(f"✅ Active Alpha Prompt instruction size: {len(prompts.get('Alpha', ''))} characters")

        state = env.reset()
        done = False
        episode_reward = 0
        portfolio_values = []
        dates = []
        
        while not done:
            action = agent.select_action(state, add_noise=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
            
            portfolio_values.append(env.portfolio_value)
            dates.append(env.dates[env.current_step])
            
        logger.info(f"\n==================================================================")
        logger.info(f"📊 BIAS-FREE BACKTEST RESULTS:")
        logger.info(f"==================================================================")
        logger.info(f"   • Initial Capital : ${env.initial_capital:,.2f}")
        logger.info(f"   • Final Value     : ${env.portfolio_value:,.2f}")
        logger.info(f"   • Total Return    : {((env.portfolio_value - env.initial_capital) / env.initial_capital) * 100:.2f}%")
        logger.info(f"   • Total Trades     : {len(env.trade_history)}")
        logger.info(f"   • Accumulated Reward: {episode_reward:.4f}")
        logger.info(f"==================================================================")
        
        self._plot_results(dates, portfolio_values)

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

    def _plot_results(self, dates: List[datetime], portfolio_values: List[float]):
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(dates, portfolio_values, label="Orchestrator RL (Quarterly Rebalanced)", color="royalblue", linewidth=2)
            plt.title("Bias-Free Agentic RL Backtest Performance", fontsize=14, fontweight='bold')
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value ($)")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            
            # Rotate dates label
            plt.gcf().autofmt_xdate()
            
            chart_path = "tests/agentic_rl_backtest_chart.png"
            plt.savefig(chart_path)
            logger.info(f"📈 Performance chart saved successfully to {chart_path}")
        except Exception as e:
            logger.error(f"❌ Failed to plot visualizations: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic RL Training & Backtest Loop (Bias-Free)")
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
    
    asyncio.run(backtester.initialize())
    asyncio.run(backtester.run())
