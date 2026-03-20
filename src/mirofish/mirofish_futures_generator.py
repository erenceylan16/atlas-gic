"""
MiroFish Futures Generator
==========================

Generates synthetic future price paths based on swarm simulation outcomes.
These price paths are used to forward-train ATLAS agents by showing them
"what could happen" and evaluating their decisions.

Key features:
1. Multiple scenario branches (base, bull, bear, tail)
2. Price paths incorporate swarm consensus + uncertainty
3. Realistic volatility and correlation structure
4. Event injection (earnings, FOMC, etc.)
5. Outputs formatted for agent training loops

Usage:
    python -m agents.mirofish_futures_generator --days 30 --scenarios 5 --print
"""

import os
import sys
import json
import logging
import random
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# Directories
STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "state"
MIROFISH_DIR = Path(__file__).resolve().parent.parent / "data" / "mirofish"
FUTURES_DIR = MIROFISH_DIR / "futures"

PREDICTIONS_FILE = STATE_DIR / "mirofish_predictions.json"


@dataclass
class PricePath:
    """A single price path for an asset."""
    ticker: str
    start_price: float
    prices: List[float]
    dates: List[str]
    returns: List[float]
    cumulative_return: float
    volatility: float
    scenario: str


@dataclass
class FutureScenario:
    """A complete future scenario with multiple asset paths."""
    scenario_id: str
    scenario_name: str
    scenario_type: str  # base, bull, bear, tail_up, tail_down
    probability: float
    description: str
    price_paths: Dict[str, PricePath]
    events: List[Dict]  # injected events
    final_state: Dict  # market state at end


# Asset characteristics
ASSET_PARAMS = {
    "SPY": {"vol": 0.18, "drift": 0.08, "name": "S&P 500"},
    "QQQ": {"vol": 0.25, "drift": 0.10, "name": "Nasdaq 100"},
    "TLT": {"vol": 0.15, "drift": 0.03, "name": "20Y Treasury"},
    "GLD": {"vol": 0.15, "drift": 0.02, "name": "Gold"},
    "XLE": {"vol": 0.28, "drift": 0.05, "name": "Energy"},
    "VXX": {"vol": 0.80, "drift": -0.30, "name": "VIX"},  # Negative drift (contango)
    "HYG": {"vol": 0.10, "drift": 0.05, "name": "High Yield"},
}

# Correlation matrix (simplified)
CORRELATIONS = {
    ("SPY", "QQQ"): 0.90,
    ("SPY", "TLT"): -0.30,
    ("SPY", "GLD"): 0.05,
    ("SPY", "XLE"): 0.70,
    ("SPY", "VXX"): -0.80,
    ("SPY", "HYG"): 0.60,
    ("QQQ", "TLT"): -0.25,
    ("QQQ", "GLD"): 0.00,
    ("QQQ", "VXX"): -0.75,
    ("TLT", "GLD"): 0.30,
    ("TLT", "VXX"): 0.20,
    ("GLD", "VXX"): 0.15,
}


class FuturesGenerator:
    """Generates synthetic future scenarios from swarm predictions."""

    def __init__(self):
        self.rng = np.random.default_rng()
        FUTURES_DIR.mkdir(parents=True, exist_ok=True)

    def load_predictions(self) -> Optional[Dict]:
        """Load latest MiroFish predictions."""
        if not PREDICTIONS_FILE.exists():
            logger.warning("No predictions file found")
            return None

        with open(PREDICTIONS_FILE) as f:
            predictions = json.load(f)

        if not predictions:
            return None

        return predictions[-1]  # Latest

    def load_current_prices(self) -> Dict[str, float]:
        """Load current prices from positions or use defaults."""
        # Try to load from positions
        positions_file = STATE_DIR / "positions.json"
        if positions_file.exists():
            with open(positions_file) as f:
                data = json.load(f)
            # Extract current prices from positions
            prices = {}
            for pos in data.get("positions", []):
                ticker = pos.get("ticker")
                price = pos.get("current_price")
                if ticker and price:
                    prices[ticker] = price
            if prices:
                return prices

        # Default prices (March 2026 estimates)
        return {
            "SPY": 673.54,
            "QQQ": 604.57,
            "TLT": 87.43,
            "GLD": 461.61,
            "XLE": 58.14,
            "VXX": 31.56,
            "HYG": 79.74,
        }

    def generate_correlated_returns(
        self,
        tickers: List[str],
        num_days: int,
        scenario_adjustments: Dict[str, float] = None
    ) -> Dict[str, np.ndarray]:
        """Generate correlated daily returns for multiple assets."""
        n = len(tickers)
        scenario_adjustments = scenario_adjustments or {}

        # Build correlation matrix
        corr_matrix = np.eye(n)
        for i, t1 in enumerate(tickers):
            for j, t2 in enumerate(tickers):
                if i != j:
                    key = (t1, t2) if (t1, t2) in CORRELATIONS else (t2, t1)
                    corr_matrix[i, j] = CORRELATIONS.get(key, 0.0)

        # Cholesky decomposition for correlated normals
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            # Matrix not positive definite, use SVD
            U, s, Vt = np.linalg.svd(corr_matrix)
            s = np.maximum(s, 1e-10)
            corr_matrix = U @ np.diag(s) @ Vt
            L = np.linalg.cholesky(corr_matrix)

        # Generate uncorrelated normals
        Z = self.rng.standard_normal((n, num_days))

        # Apply correlation
        correlated = L @ Z

        # Convert to returns
        returns = {}
        for i, ticker in enumerate(tickers):
            params = ASSET_PARAMS.get(ticker, {"vol": 0.20, "drift": 0.05})
            daily_vol = params["vol"] / np.sqrt(252)
            daily_drift = params["drift"] / 252

            # Apply scenario adjustment
            adj = scenario_adjustments.get(ticker, 0.0)
            adjusted_drift = daily_drift + adj / num_days

            returns[ticker] = adjusted_drift + daily_vol * correlated[i]

        return returns

    def generate_price_path(
        self,
        ticker: str,
        start_price: float,
        returns: np.ndarray,
        start_date: datetime,
        scenario: str
    ) -> PricePath:
        """Generate a price path from returns."""
        prices = [start_price]
        dates = [start_date.strftime("%Y-%m-%d")]

        current = start_price
        for i, ret in enumerate(returns):
            current = current * (1 + ret)
            prices.append(current)
            day = start_date + timedelta(days=i + 1)
            dates.append(day.strftime("%Y-%m-%d"))

        return PricePath(
            ticker=ticker,
            start_price=start_price,
            prices=prices,
            dates=dates,
            returns=returns.tolist(),
            cumulative_return=(prices[-1] / prices[0] - 1),
            volatility=float(np.std(returns) * np.sqrt(252)),
            scenario=scenario,
        )

    def generate_scenario(
        self,
        predictions: Dict,
        current_prices: Dict[str, float],
        scenario_type: str,
        num_days: int = 30
    ) -> FutureScenario:
        """Generate a complete scenario with price paths."""
        today = datetime.now()

        # Determine scenario adjustments based on type and predictions
        adjustments = self._calculate_adjustments(predictions, scenario_type)

        # Get tickers
        tickers = list(current_prices.keys())

        # Generate returns
        returns = self.generate_correlated_returns(tickers, num_days, adjustments)

        # Generate price paths
        paths = {}
        for ticker in tickers:
            if ticker in returns:
                paths[ticker] = self.generate_price_path(
                    ticker,
                    current_prices.get(ticker, 100.0),
                    returns[ticker],
                    today,
                    scenario_type
                )

        # Generate events
        events = self._generate_events(scenario_type, num_days, today)

        # Calculate final state
        final_state = self._calculate_final_state(paths, events)

        # Scenario metadata
        scenario_names = {
            "base": "Base Case - Consensus Path",
            "bull": "Bull Case - Risk-On Rally",
            "bear": "Bear Case - Risk-Off Correction",
            "tail_up": "Tail Risk - Melt-Up",
            "tail_down": "Tail Risk - Crash",
        }

        scenario_probs = {
            "base": 0.50,
            "bull": 0.20,
            "bear": 0.20,
            "tail_up": 0.05,
            "tail_down": 0.05,
        }

        scenario_desc = self._generate_scenario_description(
            scenario_type, predictions, paths
        )

        return FutureScenario(
            scenario_id=f"scen_{scenario_type}_{today.strftime('%Y%m%d_%H%M%S')}",
            scenario_name=scenario_names.get(scenario_type, scenario_type),
            scenario_type=scenario_type,
            probability=scenario_probs.get(scenario_type, 0.10),
            description=scenario_desc,
            price_paths={t: asdict(p) for t, p in paths.items()},
            events=events,
            final_state=final_state,
        )

    def _calculate_adjustments(
        self,
        predictions: Dict,
        scenario_type: str
    ) -> Dict[str, float]:
        """Calculate drift adjustments based on predictions and scenario."""
        adjustments = {}

        # Get consensus predictions
        consensus = predictions.get("consensus_predictions", [])
        hct = predictions.get("highest_conviction_trade", {})

        # Base adjustments from predictions
        for pred in consensus:
            ticker = None
            # Map event to ticker
            if "VXX" in pred.get("prediction", ""):
                ticker = "VXX"
            elif "GLD" in pred.get("prediction", ""):
                ticker = "GLD"
            elif "S&P" in pred.get("event", ""):
                ticker = "SPY"
            elif "Tech" in pred.get("event", ""):
                ticker = "QQQ"

            if ticker:
                confidence = pred.get("confidence", 0.5)
                direction = 1 if "rise" in pred.get("prediction", "").lower() else -1
                adjustments[ticker] = direction * confidence * 0.10  # 10% max adjustment

        # Apply scenario multipliers
        multipliers = {
            "base": 1.0,
            "bull": 1.5,
            "bear": -1.2,
            "tail_up": 2.5,
            "tail_down": -2.0,
        }

        mult = multipliers.get(scenario_type, 1.0)

        # Modify adjustments
        for ticker in list(adjustments.keys()):
            if scenario_type in ["bear", "tail_down"]:
                # Invert equity and risk assets
                if ticker in ["SPY", "QQQ", "XLE", "HYG"]:
                    adjustments[ticker] = -abs(adjustments.get(ticker, 0.05)) * abs(mult)
                # Safe havens rally
                elif ticker in ["GLD", "TLT"]:
                    adjustments[ticker] = abs(adjustments.get(ticker, 0.05)) * abs(mult)
                # VXX spikes
                elif ticker == "VXX":
                    adjustments[ticker] = 0.50 * abs(mult)  # Vol spike
            elif scenario_type in ["bull", "tail_up"]:
                # Equities rally
                if ticker in ["SPY", "QQQ", "XLE"]:
                    adjustments[ticker] = abs(adjustments.get(ticker, 0.05)) * mult
                # Safe havens flat or down
                elif ticker in ["GLD", "TLT"]:
                    adjustments[ticker] = -0.02 * mult
                # VXX crashes
                elif ticker == "VXX":
                    adjustments[ticker] = -0.30 * mult

        return adjustments

    def _generate_events(
        self,
        scenario_type: str,
        num_days: int,
        start_date: datetime
    ) -> List[Dict]:
        """Generate events that occur during the scenario."""
        events = []

        # Scheduled events (from seed generator catalysts)
        scheduled = [
            {"day": 8, "event": "CPI Release", "impact": "HIGH"},
            {"day": 15, "event": "FOMC Meeting", "impact": "HIGH"},
            {"day": 17, "event": "Nonfarm Payrolls", "impact": "HIGH"},
            {"day": 28, "event": "NVDA Earnings", "impact": "HIGH"},
        ]

        for s in scheduled:
            if s["day"] <= num_days:
                event_date = start_date + timedelta(days=s["day"])
                outcome = self._generate_event_outcome(
                    s["event"], scenario_type
                )
                events.append({
                    "date": event_date.strftime("%Y-%m-%d"),
                    "day": s["day"],
                    "event": s["event"],
                    "impact": s["impact"],
                    "outcome": outcome,
                    "market_reaction": self._get_market_reaction(outcome, scenario_type),
                })

        # Random events for tail scenarios
        if scenario_type in ["tail_up", "tail_down"]:
            shock_day = random.randint(5, min(20, num_days))
            shock_date = start_date + timedelta(days=shock_day)
            shock_event = self._generate_shock_event(scenario_type)
            events.append({
                "date": shock_date.strftime("%Y-%m-%d"),
                "day": shock_day,
                "event": shock_event["event"],
                "impact": "EXTREME",
                "outcome": shock_event["outcome"],
                "market_reaction": shock_event["reaction"],
            })

        return sorted(events, key=lambda x: x["day"])

    def _generate_event_outcome(self, event: str, scenario_type: str) -> str:
        """Generate event outcome based on scenario."""
        outcomes = {
            "CPI Release": {
                "base": "CPI in line at 2.4% YoY",
                "bull": "CPI drops to 2.2%, disinflation accelerates",
                "bear": "CPI rises to 2.8%, inflation concerns resurface",
                "tail_up": "CPI plunges to 1.8%, Fed signals aggressive cuts",
                "tail_down": "CPI spikes to 3.5%, stagflation fears emerge",
            },
            "FOMC Meeting": {
                "base": "Fed holds, signals data dependency",
                "bull": "Fed signals 50bp cut coming, dovish surprise",
                "bear": "Fed warns of inflation persistence, hawkish hold",
                "tail_up": "Emergency 100bp cut, QE restart announced",
                "tail_down": "Fed warns of rate hike, credit concerns",
            },
            "Nonfarm Payrolls": {
                "base": "Jobs +180k, unemployment holds at 4.4%",
                "bull": "Strong jobs +250k, wage growth moderates",
                "bear": "Weak jobs +50k, unemployment rises to 4.7%",
                "tail_up": "Blowout +400k, Goldilocks narrative",
                "tail_down": "Jobs -100k, recession fears spike",
            },
            "NVDA Earnings": {
                "base": "NVDA beats, guides in line",
                "bull": "NVDA massive beat, raises guidance 20%",
                "bear": "NVDA misses, AI demand slowing",
                "tail_up": "NVDA +30% beat, announces $100B AI orders",
                "tail_down": "NVDA warns of demand cliff, stock -25%",
            },
        }

        return outcomes.get(event, {}).get(scenario_type, "Event occurs as expected")

    def _get_market_reaction(self, outcome: str, scenario_type: str) -> str:
        """Get market reaction description."""
        if scenario_type in ["bull", "tail_up"]:
            return "Markets rally, risk appetite increases"
        elif scenario_type in ["bear", "tail_down"]:
            return "Markets sell off, flight to quality"
        else:
            return "Muted reaction, markets consolidate"

    def _generate_shock_event(self, scenario_type: str) -> Dict:
        """Generate a shock event for tail scenarios."""
        if scenario_type == "tail_up":
            events = [
                {
                    "event": "Major AI breakthrough announced",
                    "outcome": "Google announces AGI-level system, productivity revolution",
                    "reaction": "Tech stocks surge 15%, SPY +5% in single day",
                },
                {
                    "event": "Peace deal in Middle East",
                    "outcome": "Iran-Israel peace agreement, oil drops 20%",
                    "reaction": "Risk-on rally, consumer stocks surge",
                },
                {
                    "event": "Fed announces QE4",
                    "outcome": "Unlimited bond buying, rates to 0",
                    "reaction": "Melt-up begins, VIX collapses",
                },
            ]
        else:  # tail_down
            events = [
                {
                    "event": "Major bank failure",
                    "outcome": "Large regional bank collapses, contagion fears",
                    "reaction": "Flight to quality, -8% SPY, VIX >50",
                },
                {
                    "event": "Geopolitical crisis escalates",
                    "outcome": "Iran-Israel conflict widens, oil spikes 40%",
                    "reaction": "Panic selling, bonds rally, gold spikes",
                },
                {
                    "event": "US credit downgrade",
                    "outcome": "Moody's downgrades US debt, yields spike",
                    "reaction": "Global risk-off, emerging markets crash",
                },
            ]

        return random.choice(events)

    def _calculate_final_state(
        self,
        paths: Dict[str, PricePath],
        events: List[Dict]
    ) -> Dict:
        """Calculate the market state at the end of the scenario."""
        state = {
            "assets": {},
            "narrative": "",
            "regime": "",
        }

        for ticker, path in paths.items():
            state["assets"][ticker] = {
                "start_price": path.start_price,
                "end_price": path.prices[-1],
                "return": path.cumulative_return,
                "volatility": path.volatility,
            }

        # Determine narrative from returns
        spy_return = paths.get("SPY", PricePath("", 0, [0, 0], [], [], 0, 0, "")).cumulative_return
        vxx_return = paths.get("VXX", PricePath("", 0, [0, 0], [], [], 0, 0, "")).cumulative_return

        if spy_return > 0.10:
            state["narrative"] = "Strong bull market, risk-on dominates"
            state["regime"] = "RISK_ON"
        elif spy_return < -0.10:
            state["narrative"] = "Sharp correction, defensive positioning wins"
            state["regime"] = "RISK_OFF"
        elif vxx_return > 0.30:
            state["narrative"] = "Elevated volatility, hedges pay off"
            state["regime"] = "HIGH_VOL"
        else:
            state["narrative"] = "Range-bound, stock pickers' market"
            state["regime"] = "NEUTRAL"

        return state

    def _generate_scenario_description(
        self,
        scenario_type: str,
        predictions: Dict,
        paths: Dict[str, PricePath]
    ) -> str:
        """Generate a narrative description of the scenario."""
        spy_ret = paths.get("SPY", PricePath("", 0, [0, 0], [], [], 0, 0, "")).cumulative_return
        gld_ret = paths.get("GLD", PricePath("", 0, [0, 0], [], [], 0, 0, "")).cumulative_return

        hct = predictions.get("highest_conviction_trade", {})
        hct_ticker = hct.get("ticker", "N/A")
        hct_dir = hct.get("direction", "N/A")

        descriptions = {
            "base": f"Markets follow consensus path. SPY {spy_ret*100:+.1f}%, GLD {gld_ret*100:+.1f}%. "
                    f"MiroFish HCT ({hct_ticker} {hct_dir}) plays out as expected.",
            "bull": f"Risk-on rally develops. SPY {spy_ret*100:+.1f}% as Fed pivots dovish. "
                    f"Tech leads, VXX crushed. Reflexive buying amplifies gains.",
            "bear": f"Risk-off correction unfolds. SPY {spy_ret*100:+.1f}% on macro concerns. "
                    f"Flight to quality, GLD {gld_ret*100:+.1f}%. Defensive positioning wins.",
            "tail_up": f"Melt-up scenario. SPY {spy_ret*100:+.1f}% as positive shocks cascade. "
                       f"FOMO intensifies, volatility collapses.",
            "tail_down": f"Crisis scenario. SPY {spy_ret*100:+.1f}% as negative feedback loops. "
                         f"VXX spikes, credit spreads widen, flight to quality.",
        }

        return descriptions.get(scenario_type, f"Scenario: {scenario_type}")

    def generate_all_scenarios(
        self,
        num_days: int = 30,
        scenarios: List[str] = None
    ) -> List[FutureScenario]:
        """Generate all scenario types."""
        predictions = self.load_predictions()
        if not predictions:
            logger.error("No predictions available")
            return []

        current_prices = self.load_current_prices()

        if scenarios is None:
            scenarios = ["base", "bull", "bear", "tail_up", "tail_down"]

        results = []
        for scenario_type in scenarios:
            try:
                scenario = self.generate_scenario(
                    predictions, current_prices, scenario_type, num_days
                )
                results.append(scenario)
                logger.info(f"Generated {scenario_type} scenario")
            except Exception as e:
                logger.error(f"Failed to generate {scenario_type}: {e}")

        # Save to file
        self._save_scenarios(results)

        return results

    def _save_scenarios(self, scenarios: List[FutureScenario]) -> None:
        """Save scenarios to file."""
        today = datetime.now().strftime("%Y%m%d")
        output_file = FUTURES_DIR / f"futures_{today}.json"

        data = {
            "generated_at": datetime.now().isoformat(),
            "scenarios": [asdict(s) for s in scenarios],
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved scenarios to {output_file}")


def print_scenarios(scenarios: List[FutureScenario]) -> None:
    """Print formatted scenario output."""
    print("\n" + "=" * 70)
    print("MIROFISH FUTURE SCENARIOS")
    print("=" * 70)

    for scenario in scenarios:
        print(f"\n{'─' * 70}")
        print(f"SCENARIO: {scenario.scenario_name}")
        print(f"Type: {scenario.scenario_type} | Probability: {scenario.probability:.0%}")
        print(f"{'─' * 70}")
        print(f"\n{scenario.description}\n")

        print("Price Paths:")
        for ticker, path_data in scenario.price_paths.items():
            ret = path_data["cumulative_return"]
            vol = path_data["volatility"]
            print(f"  {ticker}: {ret*100:+.1f}% (vol: {vol*100:.1f}%)")

        if scenario.events:
            print("\nKey Events:")
            for event in scenario.events[:3]:
                print(f"  Day {event['day']}: {event['event']}")
                print(f"    Outcome: {event['outcome']}")

        print(f"\nFinal State: {scenario.final_state['narrative']}")
        print(f"Regime: {scenario.final_state['regime']}")

    print("\n" + "=" * 70)


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    parser = argparse.ArgumentParser(description="MiroFish Futures Generator")
    parser.add_argument("--days", type=int, default=30, help="Days to simulate")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Scenario types to generate")
    parser.add_argument("--print", action="store_true", help="Print output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    generator = FuturesGenerator()
    scenarios = generator.generate_all_scenarios(args.days, args.scenarios)

    if scenarios:
        if args.json:
            print(json.dumps([asdict(s) for s in scenarios], indent=2))
        elif args.print:
            print_scenarios(scenarios)
        else:
            print(f"Generated {len(scenarios)} scenarios to {FUTURES_DIR}")


if __name__ == "__main__":
    main()
