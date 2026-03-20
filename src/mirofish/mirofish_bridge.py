"""
MiroFish Simulation Bridge for ATLAS
=====================================

Integrates swarm intelligence simulations with ATLAS trading decisions.

Two modes:
1. LIGHTWEIGHT MODE (default): Uses Claude to simulate multi-agent interactions
   - No external dependencies (Zep, OASIS)
   - Faster, lower latency
   - Good for rapid scenario exploration

2. FULL MIROFISH MODE: Uses MiroFish engine with Zep graph memory
   - Requires ZEP_API_KEY in .env
   - Richer agent interactions
   - Better for complex multi-round simulations

The bridge:
1. Generates seed data from current market state
2. Defines scenarios based on agent debates & market risks
3. Runs multi-agent simulations (fund managers, traders, central bankers)
4. Formats predictions for ATLAS agents
5. Scores predictions against actual outcomes

Output: data/state/mirofish_predictions.json
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict

import anthropic

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MODEL_PREMIUM
from data.macro_client import MacroClient
from data.price_client import PriceClient

logger = logging.getLogger(__name__)

# State files
STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "state"
PREDICTIONS_FILE = STATE_DIR / "mirofish_predictions.json"
SCORECARD_FILE = STATE_DIR / "mirofish_scorecard.json"
SEEDS_FILE = STATE_DIR / "mirofish_seeds.json"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Scenario:
    """A scenario to simulate."""
    id: str
    description: str
    key_variables: Dict[str, Any]
    timeframe: str  # '1w', '1m', '3m'
    source_agent: str  # Which ATLAS agent flagged this
    probability_prior: float = 0.5

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SimulatedPrediction:
    """A prediction from simulation."""
    event: str
    prediction: str
    confidence: float
    simulated_agents_agreeing: int
    key_driver: str
    timeframe: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TailRisk:
    """A tail risk identified by simulation."""
    event: str
    probability: float
    portfolio_impact: str
    recommended_hedge: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SimulationResult:
    """Full simulation output."""
    simulation_date: str
    mode: str  # 'lightweight' or 'mirofish'
    scenarios_simulated: int
    seed_summary: str
    consensus_predictions: List[SimulatedPrediction]
    divergent_predictions: List[Dict]
    tail_risks: List[TailRisk]
    highest_conviction_trade: Optional[Dict]
    reflexivity_signals: List[Dict]
    raw_debate: str

    def to_dict(self) -> Dict:
        return {
            "simulation_date": self.simulation_date,
            "mode": self.mode,
            "scenarios_simulated": self.scenarios_simulated,
            "seed_summary": self.seed_summary,
            "consensus_predictions": [p.to_dict() for p in self.consensus_predictions],
            "divergent_predictions": self.divergent_predictions,
            "tail_risks": [t.to_dict() for t in self.tail_risks],
            "highest_conviction_trade": self.highest_conviction_trade,
            "reflexivity_signals": self.reflexivity_signals,
            "raw_debate": self.raw_debate[:2000] + "..." if len(self.raw_debate) > 2000 else self.raw_debate
        }


# ============================================================================
# SEED GENERATION
# ============================================================================

class SeedGenerator:
    """
    Collects current market data and formats as seed material for simulation.
    """

    def __init__(self):
        self.macro = MacroClient()
        self.prices = PriceClient()
        self._ensure_state_dir()

    def _ensure_state_dir(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)

    def generate_seed(self) -> Dict[str, Any]:
        """
        Collect current market data and format as simulation seed.
        """
        logger.info("[MiroFish] Generating seed data...")

        # Macro snapshot
        try:
            macro = self.macro.get_macro_snapshot()
        except Exception as e:
            logger.warning(f"Failed to get macro data: {e}")
            macro = {}

        # Key prices
        try:
            key_tickers = ["SPY", "QQQ", "TLT", "GLD", "USO", "VXX", "UUP"]
            prices = {}
            for ticker in key_tickers:
                try:
                    p = self.prices.get_price(ticker)
                    if p:
                        prices[ticker] = p
                except:
                    pass
        except Exception as e:
            logger.warning(f"Failed to get prices: {e}")
            prices = {}

        # Load recent agent debates
        debates = self._load_recent_debates()

        # Load portfolio state
        portfolio = self._load_portfolio()

        seed = {
            "timestamp": datetime.utcnow().isoformat(),
            "macro_snapshot": {
                "fed_funds_rate": macro.get("fed_funds_rate"),
                "m2_yoy_change": macro.get("m2_yoy_change"),
                "yield_curve_10y_2y": macro.get("yield_curve_10y_2y"),
                "vix": macro.get("vix"),
                "cpi_yoy": macro.get("cpi_yoy"),
                "unemployment_rate": macro.get("unemployment_rate"),
            },
            "key_prices": prices,
            "agent_debates": debates,
            "portfolio_context": portfolio,
            "market_regime": self._assess_regime(macro, prices),
        }

        # Save seed
        self._save_seed(seed)

        return seed

    def _load_recent_debates(self) -> List[Dict]:
        """Load recent agent debate excerpts."""
        debates = []

        # Check desk briefs
        for desk in ["druckenmiller", "semiconductor", "biotech", "energy"]:
            brief_file = STATE_DIR / f"{desk}_briefs.json"
            if brief_file.exists():
                try:
                    with open(brief_file) as f:
                        briefs = json.load(f)
                    if briefs and isinstance(briefs, list) and len(briefs) > 0:
                        latest = briefs[-1]
                        debates.append({
                            "agent": desk,
                            "timestamp": latest.get("analyzed_at", latest.get("timestamp", "")),
                            "headline": latest.get("headline", ""),
                            "signal": latest.get("portfolio_tilt", latest.get("signal", "")),
                            "conviction": latest.get("conviction_level", latest.get("confidence", 0)),
                        })
                except Exception as e:
                    logger.debug(f"Could not load {desk} briefs: {e}")

        return debates

    def _load_portfolio(self) -> Dict:
        """Load current portfolio summary."""
        positions_file = STATE_DIR / "positions.json"
        if positions_file.exists():
            try:
                with open(positions_file) as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _assess_regime(self, macro: Dict, prices: Dict) -> str:
        """Quick regime assessment."""
        vix = macro.get("vix", 20)
        yc = macro.get("yield_curve_10y_2y", 0)

        if vix > 30:
            return "RISK_OFF_HIGH_VOL"
        elif vix > 20 and yc < 0:
            return "RISK_OFF_INVERSION"
        elif vix < 15:
            return "RISK_ON_COMPLACENT"
        else:
            return "NEUTRAL"

    def _save_seed(self, seed: Dict):
        """Save seed to file."""
        seeds = []
        if SEEDS_FILE.exists():
            try:
                with open(SEEDS_FILE) as f:
                    seeds = json.load(f)
            except:
                seeds = []

        seeds.append(seed)
        seeds = seeds[-30:]  # Keep last 30

        with open(SEEDS_FILE, "w") as f:
            json.dump(seeds, f, indent=2, default=str)


# ============================================================================
# SCENARIO GENERATION
# ============================================================================

class ScenarioGenerator:
    """
    Generates simulation scenarios from current market context and agent debates.
    """

    DEFAULT_SCENARIOS = [
        Scenario(
            id="fed_cut",
            description="Fed emergency rate cut of 50bp at next meeting",
            key_variables={"fed_funds_delta": -0.5, "vix_impact": -5},
            timeframe="1m",
            source_agent="macro_agent",
            probability_prior=0.15
        ),
        Scenario(
            id="fed_hike",
            description="Fed surprises with 25bp hike due to inflation",
            key_variables={"fed_funds_delta": 0.25, "vix_impact": 8},
            timeframe="1m",
            source_agent="macro_agent",
            probability_prior=0.10
        ),
        Scenario(
            id="oil_spike",
            description="Middle East escalation pushes oil above $120",
            key_variables={"oil_price": 120, "inflation_impact": 0.5},
            timeframe="1w",
            source_agent="geopolitical_agent",
            probability_prior=0.20
        ),
        Scenario(
            id="oil_crash",
            description="De-escalation and demand concerns crash oil to $75",
            key_variables={"oil_price": 75, "inflation_impact": -0.3},
            timeframe="1m",
            source_agent="energy_desk",
            probability_prior=0.15
        ),
        Scenario(
            id="tech_earnings_miss",
            description="Major AI names miss earnings, NVDA down 15%",
            key_variables={"nvda_move": -0.15, "qqq_move": -0.08},
            timeframe="1w",
            source_agent="semiconductor_desk",
            probability_prior=0.25
        ),
        Scenario(
            id="china_stimulus",
            description="China announces massive $500B stimulus package",
            key_variables={"em_rally": 0.10, "commodities_rally": 0.08},
            timeframe="1m",
            source_agent="macro_agent",
            probability_prior=0.20
        ),
        Scenario(
            id="recession_signal",
            description="US enters technical recession, 2 negative GDP quarters",
            key_variables={"spy_move": -0.12, "tlt_move": 0.08},
            timeframe="3m",
            source_agent="macro_agent",
            probability_prior=0.15
        ),
        Scenario(
            id="credit_event",
            description="Major corporate default triggers credit spread widening",
            key_variables={"hyg_move": -0.05, "vix_impact": 12},
            timeframe="1m",
            source_agent="bond_desk",
            probability_prior=0.10
        ),
    ]

    def generate_scenarios(self, seed: Dict, num_scenarios: int = 8) -> List[Scenario]:
        """
        Generate scenarios based on current market context.
        Uses both default scenarios and dynamic generation from debates.
        """
        scenarios = []

        # Start with relevant default scenarios
        vix = seed.get("macro_snapshot", {}).get("vix", 20)
        regime = seed.get("market_regime", "NEUTRAL")

        # Filter scenarios by relevance
        for s in self.DEFAULT_SCENARIOS:
            # Adjust probabilities based on regime
            adjusted = Scenario(
                id=s.id,
                description=s.description,
                key_variables=s.key_variables,
                timeframe=s.timeframe,
                source_agent=s.source_agent,
                probability_prior=self._adjust_probability(s, regime, vix)
            )
            scenarios.append(adjusted)

        # Add scenarios from agent debates
        debates = seed.get("agent_debates", [])
        for debate in debates:
            if debate.get("conviction", 0) > 0.7:
                # High conviction agent view becomes a scenario
                scenarios.append(Scenario(
                    id=f"agent_{debate['agent']}_view",
                    description=f"{debate['agent']} high conviction: {debate.get('headline', '')}",
                    key_variables={"agent_signal": debate.get("signal", "")},
                    timeframe="1m",
                    source_agent=debate["agent"],
                    probability_prior=debate.get("conviction", 0.5) * 0.8
                ))

        # Sort by probability and take top N
        scenarios.sort(key=lambda x: x.probability_prior, reverse=True)
        return scenarios[:num_scenarios]

    def _adjust_probability(self, scenario: Scenario, regime: str, vix: float) -> float:
        """Adjust scenario probability based on current regime."""
        prob = scenario.probability_prior

        if regime == "RISK_OFF_HIGH_VOL":
            if "crash" in scenario.id or "miss" in scenario.id or "recession" in scenario.id:
                prob *= 1.5
            elif "rally" in scenario.description.lower():
                prob *= 0.5
        elif regime == "RISK_ON_COMPLACENT":
            if "crash" in scenario.id or "miss" in scenario.id:
                prob *= 0.7  # Less likely but still possible
            if "spike" in scenario.id or "rally" in scenario.description.lower():
                prob *= 1.2

        return min(prob, 0.9)  # Cap at 90%


# ============================================================================
# LIGHTWEIGHT SIMULATION (Claude-based)
# ============================================================================

SIMULATION_SYSTEM_PROMPT = """You are simulating a financial market environment with multiple agents.

You will role-play ALL of the following agent types simultaneously, generating their reactions to scenarios:

AGENTS:
1. HEDGE FUND MANAGERS (3 types):
   - Macro Fund: Trades on Fed policy, liquidity, cycles. Concentrated bets.
   - Quant Fund: Systematic signals, mean reversion, momentum factors.
   - L/S Equity: Bottom-up stock picking, pair trades.

2. MARKET PARTICIPANTS:
   - Central Banker: Monitors inflation, employment, financial stability.
   - Corporate Treasurer: Manages FX exposure, debt issuance, buybacks.
   - Retail Investor: Follows social sentiment, meme stocks, options.
   - Pension Fund: Long-term allocation, liability matching.

3. INFORMATION AGENTS:
   - Sell-Side Analyst: Publishes price targets, earnings estimates.
   - Business Journalist: Breaks news, shapes narrative.

REFLEXIVITY RULES (CRITICAL):
These feedback loops must be modeled in your simulation:

1. PRICE -> FUNDAMENTALS
   - Stock drops >15%: credit downgrade risk, talent flight, customer renegotiations
   - Stock rises >20%: cheap capital access, M&A capability, talent attraction

2. P&L -> BEHAVIOR
   - Fund with >10% drawdown: forced selling at further weakness
   - Fund with >15% gains: increases position sizes, takes concentrated bets
   - Margin calls cascade: forced liquidation triggers more forced liquidation

3. NARRATIVE -> FLOWS
   - 3+ analysts converge on thesis: retail follows within 2 rounds
   - After 5 rounds of consensus: contrarian reversals emerge

4. MARKET -> POLICY
   - Equity drawdown >15%: central bank signals easing
   - Oil >$130: strategic reserve releases
   - Unemployment >5%: fiscal stimulus announced

5. REFLEXIVE REVERSAL DETECTION
   - Track feedback loops running 5+ rounds in one direction
   - Flag as "reflexive extreme" - historical turning points
   - When loop shows signs of breaking, that's highest conviction signal

OUTPUT FORMAT (JSON):
{
  "round_by_round": [
    {
      "round": 1,
      "agent_actions": [
        {"agent": "Macro Fund", "action": "...", "reasoning": "..."},
        ...
      ],
      "market_impact": {"spy": "+/-X%", "tlt": "+/-X%", "vix": "+/-X"},
      "reflexive_loops_active": ["PRICE->FUNDAMENTALS on NVDA"]
    }
  ],
  "final_state": {
    "spy_change": "+/-X%",
    "tlt_change": "+/-X%",
    "vix_level": X,
    "dominant_narrative": "...",
    "reflexive_extremes": ["loop description"]
  },
  "predictions": [
    {
      "event": "description",
      "prediction": "outcome",
      "confidence": 0.0-1.0,
      "agents_agreeing": N,
      "key_driver": "...",
      "timeframe": "1w/1m/3m"
    }
  ],
  "tail_risks": [
    {
      "event": "description",
      "probability": 0.0-1.0,
      "portfolio_impact": "-X% in Y hours",
      "recommended_hedge": "..."
    }
  ],
  "highest_conviction_trade": {
    "ticker": "...",
    "direction": "LONG/SHORT",
    "reasoning": "...",
    "timeframe": "...",
    "agents_supporting": N
  }
}
"""


class LightweightSimulator:
    """
    Claude-based multi-agent simulation.
    Simulates market agent interactions without external dependencies.
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = CLAUDE_MODEL_PREMIUM  # Premium for complex reasoning

    def run_simulation(
        self,
        seed: Dict,
        scenarios: List[Scenario],
        num_rounds: int = 10
    ) -> SimulationResult:
        """
        Run multi-agent simulation using Claude.
        """
        logger.info(f"[MiroFish] Running lightweight simulation with {len(scenarios)} scenarios, {num_rounds} rounds")

        # Build simulation prompt
        prompt = self._build_simulation_prompt(seed, scenarios, num_rounds)

        # Call Claude
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=SIMULATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_response = response.content[0].text
        except Exception as e:
            logger.error(f"[MiroFish] Simulation failed: {e}")
            return self._empty_result()

        # Parse response
        return self._parse_simulation_result(raw_response, seed, scenarios)

    def _build_simulation_prompt(
        self,
        seed: Dict,
        scenarios: List[Scenario],
        num_rounds: int
    ) -> str:
        """Build the simulation prompt with seed data and scenarios."""

        macro = seed.get("macro_snapshot", {})
        prices = seed.get("key_prices", {})
        regime = seed.get("market_regime", "NEUTRAL")
        debates = seed.get("agent_debates", [])

        prompt = f"""SIMULATION SEED DATA
==================

Date: {seed.get('timestamp', datetime.utcnow().isoformat())}
Market Regime: {regime}

MACRO SNAPSHOT:
- Fed Funds Rate: {macro.get('fed_funds_rate', 'N/A')}%
- M2 YoY Change: {macro.get('m2_yoy_change', 'N/A')}%
- Yield Curve (10Y-2Y): {macro.get('yield_curve_10y_2y', 'N/A')} bps
- VIX: {macro.get('vix', 'N/A')}
- CPI YoY: {macro.get('cpi_yoy', 'N/A')}%
- Unemployment: {macro.get('unemployment_rate', 'N/A')}%

KEY PRICES:
"""
        for ticker, price in prices.items():
            prompt += f"- {ticker}: ${price}\n"

        if debates:
            prompt += "\nATLAS AGENT VIEWS (from recent debates):\n"
            for d in debates:
                prompt += f"- {d.get('agent', 'Agent')}: {d.get('headline', 'No view')} (conviction: {d.get('conviction', 0):.0%})\n"

        prompt += "\n\nSCENARIOS TO SIMULATE:\n"
        prompt += "========================\n"
        for i, s in enumerate(scenarios, 1):
            prompt += f"""
{i}. {s.description}
   Prior probability: {s.probability_prior:.0%}
   Timeframe: {s.timeframe}
   Key variables: {json.dumps(s.key_variables)}
   Source: {s.source_agent}
"""

        prompt += f"""

SIMULATION INSTRUCTIONS:
========================

Run {num_rounds} rounds of agent interaction for each scenario.

For each round:
1. Each agent type responds to the current state
2. Track cumulative market impact
3. Identify active reflexive feedback loops
4. Note when loops approach extremes

After all rounds:
1. Synthesize predictions across scenarios
2. Calculate confidence-weighted consensus
3. Identify divergent predictions (where agents disagree)
4. Flag tail risks
5. Determine highest conviction trade signal

IMPORTANT: Model reflexive dynamics. Prices affect fundamentals. P&L affects behavior. Narrative drives flows. Policy responds to markets.

Return your analysis as valid JSON matching the output format.
"""

        return prompt

    def _parse_simulation_result(
        self,
        raw: str,
        seed: Dict,
        scenarios: List[Scenario]
    ) -> SimulationResult:
        """Parse Claude's simulation output."""

        try:
            # Extract JSON
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0]
            else:
                json_str = raw

            data = json.loads(json_str.strip())

            # Parse predictions
            predictions = []
            for p in data.get("predictions", []):
                predictions.append(SimulatedPrediction(
                    event=p.get("event", ""),
                    prediction=p.get("prediction", ""),
                    confidence=p.get("confidence", 0.5),
                    simulated_agents_agreeing=p.get("agents_agreeing", 0),
                    key_driver=p.get("key_driver", ""),
                    timeframe=p.get("timeframe", "1m")
                ))

            # Parse tail risks
            tail_risks = []
            for t in data.get("tail_risks", []):
                tail_risks.append(TailRisk(
                    event=t.get("event", ""),
                    probability=t.get("probability", 0.1),
                    portfolio_impact=t.get("portfolio_impact", ""),
                    recommended_hedge=t.get("recommended_hedge", "")
                ))

            # Parse divergent predictions
            divergent = []
            final_state = data.get("final_state", {})
            if final_state.get("reflexive_extremes"):
                for extreme in final_state["reflexive_extremes"]:
                    divergent.append({
                        "type": "reflexive_extreme",
                        "description": extreme
                    })

            return SimulationResult(
                simulation_date=datetime.utcnow().isoformat(),
                mode="lightweight",
                scenarios_simulated=len(scenarios),
                seed_summary=f"Regime: {seed.get('market_regime')}, VIX: {seed.get('macro_snapshot', {}).get('vix')}",
                consensus_predictions=predictions,
                divergent_predictions=divergent,
                tail_risks=tail_risks,
                highest_conviction_trade=data.get("highest_conviction_trade"),
                reflexivity_signals=final_state.get("reflexive_extremes", []),
                raw_debate=raw
            )

        except json.JSONDecodeError as e:
            logger.error(f"[MiroFish] Failed to parse simulation: {e}")
            logger.debug(f"Raw output: {raw[:500]}...")
            return self._empty_result()

    def _empty_result(self) -> SimulationResult:
        """Return empty result on failure."""
        return SimulationResult(
            simulation_date=datetime.utcnow().isoformat(),
            mode="lightweight",
            scenarios_simulated=0,
            seed_summary="FAILED",
            consensus_predictions=[],
            divergent_predictions=[],
            tail_risks=[],
            highest_conviction_trade=None,
            reflexivity_signals=[],
            raw_debate="Simulation failed"
        )


# ============================================================================
# PREDICTION SCORING
# ============================================================================

class PredictionScorer:
    """
    Scores past predictions against actual outcomes.
    Tracks accuracy by scenario type for autoresearch integration.
    """

    def __init__(self):
        self.prices = PriceClient()
        self._ensure_state_dir()

    def _ensure_state_dir(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)

    def score_predictions(self, days_elapsed: int = 5) -> Dict:
        """
        Score predictions from N days ago against actual outcomes.
        """
        if not PREDICTIONS_FILE.exists():
            logger.warning("No predictions file found")
            return {}

        with open(PREDICTIONS_FILE) as f:
            predictions = json.load(f)

        # Find predictions from N days ago
        cutoff = datetime.utcnow() - timedelta(days=days_elapsed)
        cutoff_str = cutoff.isoformat()[:10]

        to_score = []
        for pred in predictions:
            pred_date = pred.get("simulation_date", "")[:10]
            if pred_date <= cutoff_str and not pred.get("scored", False):
                to_score.append(pred)

        if not to_score:
            logger.info("No predictions ready to score")
            return {}

        # Score each prediction
        scores = []
        for pred in to_score:
            score = self._score_single_prediction(pred)
            scores.append(score)
            pred["scored"] = True
            pred["score"] = score

        # Save updated predictions
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump(predictions, f, indent=2, default=str)

        # Update scorecard
        self._update_scorecard(scores)

        return {
            "predictions_scored": len(scores),
            "scores": scores
        }

    def _score_single_prediction(self, pred: Dict) -> Dict:
        """Score a single prediction against actual outcome."""
        consensus = pred.get("consensus_predictions", [])

        scores = []
        for p in consensus:
            event = p.get("event", "")
            prediction = p.get("prediction", "")

            # Try to match prediction to actual outcome
            actual = self._get_actual_outcome(event, pred.get("simulation_date"))

            direction_correct = self._check_direction(prediction, actual)
            magnitude_close = self._check_magnitude(prediction, actual)

            scores.append({
                "event": event,
                "prediction": prediction,
                "actual": actual,
                "direction_correct": direction_correct,
                "magnitude_close": magnitude_close,
                "score": 1.0 if direction_correct and magnitude_close else 0.5 if direction_correct else 0.0
            })

        return {
            "date": pred.get("simulation_date"),
            "prediction_scores": scores,
            "average_score": sum(s["score"] for s in scores) / len(scores) if scores else 0
        }

    def _get_actual_outcome(self, event: str, date: str) -> str:
        """Get what actually happened for this event."""
        event_lower = event.lower()

        if "spy" in event_lower or "sp500" in event_lower:
            try:
                price = self.prices.get_price("SPY")
                return f"SPY at ${price}"
            except:
                return "Unknown"

        if "oil" in event_lower:
            try:
                price = self.prices.get_price("USO")
                return f"USO at ${price}"
            except:
                return "Unknown"

        return "Outcome data not available"

    def _check_direction(self, prediction: str, actual: str) -> bool:
        """Check if predicted direction matched."""
        pred_lower = prediction.lower()
        actual_lower = actual.lower()

        pred_up = any(w in pred_lower for w in ["rise", "rally", "bull", "up", "increase", "higher"])
        pred_down = any(w in pred_lower for w in ["fall", "crash", "bear", "down", "decrease", "lower"])

        actual_up = any(w in actual_lower for w in ["rise", "rally", "bull", "up", "increase", "higher"])
        actual_down = any(w in actual_lower for w in ["fall", "crash", "bear", "down", "decrease", "lower"])

        return (pred_up and actual_up) or (pred_down and actual_down)

    def _check_magnitude(self, prediction: str, actual: str) -> bool:
        """Check if predicted magnitude was close."""
        return True  # Simplified for now

    def _update_scorecard(self, scores: List[Dict]):
        """Update running scorecard."""
        scorecard = {}
        if SCORECARD_FILE.exists():
            try:
                with open(SCORECARD_FILE) as f:
                    scorecard = json.load(f)
            except:
                scorecard = {}

        if "history" not in scorecard:
            scorecard["history"] = []

        scorecard["history"].extend(scores)
        scorecard["history"] = scorecard["history"][-100:]  # Keep last 100

        # Calculate stats
        all_scores = [s["average_score"] for s in scorecard["history"]]
        scorecard["overall_accuracy"] = sum(all_scores) / len(all_scores) if all_scores else 0
        scorecard["total_predictions_scored"] = len(scorecard["history"])
        scorecard["last_updated"] = datetime.utcnow().isoformat()

        with open(SCORECARD_FILE, "w") as f:
            json.dump(scorecard, f, indent=2, default=str)


# ============================================================================
# MAIN BRIDGE CLASS
# ============================================================================

class MiroFishBridge:
    """
    Main interface for MiroFish integration with ATLAS.

    Usage:
        bridge = MiroFishBridge()

        # Generate and run simulation
        result = bridge.generate_and_simulate()

        # Score past predictions
        scores = bridge.score_predictions()

        # Get formatted context for ATLAS agents
        context = bridge.get_agent_context()
    """

    def __init__(self, use_full_mirofish: bool = False):
        """
        Initialize bridge.

        Args:
            use_full_mirofish: If True, tries to use full MiroFish engine.
                               If False (default), uses lightweight Claude simulation.
        """
        self.seed_generator = SeedGenerator()
        self.scenario_generator = ScenarioGenerator()
        self.simulator = LightweightSimulator()
        self.scorer = PredictionScorer()
        self.use_full_mirofish = use_full_mirofish
        self._ensure_state_dir()

    def _ensure_state_dir(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)

    def generate_and_simulate(
        self,
        num_scenarios: int = 8,
        num_rounds: int = 10
    ) -> SimulationResult:
        """
        Main entry point: generate seed, scenarios, run simulation.
        """
        logger.info("[MiroFish] Starting simulation pipeline...")

        # 1. Generate seed
        seed = self.seed_generator.generate_seed()
        logger.info(f"[MiroFish] Seed generated: {seed.get('market_regime')}")

        # 2. Generate scenarios
        scenarios = self.scenario_generator.generate_scenarios(seed, num_scenarios)
        logger.info(f"[MiroFish] Generated {len(scenarios)} scenarios")

        # 3. Run simulation
        result = self.simulator.run_simulation(seed, scenarios, num_rounds)
        logger.info(f"[MiroFish] Simulation complete: {len(result.consensus_predictions)} predictions")

        # 4. Save results
        self._save_predictions(result)

        return result

    def _save_predictions(self, result: SimulationResult):
        """Save simulation results."""
        predictions = []
        if PREDICTIONS_FILE.exists():
            try:
                with open(PREDICTIONS_FILE) as f:
                    predictions = json.load(f)
            except:
                predictions = []

        predictions.append(result.to_dict())
        predictions = predictions[-30:]  # Keep last 30

        with open(PREDICTIONS_FILE, "w") as f:
            json.dump(predictions, f, indent=2, default=str)

        logger.info(f"[MiroFish] Saved predictions to {PREDICTIONS_FILE}")

    def score_predictions(self, days_elapsed: int = 5) -> Dict:
        """Score predictions from N days ago."""
        return self.scorer.score_predictions(days_elapsed)

    def get_agent_context(self) -> str:
        """
        Get formatted context string for injection into ATLAS agent prompts.
        """
        if not PREDICTIONS_FILE.exists():
            return "No MiroFish simulations available."

        with open(PREDICTIONS_FILE) as f:
            predictions = json.load(f)

        if not predictions:
            return "No MiroFish simulations available."

        latest = predictions[-1]

        context = f"""
## Forward-Looking Context (MiroFish Simulations)

Simulation Date: {latest.get('simulation_date', 'Unknown')[:10]}
Scenarios Simulated: {latest.get('scenarios_simulated', 0)}
Mode: {latest.get('mode', 'lightweight')}

### Consensus Predictions:
"""

        for pred in latest.get("consensus_predictions", [])[:5]:
            context += f"- **{pred.get('event', 'Event')}**: {pred.get('prediction', 'N/A')} "
            context += f"(confidence: {pred.get('confidence', 0):.0%}, "
            context += f"agents agreeing: {pred.get('simulated_agents_agreeing', 0)})\n"
            context += f"  Key driver: {pred.get('key_driver', 'N/A')}\n"

        if latest.get("tail_risks"):
            context += "\n### Tail Risks to Monitor:\n"
            for risk in latest.get("tail_risks", [])[:3]:
                context += f"- **{risk.get('event', 'Event')}** (prob: {risk.get('probability', 0):.0%})\n"
                context += f"  Impact: {risk.get('portfolio_impact', 'N/A')}\n"
                context += f"  Hedge: {risk.get('recommended_hedge', 'N/A')}\n"

        hct = latest.get("highest_conviction_trade")
        if hct:
            context += f"""
### Highest Conviction Trade Signal:
- Ticker: {hct.get('ticker', 'N/A')}
- Direction: {hct.get('direction', 'N/A')}
- Reasoning: {hct.get('reasoning', 'N/A')}
- Timeframe: {hct.get('timeframe', 'N/A')}
"""

        if latest.get("reflexivity_signals"):
            context += "\n### Reflexive Extremes Detected:\n"
            for signal in latest.get("reflexivity_signals", [])[:3]:
                if isinstance(signal, str):
                    context += f"- {signal}\n"
                else:
                    context += f"- {signal.get('description', str(signal))}\n"

        context += "\n*These are simulations, not certainties. Weight alongside your own analysis.*"

        return context

    def get_latest_predictions(self) -> Optional[Dict]:
        """Get the latest predictions dict."""
        if not PREDICTIONS_FILE.exists():
            return None

        with open(PREDICTIONS_FILE) as f:
            predictions = json.load(f)

        return predictions[-1] if predictions else None


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    parser = argparse.ArgumentParser(description="ATLAS MiroFish Simulation Bridge")
    parser.add_argument("--generate-and-simulate", action="store_true",
                        help="Generate seed data and run simulation")
    parser.add_argument("--score", action="store_true",
                        help="Score predictions from 5 days ago")
    parser.add_argument("--context", action="store_true",
                        help="Print agent context string")
    parser.add_argument("--scenarios", type=int, default=8,
                        help="Number of scenarios to simulate (default: 8)")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of simulation rounds (default: 10)")
    parser.add_argument("--days", type=int, default=5,
                        help="Days elapsed for scoring (default: 5)")
    args = parser.parse_args()

    bridge = MiroFishBridge()

    if args.generate_and_simulate:
        print("\n" + "="*70)
        print("MIROFISH SIMULATION")
        print("="*70)

        result = bridge.generate_and_simulate(
            num_scenarios=args.scenarios,
            num_rounds=args.rounds
        )

        print(f"\nSimulation Date: {result.simulation_date}")
        print(f"Mode: {result.mode}")
        print(f"Scenarios Simulated: {result.scenarios_simulated}")
        print(f"Seed Summary: {result.seed_summary}")

        print("\n--- CONSENSUS PREDICTIONS ---")
        for p in result.consensus_predictions[:5]:
            print(f"\n{p.event}")
            print(f"  Prediction: {p.prediction}")
            print(f"  Confidence: {p.confidence:.0%}")
            print(f"  Key Driver: {p.key_driver}")

        print("\n--- TAIL RISKS ---")
        for t in result.tail_risks[:3]:
            print(f"\n{t.event} (prob: {t.probability:.0%})")
            print(f"  Impact: {t.portfolio_impact}")
            print(f"  Hedge: {t.recommended_hedge}")

        if result.highest_conviction_trade:
            hct = result.highest_conviction_trade
            print("\n--- HIGHEST CONVICTION TRADE ---")
            print(f"Ticker: {hct.get('ticker', 'N/A')}")
            print(f"Direction: {hct.get('direction', 'N/A')}")
            print(f"Reasoning: {hct.get('reasoning', 'N/A')}")

        print("\n" + "="*70)
        print(f"Results saved to: {PREDICTIONS_FILE}")

    elif args.score:
        print("\nScoring predictions...")
        scores = bridge.score_predictions(days_elapsed=args.days)
        print(json.dumps(scores, indent=2, default=str))

    elif args.context:
        context = bridge.get_agent_context()
        print(context)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
