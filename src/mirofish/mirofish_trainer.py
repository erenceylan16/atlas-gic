"""
MiroFish Forward Trainer
========================

Trains ATLAS agents by presenting them with MiroFish future scenarios
and evaluating their decisions against synthetic outcomes.

Process:
1. Load future scenarios from mirofish_futures_generator
2. Present each scenario to ATLAS agents
3. Collect agent recommendations
4. "Fast-forward" to scenario outcome
5. Score agent decisions
6. Update agent weights in autoresearch

This creates a feedback loop where agents learn from simulated futures,
separate from their real-world P&L-based learning.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

# Directories
STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "state"
MIROFISH_DIR = Path(__file__).resolve().parent.parent / "data" / "mirofish"
FUTURES_DIR = MIROFISH_DIR / "futures"
TRAINING_DIR = MIROFISH_DIR / "training"

# Agent weights file (shared with autoresearch)
AGENT_WEIGHTS_FILE = STATE_DIR / "agent_weights.json"
TRAINING_LOG_FILE = TRAINING_DIR / "training_log.json"


# Simplified agent prompts for training
AGENT_CONFIGS = {
    "druckenmiller": {
        "name": "Macro Strategist (Druckenmiller Style)",
        "style": "Concentrated macro bets, liquidity focused, 18-month horizon",
        "focus": "Fed policy, yield curve, dollar, gold, oil",
    },
    "ackman": {
        "name": "Activist Value (Ackman Style)",
        "style": "Concentrated positions, catalyst driven, corporate engagement",
        "focus": "Undervalued large caps with clear catalyst path",
    },
    "quant": {
        "name": "Systematic Quant",
        "style": "Factor-based, momentum, mean reversion signals",
        "focus": "Cross-sectional momentum, value, quality factors",
    },
    "bond_desk": {
        "name": "Bond Desk",
        "style": "Fixed income specialist, duration calls, credit spreads",
        "focus": "Rates trajectory, credit risk, flight-to-quality",
    },
    "energy_desk": {
        "name": "Energy Desk",
        "style": "Oil & gas specialist, geopolitical overlay",
        "focus": "Supply/demand, OPEC, geopolitical risk premium",
    },
}


class ForwardTrainer:
    """Trains ATLAS agents using MiroFish future scenarios."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = "claude-3-haiku-20240307"  # Fast model for training
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    def load_scenarios(self) -> Optional[Dict]:
        """Load latest future scenarios."""
        if not FUTURES_DIR.exists():
            logger.warning("No futures directory found")
            return None

        files = sorted(FUTURES_DIR.glob("futures_*.json"), reverse=True)
        if not files:
            logger.warning("No scenario files found")
            return None

        with open(files[0]) as f:
            return json.load(f)

    def load_agent_weights(self) -> Dict[str, float]:
        """Load current agent weights."""
        if AGENT_WEIGHTS_FILE.exists():
            with open(AGENT_WEIGHTS_FILE) as f:
                return json.load(f)
        # Default weights
        return {agent: 1.0 for agent in AGENT_CONFIGS}

    def save_agent_weights(self, weights: Dict[str, float]) -> None:
        """Save updated agent weights."""
        with open(AGENT_WEIGHTS_FILE, "w") as f:
            json.dump(weights, f, indent=2)

    def present_scenario_to_agent(
        self,
        agent_id: str,
        scenario: Dict,
        reveal_outcome: bool = False
    ) -> Dict:
        """Present a scenario to an agent and get their recommendation."""
        config = AGENT_CONFIGS.get(agent_id, {
            "name": agent_id,
            "style": "General",
            "focus": "Multi-asset",
        })

        # Build scenario context (don't reveal outcome yet)
        scenario_text = f"""
SCENARIO: {scenario['scenario_name']}
Type: {scenario['scenario_type']}
Probability: {scenario['probability']:.0%}

SETUP:
{scenario['description'][:200]}...

KEY EVENTS:
"""
        for event in scenario.get("events", [])[:3]:
            if reveal_outcome:
                scenario_text += f"- Day {event['day']}: {event['event']} → {event['outcome']}\n"
            else:
                scenario_text += f"- Day {event['day']}: {event['event']} (outcome unknown)\n"

        scenario_text += "\nPRICE PATHS (30-day forward simulation):\n"
        for ticker, path in list(scenario.get("price_paths", {}).items())[:10]:
            ret = path.get("cumulative_return", 0)
            scenario_text += f"  {ticker}: {ret*100:+.1f}%\n"

        # Build prompt
        prompt = f"""You are {config['name']}, with this investment style: {config['style']}.
Your focus areas: {config['focus']}.

Given this simulated future scenario, provide your trading recommendation.

{scenario_text}

Respond in this exact JSON format:
{{
    "recommendation": "BUY" | "SELL" | "HOLD",
    "tickers": ["ticker1", "ticker2"],
    "conviction": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text

            # Parse JSON from response
            # Find JSON block
            if "{" in text and "}" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                json_str = text[start:end]
                return json.loads(json_str)
            else:
                return {"error": "No JSON in response", "raw": text}

        except Exception as e:
            logger.error(f"Agent {agent_id} failed: {e}")
            return {"error": str(e)}

    def evaluate_recommendation(
        self,
        recommendation: Dict,
        scenario: Dict
    ) -> float:
        """Evaluate how well a recommendation performed in the scenario."""
        if "error" in recommendation:
            return 0.0

        tickers = recommendation.get("tickers", [])
        direction = recommendation.get("recommendation", "HOLD")
        conviction = recommendation.get("conviction", 0.5)

        # Calculate score based on price paths
        total_return = 0.0
        count = 0

        for ticker in tickers:
            path = scenario.get("price_paths", {}).get(ticker)
            if path:
                ret = path.get("cumulative_return", 0)

                # Adjust for direction
                if direction == "BUY":
                    total_return += ret
                elif direction == "SELL":
                    total_return -= ret  # Profit from short
                # HOLD gets 0

                count += 1

        if count == 0:
            return 0.5  # Neutral score for no picks

        avg_return = total_return / count

        # Convert to 0-1 score
        # +20% return = 1.0, -20% = 0.0, 0% = 0.5
        score = 0.5 + (avg_return / 0.40)
        score = max(0.0, min(1.0, score))

        # Weight by conviction (penalize wrong high-conviction calls)
        if avg_return < 0:
            score *= (1 - conviction * 0.3)  # Penalize wrong high-conviction
        else:
            score *= (1 + conviction * 0.2)  # Reward correct high-conviction
            score = min(1.0, score)

        return score

    def run_training_cycle(
        self,
        scenarios: List[Dict] = None,
        agents: List[str] = None
    ) -> Dict:
        """Run a full training cycle across scenarios and agents."""
        if scenarios is None:
            data = self.load_scenarios()
            if not data:
                return {"error": "No scenarios available"}
            scenarios = data.get("scenarios", [])

        if agents is None:
            agents = list(AGENT_CONFIGS.keys())

        weights = self.load_agent_weights()
        results = {
            "timestamp": datetime.now().isoformat(),
            "scenarios_tested": len(scenarios),
            "agents_trained": len(agents),
            "agent_scores": {},
            "weight_updates": {},
        }

        for agent_id in agents:
            agent_scores = []

            for scenario in scenarios:
                # Get recommendation
                rec = self.present_scenario_to_agent(agent_id, scenario)

                # Evaluate
                score = self.evaluate_recommendation(rec, scenario)
                agent_scores.append({
                    "scenario": scenario.get("scenario_type"),
                    "recommendation": rec,
                    "score": score,
                })

            # Calculate average score
            avg_score = sum(s["score"] for s in agent_scores) / len(agent_scores)
            results["agent_scores"][agent_id] = {
                "average_score": avg_score,
                "scenarios": agent_scores,
            }

            # Update weight based on performance
            # Score > 0.6 = boost, < 0.4 = reduce
            old_weight = weights.get(agent_id, 1.0)
            if avg_score > 0.6:
                new_weight = old_weight * (1 + (avg_score - 0.5) * 0.2)
            elif avg_score < 0.4:
                new_weight = old_weight * (1 - (0.5 - avg_score) * 0.2)
            else:
                new_weight = old_weight

            # Clamp weights
            new_weight = max(0.5, min(2.0, new_weight))

            if abs(new_weight - old_weight) > 0.01:
                results["weight_updates"][agent_id] = {
                    "old": old_weight,
                    "new": new_weight,
                    "change": new_weight - old_weight,
                }
                weights[agent_id] = new_weight

            logger.info(f"{agent_id}: avg_score={avg_score:.2f}, weight={new_weight:.2f}")

        # Save updated weights
        self.save_agent_weights(weights)

        # Save training log
        self._save_training_log(results)

        return results

    def _save_training_log(self, results: Dict) -> None:
        """Append results to training log."""
        if TRAINING_LOG_FILE.exists():
            with open(TRAINING_LOG_FILE) as f:
                log = json.load(f)
        else:
            log = {"sessions": []}

        log["sessions"].append(results)

        # Keep last 100 sessions
        log["sessions"] = log["sessions"][-100:]

        with open(TRAINING_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)

    def get_training_stats(self) -> Dict:
        """Get aggregate training statistics."""
        if not TRAINING_LOG_FILE.exists():
            return {"error": "No training data"}

        with open(TRAINING_LOG_FILE) as f:
            log = json.load(f)

        sessions = log.get("sessions", [])
        if not sessions:
            return {"error": "No sessions recorded"}

        # Aggregate by agent
        agent_stats = {}
        for session in sessions:
            for agent, data in session.get("agent_scores", {}).items():
                if agent not in agent_stats:
                    agent_stats[agent] = []
                agent_stats[agent].append(data["average_score"])

        return {
            "total_sessions": len(sessions),
            "agents": {
                agent: {
                    "sessions": len(scores),
                    "avg_score": sum(scores) / len(scores),
                    "best": max(scores),
                    "worst": min(scores),
                    "trend": scores[-1] - scores[0] if len(scores) > 1 else 0,
                }
                for agent, scores in agent_stats.items()
            },
            "weights": self.load_agent_weights(),
        }


def print_training_results(results: Dict) -> None:
    """Print formatted training results."""
    print("\n" + "=" * 70)
    print("MIROFISH FORWARD TRAINING RESULTS")
    print("=" * 70)
    print(f"Timestamp: {results.get('timestamp', 'N/A')}")
    print(f"Scenarios: {results.get('scenarios_tested', 0)}")
    print(f"Agents: {results.get('agents_trained', 0)}")

    print("\n" + "-" * 70)
    print("AGENT PERFORMANCE")
    print("-" * 70)

    for agent, data in results.get("agent_scores", {}).items():
        avg = data.get("average_score", 0)
        bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        print(f"  {agent:20} {bar} {avg:.2f}")

        # Show best/worst scenario
        scenarios = data.get("scenarios", [])
        if scenarios:
            best = max(scenarios, key=lambda x: x["score"])
            worst = min(scenarios, key=lambda x: x["score"])
            print(f"    Best:  {best['scenario']} ({best['score']:.2f})")
            print(f"    Worst: {worst['scenario']} ({worst['score']:.2f})")

    # Weight updates
    updates = results.get("weight_updates", {})
    if updates:
        print("\n" + "-" * 70)
        print("WEIGHT ADJUSTMENTS")
        print("-" * 70)
        for agent, data in updates.items():
            change = data["change"]
            direction = "↑" if change > 0 else "↓"
            print(f"  {agent}: {data['old']:.2f} → {data['new']:.2f} ({direction} {abs(change):.2f})")

    print("\n" + "=" * 70)


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    parser = argparse.ArgumentParser(description="MiroFish Forward Trainer")
    parser.add_argument("--train", action="store_true", help="Run training cycle")
    parser.add_argument("--stats", action="store_true", help="Show training stats")
    parser.add_argument("--agents", nargs="+", help="Specific agents to train")
    args = parser.parse_args()

    trainer = ForwardTrainer()

    if args.train:
        results = trainer.run_training_cycle(agents=args.agents)
        if "error" not in results:
            print_training_results(results)
        else:
            print(f"Error: {results['error']}")

    elif args.stats:
        stats = trainer.get_training_stats()
        print(json.dumps(stats, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
