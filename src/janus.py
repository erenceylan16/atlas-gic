"""
JANUS - Meta-Weighting Layer for ATLAS

Sits above two (or more) agent cohorts and dynamically weights their
recommendations based on recent accuracy. The weight differential
between cohorts is an emergent regime detector.

Usage:
    from agents.janus import Janus

    janus = Janus(cohorts=["18month", "10year"])
    janus.update_weights()  # Score recent performance, compute new weights
    blended = janus.blend_recommendations()  # Get weighted blend
    regime = janus.regime_signal()  # Detect market regime
"""

import json
import logging
import math
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)


class Janus:
    """
    Meta-weighting layer that blends recommendations from multiple agent cohorts
    based on their rolling accuracy.
    """

    # Weight constraints
    MIN_WEIGHT = 0.2  # No cohort drops below 20% influence
    MAX_WEIGHT = 0.8  # No cohort dominates above 80%
    ROLLING_WINDOW = 30  # Days for rolling accuracy calculation
    REGIME_THRESHOLD = 0.15  # Weight diff threshold for regime signals

    def __init__(self, cohorts: List[str] = None):
        """
        Initialize JANUS with specified cohorts.

        Args:
            cohorts: List of cohort names (e.g., ["18month", "10year"])
                     Defaults to ["18month", "10year"] if not specified.
        """
        self.cohorts = cohorts or ["18month", "10year"]
        self.cohort_weights: Dict[str, float] = {}
        self.cohort_accuracy: Dict[str, Dict[str, float]] = {}
        self.daily_file = STATE_DIR / "janus_daily.json"
        self.history_file = STATE_DIR / "janus_history.json"

        # Initialize equal weights
        self._initialize_weights()
        logger.info(f"[janus] Initialized with cohorts: {self.cohorts}")

    def _initialize_weights(self):
        """Set equal weights for all cohorts."""
        equal_weight = 1.0 / len(self.cohorts)
        for cohort in self.cohorts:
            self.cohort_weights[cohort] = equal_weight
            self.cohort_accuracy[cohort] = {"hit_rate": 0.5, "sharpe": 0.0}

    def _get_recommendation_file(self, cohort: str) -> Path:
        """Get path to cohort's recommendation file."""
        return STATE_DIR / f"recommendations_{cohort}.json"

    def _get_scored_outcomes_file(self) -> Path:
        """Get path to scored outcomes file."""
        return STATE_DIR / "scored_outcomes.json"

    def load_recommendations(self, cohort: str) -> Optional[Dict]:
        """
        Load latest recommendations for a cohort.

        Args:
            cohort: Cohort name

        Returns:
            Recommendation dict or None if not found
        """
        rec_file = self._get_recommendation_file(cohort)
        if not rec_file.exists():
            logger.warning(f"[janus] No recommendations file for cohort: {cohort}")
            return None

        try:
            with open(rec_file, "r") as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"[janus] Error loading recommendations for {cohort}: {e}")
            return None

    def load_all_recommendations(self) -> Dict[str, Dict]:
        """Load recommendations from all cohorts."""
        all_recs = {}
        for cohort in self.cohorts:
            recs = self.load_recommendations(cohort)
            if recs:
                all_recs[cohort] = recs
        return all_recs

    def load_scored_outcomes(self) -> List[Dict]:
        """
        Load historical scored outcomes for accuracy calculation.

        Returns:
            List of scored outcome records
        """
        outcomes_file = self._get_scored_outcomes_file()
        if not outcomes_file.exists():
            logger.warning("[janus] No scored outcomes file found")
            return []

        try:
            with open(outcomes_file, "r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else data.get("outcomes", [])
        except Exception as e:
            logger.error(f"[janus] Error loading scored outcomes: {e}")
            return []

    def score_recommendation(self, rec: Dict, actual_return: float) -> Dict:
        """
        Score a single recommendation against actual return.

        Args:
            rec: Recommendation dict with 'direction' and 'conviction'
            actual_return: Next-day return (decimal, e.g., 0.02 for +2%)

        Returns:
            Scored record with hit/miss and conviction-weighted return
        """
        direction = rec.get("direction", "LONG").upper()
        conviction = rec.get("conviction", 50) / 100.0  # Normalize to 0-1

        # Determine if prediction was correct
        if direction == "LONG":
            is_hit = actual_return > 0
        else:  # SHORT
            is_hit = actual_return < 0

        # Conviction-weighted return (positive for correct, negative for wrong)
        if direction == "LONG":
            weighted_return = conviction * actual_return
        else:  # SHORT: profit from decline
            weighted_return = conviction * (-actual_return)

        return {
            "ticker": rec.get("ticker"),
            "direction": direction,
            "conviction": rec.get("conviction"),
            "actual_return": actual_return,
            "is_hit": is_hit,
            "weighted_return": weighted_return
        }

    def calculate_cohort_metrics(self, cohort: str, outcomes: List[Dict]) -> Dict[str, float]:
        """
        Calculate accuracy metrics for a cohort over the rolling window.

        Args:
            cohort: Cohort name
            outcomes: List of scored outcomes

        Returns:
            Dict with hit_rate and sharpe ratio
        """
        # Filter outcomes for this cohort within rolling window
        cutoff_date = (date.today() - timedelta(days=self.ROLLING_WINDOW)).isoformat()
        cohort_outcomes = [
            o for o in outcomes
            if o.get("cohort") == cohort and o.get("date", "") >= cutoff_date
        ]

        if not cohort_outcomes:
            return {"hit_rate": 0.5, "sharpe": 0.0}

        # Calculate hit rate
        hits = sum(1 for o in cohort_outcomes if o.get("is_hit", False))
        hit_rate = hits / len(cohort_outcomes)

        # Calculate Sharpe ratio of weighted returns
        weighted_returns = [o.get("weighted_return", 0) for o in cohort_outcomes]

        if len(weighted_returns) < 2:
            sharpe = 0.0
        else:
            mean_return = sum(weighted_returns) / len(weighted_returns)
            variance = sum((r - mean_return) ** 2 for r in weighted_returns) / (len(weighted_returns) - 1)
            std_dev = math.sqrt(variance) if variance > 0 else 0.0001

            # Annualized Sharpe (assuming daily returns)
            sharpe = (mean_return / std_dev) * math.sqrt(252) if std_dev > 0 else 0.0

        return {"hit_rate": hit_rate, "sharpe": sharpe}

    def _softmax_with_constraints(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Apply softmax to scores with min/max weight constraints.

        Args:
            scores: Dict mapping cohort names to raw scores

        Returns:
            Dict mapping cohort names to constrained weights summing to 1.0
        """
        if not scores:
            return {}

        # Softmax
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        weights = {k: v / total for k, v in exp_scores.items()}

        # Apply floor constraint
        for cohort in weights:
            if weights[cohort] < self.MIN_WEIGHT:
                weights[cohort] = self.MIN_WEIGHT

        # Renormalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Apply ceiling constraint
        for cohort in weights:
            if weights[cohort] > self.MAX_WEIGHT:
                weights[cohort] = self.MAX_WEIGHT

        # Final renormalization
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return weights

    def update_weights(self, outcomes: List[Dict] = None):
        """
        Update cohort weights based on recent accuracy.

        Args:
            outcomes: Optional list of scored outcomes. If not provided,
                     loads from file.
        """
        if outcomes is None:
            outcomes = self.load_scored_outcomes()

        # Calculate metrics for each cohort
        raw_scores = {}
        for cohort in self.cohorts:
            metrics = self.calculate_cohort_metrics(cohort, outcomes)
            self.cohort_accuracy[cohort] = metrics

            # Combined score: 50% hit rate + 50% normalized Sharpe
            # Normalize Sharpe to roughly 0-1 range (divide by 2, clip)
            norm_sharpe = max(0, min(1, (metrics["sharpe"] + 1) / 2))
            raw_scores[cohort] = 0.5 * metrics["hit_rate"] + 0.5 * norm_sharpe

        # Apply softmax with constraints
        self.cohort_weights = self._softmax_with_constraints(raw_scores)

        logger.info(f"[janus] Updated weights: {self.cohort_weights}")
        logger.info(f"[janus] Cohort accuracy: {self.cohort_accuracy}")

    def regime_signal(self) -> str:
        """
        Detect market regime based on weight differential.

        Returns:
            "NOVEL_REGIME" - Short-window agents outperforming (unusual market)
            "HISTORICAL_REGIME" - Long-window agents outperforming (classical patterns)
            "MIXED" - Both roughly equal
        """
        # Get weights for short vs long trained cohorts
        short_weight = self.cohort_weights.get("18month", 0.5)
        long_weight = self.cohort_weights.get("10year", 0.5)

        weight_diff = short_weight - long_weight

        if weight_diff > self.REGIME_THRESHOLD:
            return "NOVEL_REGIME"
        elif weight_diff < -self.REGIME_THRESHOLD:
            return "HISTORICAL_REGIME"
        else:
            return "MIXED"

    def blend_recommendations(self, cohort_recs: Dict[str, Dict] = None) -> Dict:
        """
        Blend recommendations from all cohorts using current weights.

        Args:
            cohort_recs: Optional dict mapping cohort names to their recommendations.
                        If not provided, loads from files.

        Returns:
            Blended recommendations dict
        """
        if cohort_recs is None:
            cohort_recs = self.load_all_recommendations()

        if not cohort_recs:
            logger.warning("[janus] No recommendations to blend")
            return {"blended_recommendations": [], "contested_tickers": []}

        # Collect all tickers across cohorts
        ticker_recs: Dict[str, List[Tuple[str, Dict]]] = {}  # ticker -> [(cohort, rec)]

        for cohort, data in cohort_recs.items():
            recs = data.get("recommendations", [])
            for rec in recs:
                ticker = rec.get("ticker")
                if ticker:
                    if ticker not in ticker_recs:
                        ticker_recs[ticker] = []
                    ticker_recs[ticker].append((cohort, rec))

        blended = []
        contested = []

        for ticker, cohort_rec_list in ticker_recs.items():
            result = self._blend_ticker_recommendations(ticker, cohort_rec_list)
            blended.append(result)
            if result.get("contested"):
                contested.append(ticker)

        # Sort by blended conviction
        blended.sort(key=lambda x: x.get("conviction", 0), reverse=True)

        return {
            "blended_recommendations": blended,
            "contested_tickers": contested
        }

    def _blend_ticker_recommendations(
        self,
        ticker: str,
        cohort_rec_list: List[Tuple[str, Dict]]
    ) -> Dict:
        """
        Blend recommendations for a single ticker from multiple cohorts.

        Args:
            ticker: Stock ticker
            cohort_rec_list: List of (cohort_name, recommendation) tuples

        Returns:
            Blended recommendation dict
        """
        # Separate by direction
        longs = []
        shorts = []

        for cohort, rec in cohort_rec_list:
            weight = self.cohort_weights.get(cohort, 0)
            direction = rec.get("direction", "LONG").upper()
            conviction = rec.get("conviction", 50)

            entry = {
                "cohort": cohort,
                "weight": weight,
                "conviction": conviction,
                "agents": rec.get("agents", [])
            }

            if direction == "LONG":
                longs.append(entry)
            else:
                shorts.append(entry)

        # Calculate weighted conviction for each direction
        long_weighted = sum(e["conviction"] * e["weight"] for e in longs)
        short_weighted = sum(e["conviction"] * e["weight"] for e in shorts)

        # Check for disagreement
        contested = bool(longs and shorts)

        # Determine winning direction
        if long_weighted >= short_weighted:
            direction = "LONG"
            base_conviction = long_weighted
            opposing_conviction = short_weighted
        else:
            direction = "SHORT"
            base_conviction = short_weighted
            opposing_conviction = long_weighted

        # Reduce conviction by disagreement magnitude if contested
        if contested:
            disagreement_penalty = opposing_conviction * 0.5
            final_conviction = max(0, base_conviction - disagreement_penalty)
        else:
            final_conviction = base_conviction

        # Collect all contributing agents
        all_agents = []
        for cohort, rec in cohort_rec_list:
            all_agents.extend(rec.get("agents", []))
        all_agents = list(set(all_agents))  # Dedupe

        # Collect contributing cohorts
        contributing_cohorts = {
            cohort: {
                "conviction": rec.get("conviction"),
                "direction": rec.get("direction"),
                "weight": self.cohort_weights.get(cohort, 0)
            }
            for cohort, rec in cohort_rec_list
        }

        return {
            "ticker": ticker,
            "direction": direction,
            "conviction": round(final_conviction, 1),
            "contested": contested,
            "agents": all_agents,
            "cohort_breakdown": contributing_cohorts
        }

    def run_daily(self) -> Dict:
        """
        Execute full daily JANUS cycle:
        1. Load scored outcomes
        2. Update cohort weights
        3. Load and blend recommendations
        4. Determine regime signal
        5. Save outputs

        Returns:
            Today's JANUS output
        """
        logger.info("[janus] Starting daily run...")

        # Load and score
        outcomes = self.load_scored_outcomes()
        self.update_weights(outcomes)

        # Blend recommendations
        blend_result = self.blend_recommendations()

        # Get regime
        regime = self.regime_signal()

        # Build output
        output = {
            "date": date.today().isoformat(),
            "cohort_weights": {k: round(v, 4) for k, v in self.cohort_weights.items()},
            "regime": regime,
            "blended_recommendations": blend_result["blended_recommendations"],
            "contested_tickers": blend_result["contested_tickers"],
            "cohort_accuracy_30d": {
                k: {
                    "hit_rate": round(v["hit_rate"], 4),
                    "sharpe": round(v["sharpe"], 4)
                }
                for k, v in self.cohort_accuracy.items()
            },
            "generated_at": datetime.utcnow().isoformat()
        }

        # Save daily output
        self._save_daily(output)

        # Append to history
        self._append_history(output)

        logger.info(f"[janus] Daily run complete. Regime: {regime}")
        return output

    def _save_daily(self, output: Dict):
        """Save today's JANUS output."""
        try:
            with open(self.daily_file, "w") as f:
                json.dump(output, f, indent=2, default=str)
            logger.info(f"[janus] Saved daily output to {self.daily_file}")
        except Exception as e:
            logger.error(f"[janus] Error saving daily output: {e}")

    def _append_history(self, output: Dict):
        """Append today's output to rolling history."""
        try:
            history = []
            if self.history_file.exists():
                with open(self.history_file, "r") as f:
                    history = json.load(f)

            # Keep summary for history (not full recommendations)
            history_entry = {
                "date": output["date"],
                "cohort_weights": output["cohort_weights"],
                "regime": output["regime"],
                "cohort_accuracy_30d": output["cohort_accuracy_30d"],
                "num_recommendations": len(output["blended_recommendations"]),
                "num_contested": len(output["contested_tickers"])
            }

            history.append(history_entry)

            # Keep last 365 days
            history = history[-365:]

            with open(self.history_file, "w") as f:
                json.dump(history, f, indent=2, default=str)

            logger.info(f"[janus] Appended to history ({len(history)} entries)")
        except Exception as e:
            logger.error(f"[janus] Error appending to history: {e}")

    def get_history(self, days: int = 30) -> List[Dict]:
        """
        Get recent JANUS history for charting.

        Args:
            days: Number of days of history to return

        Returns:
            List of historical entries (newest last)
        """
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, "r") as f:
                history = json.load(f)
            return history[-days:]
        except Exception as e:
            logger.error(f"[janus] Error loading history: {e}")
            return []


def main():
    """Run JANUS daily cycle from command line."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    # Parse cohorts from command line if provided
    cohorts = None
    if len(sys.argv) > 1:
        cohorts = sys.argv[1].split(",")

    janus = Janus(cohorts=cohorts)
    result = janus.run_daily()

    print("\n" + "=" * 60)
    print("JANUS DAILY OUTPUT")
    print("=" * 60)
    print(f"Date: {result['date']}")
    print(f"Regime: {result['regime']}")
    print(f"\nCohort Weights:")
    for cohort, weight in result["cohort_weights"].items():
        acc = result["cohort_accuracy_30d"].get(cohort, {})
        print(f"  {cohort}: {weight:.1%} (hit_rate: {acc.get('hit_rate', 0):.1%}, sharpe: {acc.get('sharpe', 0):.2f})")

    print(f"\nBlended Recommendations: {len(result['blended_recommendations'])}")
    for rec in result["blended_recommendations"][:5]:  # Top 5
        contested_flag = " [CONTESTED]" if rec.get("contested") else ""
        print(f"  {rec['direction']} {rec['ticker']}: {rec['conviction']:.0f}%{contested_flag}")

    if result["contested_tickers"]:
        print(f"\nContested Tickers: {', '.join(result['contested_tickers'])}")


if __name__ == "__main__":
    main()
