"""
MiroFish Context Helper
=======================

Loads MiroFish simulation predictions and formats them for injection
into ATLAS agent prompts.

Usage:
    from agents.mirofish_context import get_mirofish_context

    # In build_analysis_prompt():
    mirofish_section = get_mirofish_context()
    if mirofish_section:
        prompt_parts.append(mirofish_section)
"""

import json
from pathlib import Path
from typing import Optional


STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "state"
PREDICTIONS_FILE = STATE_DIR / "mirofish_predictions.json"


def get_mirofish_context(max_predictions: int = 5, max_risks: int = 3) -> Optional[str]:
    """
    Load latest MiroFish predictions and format as prompt context.

    Args:
        max_predictions: Maximum number of consensus predictions to include
        max_risks: Maximum number of tail risks to include

    Returns:
        Formatted string for prompt injection, or None if no predictions
    """
    if not PREDICTIONS_FILE.exists():
        return None

    try:
        with open(PREDICTIONS_FILE) as f:
            predictions = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    if not predictions:
        return None

    latest = predictions[-1]

    # Build context section
    parts = [
        "",
        "### FORWARD-LOOKING CONTEXT (MiroFish Simulations)",
        f"Simulation Date: {latest.get('simulation_date', 'Unknown')[:10]}",
        f"Scenarios Simulated: {latest.get('scenarios_simulated', 0)}",
        f"Seed Summary: {latest.get('seed_summary', 'N/A')}",
        "",
        "**Consensus Predictions:**"
    ]

    for pred in latest.get("consensus_predictions", [])[:max_predictions]:
        event = pred.get("event", "Unknown event")
        prediction = pred.get("prediction", "N/A")
        confidence = pred.get("confidence", 0)
        agents = pred.get("simulated_agents_agreeing", 0)
        driver = pred.get("key_driver", "N/A")
        timeframe = pred.get("timeframe", "N/A")

        parts.append(f"- {event}")
        parts.append(f"  Prediction: {prediction}")
        parts.append(f"  Confidence: {confidence:.0%} ({agents} agents agreeing)")
        parts.append(f"  Key driver: {driver}")
        parts.append(f"  Timeframe: {timeframe}")

    # Tail risks
    tail_risks = latest.get("tail_risks", [])
    if tail_risks:
        parts.extend(["", "**Tail Risks to Monitor:**"])
        for risk in tail_risks[:max_risks]:
            event = risk.get("event", "Unknown risk")
            prob = risk.get("probability", 0)
            impact = risk.get("portfolio_impact", "N/A")
            hedge = risk.get("recommended_hedge", "N/A")

            parts.append(f"- {event} (probability: {prob:.0%})")
            parts.append(f"  Impact: {impact}")
            parts.append(f"  Hedge: {hedge}")

    # Highest conviction trade
    hct = latest.get("highest_conviction_trade")
    if hct:
        parts.extend(["", "**Simulated Highest Conviction Trade:**"])
        parts.append(f"- Ticker: {hct.get('ticker', 'N/A')}")
        parts.append(f"  Direction: {hct.get('direction', 'N/A')}")
        parts.append(f"  Reasoning: {hct.get('reasoning', 'N/A')}")
        parts.append(f"  Agents supporting: {hct.get('agents_supporting', 0)}")

    # Reflexive extremes
    reflexives = latest.get("reflexivity_signals", [])
    if reflexives:
        parts.extend(["", "**Reflexive Extremes Detected:**"])
        for signal in reflexives[:3]:
            if isinstance(signal, str):
                parts.append(f"- {signal}")
            elif isinstance(signal, dict):
                parts.append(f"- {signal.get('description', str(signal))}")

    parts.extend([
        "",
        "*These are simulations, not certainties. Weight alongside your analysis.*",
        ""
    ])

    return "\n".join(parts)


def has_mirofish_predictions() -> bool:
    """Check if MiroFish predictions are available."""
    if not PREDICTIONS_FILE.exists():
        return False
    try:
        with open(PREDICTIONS_FILE) as f:
            predictions = json.load(f)
        return bool(predictions)
    except:
        return False


def get_latest_hct() -> Optional[dict]:
    """Get just the highest conviction trade from latest simulation."""
    if not PREDICTIONS_FILE.exists():
        return None
    try:
        with open(PREDICTIONS_FILE) as f:
            predictions = json.load(f)
        if predictions:
            return predictions[-1].get("highest_conviction_trade")
    except:
        pass
    return None
