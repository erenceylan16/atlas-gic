"""
MiroFish Financial Seed Generator
=================================

Generates comprehensive market intelligence briefings from ATLAS data
for use as seed material in swarm simulations.

Pulls from:
1. FMP API - current prices for key assets
2. FRED - macro indicators
3. ATLAS agent debates - what our agents think
4. News sentiment - recent headlines
5. Portfolio positions - current exposure
6. Upcoming catalysts - earnings, FOMC, CPI
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ANTHROPIC_API_KEY, FMP_API_KEY, FRED_API_KEY
from data.macro_client import MacroClient
from data.price_client import PriceClient

logger = logging.getLogger(__name__)

# Directories
STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "state"
MIROFISH_DIR = Path(__file__).resolve().parent.parent / "data" / "mirofish"
SEEDS_DIR = MIROFISH_DIR / "seeds"


def ensure_dirs():
    """Ensure directories exist."""
    MIROFISH_DIR.mkdir(parents=True, exist_ok=True)
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)


def get_key_prices() -> Dict[str, Any]:
    """Fetch current prices for key assets."""
    prices = PriceClient()

    tickers = {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "TLT": "20Y Treasury Bonds",
        "GLD": "Gold",
        "XLE": "Energy Sector",
        "XLF": "Financials",
        "USO": "Oil (WTI proxy)",
        "UUP": "US Dollar",
        "VXX": "VIX (volatility)",
        "HYG": "High Yield Bonds",
        "EEM": "Emerging Markets",
        "FXI": "China (CSI 300)",
    }

    result = {}
    for ticker, name in tickers.items():
        try:
            price = prices.get_current_price(ticker)
            if price:
                result[ticker] = {"name": name, "price": price}
        except Exception as e:
            logger.warning(f"Failed to get price for {ticker}: {e}")

    return result


def get_macro_snapshot() -> Dict[str, Any]:
    """Fetch macro indicators from FRED."""
    macro = MacroClient()

    try:
        snapshot = macro.get_macro_snapshot()
        return {
            "fed_funds_rate": snapshot.get("fed_funds_rate"),
            "m2_yoy_change": snapshot.get("m2_yoy_change"),
            "yield_curve_10y_2y": snapshot.get("yield_curve_10y_2y"),
            "vix": snapshot.get("vix"),
            "cpi_yoy": snapshot.get("cpi_yoy"),
            "unemployment_rate": snapshot.get("unemployment_rate"),
            "pmi_manufacturing": snapshot.get("pmi_manufacturing"),
            "treasury_10y": snapshot.get("treasury_10y"),
            "treasury_2y": snapshot.get("treasury_2y"),
            "high_yield_spread": snapshot.get("high_yield_spread"),
        }
    except Exception as e:
        logger.warning(f"Failed to get macro data: {e}")
        return {}


def get_agent_debates() -> List[Dict]:
    """Load recent agent debate excerpts."""
    debates = []

    agent_files = [
        ("druckenmiller", "Macro Strategist"),
        ("ackman", "Activist Value"),
        ("aschenbrenner", "AI/Tech Focus"),
        ("baker", "Quant Value"),
        ("semiconductor", "Semiconductor Desk"),
        ("energy", "Energy Desk"),
        ("biotech", "Biotech Desk"),
        ("bond", "Bond Desk"),
    ]

    for desk, name in agent_files:
        brief_file = STATE_DIR / f"{desk}_briefs.json"
        if brief_file.exists():
            try:
                with open(brief_file) as f:
                    briefs = json.load(f)
                if briefs and isinstance(briefs, list) and len(briefs) > 0:
                    latest = briefs[-1]
                    debates.append({
                        "agent": name,
                        "desk": desk,
                        "timestamp": latest.get("analyzed_at", latest.get("timestamp", "")),
                        "headline": latest.get("headline", ""),
                        "signal": latest.get("portfolio_tilt", latest.get("signal", "")),
                        "conviction": latest.get("conviction_level", latest.get("confidence", 0)),
                        "brief": latest.get("brief_for_cio", latest.get("brief", ""))[:500],
                    })
            except Exception as e:
                logger.debug(f"Could not load {desk} briefs: {e}")

    return debates


def get_portfolio_positions() -> Dict[str, Any]:
    """Load current portfolio positions."""
    positions_file = STATE_DIR / "positions.json"
    if not positions_file.exists():
        return {"positions": [], "total_value": 0, "cash_pct": 100}

    try:
        with open(positions_file) as f:
            data = json.load(f)

        # Format positions - ATLAS uses a list format
        positions = []
        total_value = data.get("portfolio_value", 1000000)
        cash_balance = data.get("cash_balance", 0)

        raw_positions = data.get("positions", [])
        if isinstance(raw_positions, list):
            for pos in raw_positions:
                if isinstance(pos, dict):
                    entry_price = pos.get("entry_price", 0)
                    current_price = pos.get("current_price", entry_price)
                    pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                    positions.append({
                        "ticker": pos.get("ticker", "???"),
                        "shares": pos.get("shares", 0),
                        "avg_cost": entry_price,
                        "current_price": current_price,
                        "pnl_pct": pnl_pct,
                        "weight": pos.get("allocation_pct", 0),
                    })

        cash_pct = (cash_balance / total_value * 100) if total_value > 0 else 100

        return {
            "positions": positions,
            "total_value": total_value,
            "cash_pct": cash_pct,
        }
    except Exception as e:
        logger.warning(f"Failed to load positions: {e}")
        return {"positions": [], "total_value": 0, "cash_pct": 100}


def get_upcoming_catalysts() -> List[Dict]:
    """Get upcoming market catalysts."""
    # Static list for now - could be enhanced with API
    today = datetime.now()

    catalysts = [
        {
            "event": "FOMC Meeting",
            "date": "2026-04-02",
            "impact": "HIGH",
            "description": "Fed rate decision and Powell press conference"
        },
        {
            "event": "CPI Release",
            "date": "2026-03-26",
            "impact": "HIGH",
            "description": "February inflation data"
        },
        {
            "event": "Nonfarm Payrolls",
            "date": "2026-04-04",
            "impact": "HIGH",
            "description": "March jobs report"
        },
        {
            "event": "NVDA Earnings",
            "date": "2026-04-15",
            "impact": "HIGH",
            "description": "AI bellwether earnings"
        },
        {
            "event": "Q1 GDP",
            "date": "2026-04-28",
            "impact": "MEDIUM",
            "description": "First estimate of Q1 growth"
        },
    ]

    # Filter to upcoming events
    upcoming = []
    for c in catalysts:
        try:
            event_date = datetime.strptime(c["date"], "%Y-%m-%d")
            if event_date >= today:
                days_until = (event_date - today).days
                c["days_until"] = days_until
                upcoming.append(c)
        except:
            pass

    return sorted(upcoming, key=lambda x: x.get("days_until", 999))[:10]


def assess_market_regime(macro: Dict, prices: Dict) -> Dict:
    """Assess current market regime."""
    vix = macro.get("vix", 20)
    yc = macro.get("yield_curve_10y_2y", 0)
    hy_spread = macro.get("high_yield_spread", 4)

    # Regime assessment
    if vix > 30:
        volatility_regime = "HIGH_VOLATILITY"
    elif vix > 20:
        volatility_regime = "ELEVATED"
    elif vix < 15:
        volatility_regime = "COMPLACENT"
    else:
        volatility_regime = "NORMAL"

    if yc < -0.5:
        yield_curve = "DEEPLY_INVERTED"
    elif yc < 0:
        yield_curve = "INVERTED"
    elif yc > 1:
        yield_curve = "STEEP"
    else:
        yield_curve = "FLAT"

    if hy_spread > 6:
        credit = "STRESSED"
    elif hy_spread > 4.5:
        credit = "TIGHT"
    else:
        credit = "NORMAL"

    # Overall regime
    if volatility_regime == "HIGH_VOLATILITY" and credit == "STRESSED":
        overall = "CRISIS"
    elif volatility_regime in ["HIGH_VOLATILITY", "ELEVATED"] and yield_curve in ["INVERTED", "DEEPLY_INVERTED"]:
        overall = "RISK_OFF"
    elif volatility_regime == "COMPLACENT" and credit == "NORMAL":
        overall = "RISK_ON"
    else:
        overall = "TRANSITION"

    return {
        "overall": overall,
        "volatility": volatility_regime,
        "yield_curve": yield_curve,
        "credit": credit,
        "vix_level": vix,
    }


def format_seed_document(
    prices: Dict,
    macro: Dict,
    debates: List[Dict],
    portfolio: Dict,
    catalysts: List[Dict],
    regime: Dict
) -> str:
    """Format all data into a comprehensive seed document."""

    today = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    doc = f"""
================================================================================
MARKET INTELLIGENCE BRIEFING — {today}
================================================================================

MARKET REGIME: {regime['overall']}
- Volatility: {regime['volatility']} (VIX: {regime['vix_level']:.1f})
- Yield Curve: {regime['yield_curve']}
- Credit: {regime['credit']}

================================================================================
CURRENT MARKET STATE
================================================================================

"""

    # Prices
    doc += "KEY ASSET PRICES:\n"
    for ticker, data in prices.items():
        doc += f"  {ticker} ({data['name']}): ${data['price']:.2f}\n"

    doc += f"""
MACRO INDICATORS:
  Fed Funds Rate: {macro.get('fed_funds_rate', 'N/A')}%
  10Y Treasury: {macro.get('treasury_10y', 'N/A')}%
  2Y Treasury: {macro.get('treasury_2y', 'N/A')}%
  Yield Curve (10Y-2Y): {macro.get('yield_curve_10y_2y', 'N/A')} bps
  VIX: {macro.get('vix', 'N/A')}
  CPI YoY: {macro.get('cpi_yoy', 'N/A')}%
  Unemployment: {macro.get('unemployment_rate', 'N/A')}%
  High Yield Spread: {macro.get('high_yield_spread', 'N/A')} bps
  M2 YoY Change: {macro.get('m2_yoy_change', 'N/A')}%

================================================================================
KEY ACTORS AND THEIR CURRENT POSITIONS
================================================================================

INSTITUTIONAL AGENTS (Our AI Analysts):
"""

    for debate in debates:
        conv = debate.get('conviction', 0)
        if isinstance(conv, float):
            conv_str = f"{conv:.0%}"
        else:
            conv_str = str(conv)

        doc += f"""
  {debate['agent']} (Conviction: {conv_str}):
    Signal: {debate.get('signal', 'N/A')}
    View: {debate.get('headline', 'No headline')}
    Brief: {debate.get('brief', 'N/A')[:200]}...
"""

    doc += """
================================================================================
CURRENT PORTFOLIO EXPOSURE
================================================================================
"""

    doc += f"Total Value: ${portfolio.get('total_value', 0):,.0f}\n"
    doc += f"Cash: {portfolio.get('cash_pct', 0):.1f}%\n\n"
    doc += "Positions:\n"

    for pos in portfolio.get("positions", [])[:10]:
        doc += f"  {pos['ticker']}: {pos.get('weight', 0):.1f}% "
        doc += f"(P&L: {pos.get('pnl_pct', 0):+.1f}%)\n"

    doc += """
================================================================================
UPCOMING CATALYSTS
================================================================================
"""

    for cat in catalysts[:7]:
        doc += f"  [{cat.get('impact', 'MED')}] {cat['event']} — {cat['date']} "
        doc += f"({cat.get('days_until', '?')} days)\n"
        doc += f"       {cat.get('description', '')}\n"

    doc += """
================================================================================
SIMULATION REQUEST
================================================================================

Simulate how the following market participants will interact over the next
30 days given this starting state:

AGENT TYPES TO SIMULATE:
1. HEDGE FUND MANAGERS:
   - Macro Fund (Druckenmiller style): Concentrated bets on Fed policy, liquidity
   - Long/Short Equity: Bottom-up stock picking, pair trades
   - Quant Fund: Systematic signals, factor exposure, momentum
   - Activist Fund: Concentrated positions, corporate engagement
   - CTA/Trend Following: Pure momentum, ignores fundamentals

2. INSTITUTIONAL INVESTORS:
   - Pension Fund: Long-term allocation, liability matching, slow to move
   - Endowment: Alternative assets, illiquid positions
   - Insurance Company: Fixed income focused, regulatory constraints
   - Sovereign Wealth Fund: Very long-term, contrarian at extremes

3. SELL-SIDE & MARKET MAKERS:
   - Investment Bank Trader: Flow information, market making
   - Sell-Side Analyst: Publishes price targets, rating changes
   - Options Market Maker: Delta hedging, gamma exposure

4. CORPORATE ACTORS:
   - Corporate Treasurer: Debt issuance, buyback decisions, FX hedging
   - CEO/CFO: M&A, guidance, insider transactions

5. POLICY ACTORS:
   - Fed Governor: Data dependent, dual mandate
   - Treasury Official: Debt management, market stability

6. RETAIL:
   - Retail Trader: Follows social sentiment, options heavy, FOMO
   - Financial Media: Amplifies narratives, creates feedback loops

7. INFORMATION AGENTS:
   - Business Journalist: Breaks news, shapes narrative
   - Social Media Influencer: Viral takes, retail following

FOCUS AREAS:
1. Oil price trajectory and geopolitical premium
2. Tech/AI sector — does the narrative hold or break?
3. Bond market — flight to quality or inflation fears?
4. Fed response — do they cut, hold, or warn?
5. Reflexive feedback loops — where are cascades forming?

OUTPUT REQUIRED:
- Round-by-round agent actions and reasoning
- Price impact estimates after each round
- Identification of reflexive feedback loops
- Consensus predictions across scenarios
- Tail risk scenarios and their probability
- Highest conviction trade signals
"""

    return doc


def generate_seed() -> str:
    """Main entry point - generate complete seed document."""
    ensure_dirs()

    logger.info("[MiroFish Seed] Generating financial seed document...")

    # Gather all data
    prices = get_key_prices()
    macro = get_macro_snapshot()
    debates = get_agent_debates()
    portfolio = get_portfolio_positions()
    catalysts = get_upcoming_catalysts()
    regime = assess_market_regime(macro, prices)

    # Format document
    doc = format_seed_document(prices, macro, debates, portfolio, catalysts, regime)

    # Save to file
    today = datetime.now().strftime("%Y-%m-%d")
    seed_file = SEEDS_DIR / f"seed_{today}.txt"
    with open(seed_file, "w") as f:
        f.write(doc)

    logger.info(f"[MiroFish Seed] Saved to {seed_file}")

    # Also save structured data
    data_file = SEEDS_DIR / f"seed_{today}.json"
    with open(data_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "prices": prices,
            "macro": macro,
            "debates": debates,
            "portfolio": portfolio,
            "catalysts": catalysts,
            "regime": regime,
        }, f, indent=2, default=str)

    return doc


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    parser = argparse.ArgumentParser(description="MiroFish Financial Seed Generator")
    parser.add_argument("--print", action="store_true", help="Print seed to stdout")
    args = parser.parse_args()

    doc = generate_seed()

    if args.print:
        print(doc)
    else:
        print(f"Seed saved to {SEEDS_DIR}")


if __name__ == "__main__":
    main()
