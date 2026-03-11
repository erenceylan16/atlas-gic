# ATLAS - Self-Improving AI Trading Agents

**Built by [General Intelligence Capital](https://generalintelligencecapital.com)**

[Karpathy's autoresearch](https://github.com/karpathy/autoresearch) applied to financial markets. The agent prompts are the weights. Sharpe ratio is the loss function. No GPU needed.

---

## What Is This?

ATLAS is a framework for autonomous AI trading agents that improve their own prompts through market feedback.

25 agents debate markets daily across 4 layers. Every recommendation is scored against real outcomes. The worst-performing agent gets its prompt rewritten. If performance improves, the git commit survives. If not, git revert.

The trained prompts are the product of 378 days of evolutionary optimisation against 18 months of real market data. They are proprietary. This repo contains the framework and the results.

---

## Architecture

### Layer 1 - Macro (10 agents)
Central bank, geopolitical, China, dollar, yield curve, commodities, volatility, emerging markets, news sentiment, institutional flow.

These agents set the regime. Risk on or risk off? What's the macro backdrop?

### Layer 2 - Sector Desks (7 agents)
Semiconductor, energy, biotech, consumer, industrials, financials, plus a Bloomberg-style relationship mapper that tracks supply chains, ownership, analyst coverage, and competitive dynamics for every name.

They take the macro regime from Layer 1 and identify the best names within each sector.

### Layer 3 - Superinvestors (4 agents)
- **Druckenmiller** - macro/momentum: what's the big asymmetric trade?
- **Aschenbrenner** - AI/compute: who benefits from the capex cycle?
- **Baker** - deep tech/biotech: who has real IP moats?
- **Ackman** - quality compounder: pricing power + FCF + catalyst?

They filter sector picks through different investment philosophies.

### Layer 4 - Decision (4 agents)
- **CRO** - adversarial risk officer: attacks every idea, finds correlated risks
- **Alpha Discovery** - finds names nobody else mentioned
- **Autonomous Execution** - converts signals to sized trades
- **CIO** - synthesises all prior layers, weighted by Darwinian agent scores, makes the final call

Each layer feeds into the next. The CIO only sees ideas that survived three rounds of analysis.

---

## The Autoresearch Loop

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Same pattern, different domain.

**Karpathy's version:**
- Agent modifies training code (train.py)
- 5-minute GPU training run
- Check validation loss
- Keep or revert

**Ours:**
- System identifies worst agent by rolling Sharpe
- Generates one targeted prompt modification
- Runs for 5 trading days
- Checks if agent's Sharpe improved
- Keep (git commit) or revert (git reset)

The agent prompts are the weights being optimised. Each trading day is one training iteration. A $20/month VM replaces the H100.

**Darwinian Weights:**
Each agent has a weight between 0.3 (minimum, near-silenced) and 2.5 (maximum, highly trusted). After each day, top quartile agents get weight x 1.05. Bottom quartile get weight x 0.95. The CIO proportionally weights agent input by these scores.

Over time, good agents get louder. Bad agents get quieter. The system learns who to trust.

---

## 18-Month Backtest Results

**Period:** September 2024 - March 2026 (378 trading days)

### Autoresearch Stats
- Prompt modifications attempted: 54
- Survived (kept): 16 (30%)
- Reverted: 37 (70%)
- Agents modified: volatility, semiconductor, china, news sentiment, emerging markets, financials, and others

### Performance
- Deployment phase return: **+22% in 173 days**
- Best individual pick: **AVGO at $152, held for +128%**

### Darwinian Agent Weights

The system learned which agents to trust. Macro-regime and quality-compounder agents rose to maximum weight. The system's own portfolio manager (CIO) was downweighted to the minimum — it discovered the orchestration bottleneck before we did.

Full agent weight details are proprietary.

### Equity Curve

![Equity Curve](results/equity_curve.png)

### Darwinian Weight Evolution

![Agent Weights](results/agent_weights.png)

---

## Key Insight

The orchestration layer matters as much as the intelligence layer.

Individual agents improved measurably through autoresearch - the financials agent's Sharpe improved from -4.14 to 0.45, emerging markets improved from -0.45 to -0.06, semiconductor improved from -0.26 to -0.06. But portfolio returns depend heavily on how agent signals are converted to sized positions.

**The lesson:** in any multi-agent system, the synthesis/decision layer is the bottleneck. Improving individual agent intelligence without improving orchestration yields diminishing returns.

---

## What's Included

- Framework architecture and pipeline structure
- Autoresearch loop design
- Backtest results, equity curve, agent weight evolution
- Example placeholder prompts (generic, not trained)
- Full methodology documentation

## What's NOT Included

- Trained agent prompts (proprietary - 378 days of evolutionary optimisation)
- CIO active management rules
- Agent scorecard data
- Live portfolio positions
- Infrastructure and deployment details

The trained prompts are the core intellectual property. They are the product of market-driven evolutionary selection.

---

## Now Running Live

The trained agents are deployed on real market data.

---

## Tech Stack

- **Agents:** Claude Sonnet (Anthropic API)
- **Data:** FMP, Finnhub, Polygon, FRED
- **Infrastructure:** Azure VM ($20/month)
- **Version Control:** Git feature branches for autoresearch tracking
- **Cost:** ~$50-80 for full 18-month backtest

---

## Contact

**Chris Worsey** - CEO & Technical Founder, General Intelligence Capital

chris@generalintelligencecapital.com

[generalintelligencecapital.com](https://generalintelligencecapital.com)
