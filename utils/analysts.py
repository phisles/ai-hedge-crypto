

print("\n📦 Initializing ANALYST_CONFIG...")

from agents.ben_graham import ben_graham_agent
print("✅ ben_graham_agent loaded")

from agents.bill_ackman import bill_ackman_agent
print("✅ bill_ackman_agent loaded")

from agents.cathie_wood import cathie_wood_agent
print("✅ cathie_wood_agent loaded")

from agents.charlie_munger import charlie_munger_agent
print("✅ charlie_munger_agent loaded")

from agents.fundamentals import fundamentals_agent
print("✅ fundamentals_agent loaded")

from agents.michael_burry import michael_burry_agent
print("✅ michael_burry_agent loaded")

from agents.phil_fisher import phil_fisher_agent
print("✅ phil_fisher_agent loaded")

from agents.peter_lynch import peter_lynch_agent
print("✅ peter_lynch_agent loaded")

from agents.sentiment import sentiment_agent
print("✅ sentiment_agent loaded")

from agents.stanley_druckenmiller import stanley_druckenmiller_agent
print("✅ stanley_druckenmiller_agent loaded")

from agents.technicals import technical_analyst_agent
print("✅ technical_analyst_agent loaded")

from agents.valuation import valuation_agent
print("✅ valuation_agent loaded")

from agents.warren_buffett import warren_buffett_agent
print("✅ warren_buffett_agent loaded")

# Define analyst configuration - single source of truth
ANALYST_CONFIG = {
    "ben_graham": {
        "display_name": "Ben Graham",
        "agent_func": ben_graham_agent,
        "order": 0,
    },
    "bill_ackman": {
        "display_name": "Bill Ackman",
        "agent_func": bill_ackman_agent,
        "order": 1,
    },
    "cathie_wood": {
        "display_name": "Cathie Wood",
        "agent_func": cathie_wood_agent,
        "order": 2,
    },
    "charlie_munger": {
        "display_name": "Charlie Munger",
        "agent_func": charlie_munger_agent,
        "order": 3,
    },
    "michael_burry": {
        "display_name": "Michael Burry",
        "agent_func": michael_burry_agent,
        "order": 4,
    },
    "peter_lynch": {
        "display_name": "Peter Lynch",
        "agent_func": peter_lynch_agent,
        "order": 5,
    },
    "phil_fisher": {
        "display_name": "Phil Fisher",
        "agent_func": phil_fisher_agent,
        "order": 6,
    },
    "stanley_druckenmiller": {
        "display_name": "Stanley Druckenmiller",
        "agent_func": stanley_druckenmiller_agent,
        "order": 7,
    },
    "warren_buffett": {
        "display_name": "Warren Buffett",
        "agent_func": warren_buffett_agent,
        "order": 8,
    },
    "technical_analyst": {
        "display_name": "Technical Analyst",
        "agent_func": technical_analyst_agent,
        "order": 9,
    },
    "fundamentals_analyst": {
        "display_name": "Fundamentals Analyst",
        "agent_func": fundamentals_agent,
        "order": 10,
    },
    "sentiment_analyst": {
        "display_name": "Sentiment Analyst",
        "agent_func": sentiment_agent,
        "order": 11,
    },
    "valuation_analyst": {
        "display_name": "Valuation Analyst",
        "agent_func": valuation_agent,
        "order": 12,
    },
}

# Derive ANALYST_ORDER from ANALYST_CONFIG for backwards compatibility
ANALYST_ORDER = [(config["display_name"], key) for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])]


def get_analyst_nodes():
    """Get the mapping of analyst keys to their (node_name, agent_func) tuples."""
    return {key: (f"{key}_agent", config["agent_func"]) for key, config in ANALYST_CONFIG.items()}