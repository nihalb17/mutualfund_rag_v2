"""
guardrails.py - Phase 2
Pragmatic guardrails to detect out-of-scope queries, known schemes, and investment advice.
"""

import re

# Keywords that suggest a request for investment advice or recommendation
ADVICE_KEYWORDS = [
    r"\bshould\s+i\s+(?:invest|buy|sell|purchase|start)\b",
    r"\binvestment\s+advice\b",
    r"\bis\s+it\s+good\s+to\b",
    r"\best\s+fund\b",
    r"\brecommend\b",
    r"\btip\b",
]

# Known schemes from the architecture doc
KNOWN_SCHEMES_PATTERNS = {
    "axis_liquid": r"axis\s+liquid",
    "axis_elss": r"axis\s+elss|axis\s+tax\s+saver",
    "axis_flexi": r"axis\s+flexi\s+cap",
}

# Generic financial/mutual fund entity pattern to detect other funds
OTHER_FUND_KEYWORDS = [
    r"\bmirae\b", r"\bsbi\b", r"\bhdfc\b", r"\bicici\b", r"\buti\b", r"\bnippon\b",
    r"\btata\b", r"\bquant\b", r"\bparag\s+parikh\b", r"\bcanara\b", r"\bkotak\b"
]

def is_investment_advice(query: str) -> bool:
    """Detects if the user is asking for investment advice."""
    q_lower = query.lower()
    for pattern in ADVICE_KEYWORDS:
        if re.search(pattern, q_lower):
            return True
    return False

def is_unknown_scheme(query: str) -> bool:
    """
    Detects if the user is asking about a specific scheme NOT in our database.
    If they mention a known fund, it returns False.
    If they mention a competitor fund or a generic name not matching our 3, returns True.
    """
    q_lower = query.lower()
    
    # Check if any known scheme is mentioned
    is_known = False
    for pat in KNOWN_SCHEMES_PATTERNS.values():
        if re.search(pat, q_lower):
            is_known = True
            break
    
    # If no known scheme is mentioned, but other fund keywords are present, it's definitely unknown
    if not is_known:
        for pattern in OTHER_FUND_KEYWORDS:
            if re.search(pattern, q_lower):
                return True
        
        # If it's a mutual fund query but doesn't hit our 3, we'll let it pass to retrieval 
        # but the RAG engine will handle empty context.
    
    return False

def get_guardrail_response(query: str) -> str | None:
    """
    Returns a refusal message if a guardrail is triggered, else None.
    """
    if is_investment_advice(query):
        return "I am only here to provide information regarding mutual funds. I cannot provide investment advice, buy/sell recommendations, or opinions."
    
    if is_unknown_scheme(query):
        return "I don't have information regarding the scheme you are asking about. I can only provide details for Axis Liquid Direct Fund, Axis ELSS Tax Saver, and Axis Flexi Cap Fund."
    
    return None

if __name__ == "__main__":
    test_queries = [
        "What is the NAV of Axis Liquid Fund?",
        "Should I invest in Axis ELSS?",
        "Tell me about HDFC Liquid Fund",
        "What is the capital of France?"
    ]
    for q in test_queries:
        resp = get_guardrail_response(q)
        print(f"Query: {q}\nGuardrail: {resp}\n")
