"""
parser.py - Phase 1
Parses raw inner-text scraped from Groww mutual fund pages into structured
field dictionaries.  All regex patterns are anchored to known section labels
found on the Groww page layout.
"""

import re
from pathlib import Path
from typing import Optional

# -- Static scheme metadata ----------------------------------------------------
SCHEME_METADATA: dict[str, dict] = {
    "axis_liquid_direct_fund": {
        "scheme_name": "Axis Liquid Direct Fund Growth",
        "source_url": "https://groww.in/mutual-funds/axis-liquid-direct-fund-growth",
        "category": "Liquid Fund",
        "sub_category": "Debt",
        "fund_house": "Axis Mutual Fund",
        "lock_in": "None",
    },
    "axis_elss_tax_saver": {
        "scheme_name": "Axis ELSS Tax Saver Direct Plan Growth",
        "source_url": "https://groww.in/mutual-funds/axis-elss-tax-saver-direct-plan-growth",
        "category": "ELSS (Equity Linked Savings Scheme)",
        "sub_category": "Equity",
        "fund_house": "Axis Mutual Fund",
        "lock_in": "3 years (mandatory)",
    },
    "axis_flexi_cap_fund": {
        "scheme_name": "Axis Flexi Cap Fund Direct Growth",
        "source_url": "https://groww.in/mutual-funds/axis-flexi-cap-fund-direct-growth",
        "category": "Flexi Cap Fund",
        "sub_category": "Equity",
        "fund_house": "Axis Mutual Fund",
        "lock_in": "None",
    },
}

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


# -- Helper ---------------------------------------------------------------------
def _first(pattern: str, text: str, flags: int = 0) -> Optional[str]:
    """Returns the first capture group of a regex match, or None."""
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _extract_nav(text: str) -> str:
    # Groww shows: "NAV\n₹2,457.1234\nDD MMM YYYY" or "Current NAV ₹..."
    patterns = [
        r"Current NAV\s*[:\s]*₹?\s*([\d,]+\.?\d*)",
        r"\bNAV\b[^\n]*\n\s*₹?\s*([\d,]+\.?\d*)",
        r"NAV\s*₹\s*([\d,]+\.?\d*)",
    ]
    for pat in patterns:
        val = _first(pat, text, re.IGNORECASE)
        if val:
            return f"₹{val}"
    return "Not available"


def _extract_aum(text: str) -> str:
    # "Fund size\n₹xx,xxx Cr" or "AUM ₹xx,xxx Cr"
    patterns = [
        r"Fund\s*[Ss]ize\s*\n\s*₹?\s*([\d,]+(?:\.\d+)?)\s*Cr",
        r"AUM\s*[:\s]*₹?\s*([\d,]+(?:\.\d+)?)\s*Cr",
        r"Assets Under Management\s*[:\s]*₹?\s*([\d,]+(?:\.\d+)?)\s*Cr",
        r"₹\s*([\d,]+(?:\.\d+)?)\s*Cr\s*Fund\s*[Ss]ize",
    ]
    for pat in patterns:
        val = _first(pat, text, re.IGNORECASE)
        if val:
            return f"₹{val} Cr"
    return "Not available"


def _extract_expense_ratio(text: str) -> str:
    patterns = [
        r"Expense\s*[Rr]atio\s*\n\s*([\d.]+%?)",
        r"Expense\s*[Rr]atio[^\n]*\n.*?\n\s*([\d.]+%?)",
        r"Expense\s*[Rr]atio\s*[:\s]*([\d.]+%?)",
    ]
    for pat in patterns:
        val = _first(pat, text, re.IGNORECASE | re.DOTALL)
        if val and re.search(r"\d", val):
            val = val.strip()
            if "%" not in val:
                val += "%"
            return val
    return "Not available"


def _extract_min_sip(text: str) -> str:
    patterns = [
        r"Min\.\s*SIP\s*[Aa]mount\s*\n\s*₹?\s*([\d,]+)",
        r"Minimum\s*SIP\s*[:\s]*₹?\s*([\d,]+)",
        r"SIP\s*[Mm]in\w*\s*[:\s]*₹?\s*([\d,]+)",
        r"Min SIP\s*[:\n\s]*₹?\s*([\d,]+)",
    ]
    for pat in patterns:
        val = _first(pat, text, re.IGNORECASE)
        if val:
            return f"₹{val}"
    return "Not available"


def _extract_min_lumpsum(text: str) -> str:
    patterns = [
        r"Min\.\s*(?:Lumpsum|One[- ]?[Tt]ime)\s*[Ii]nvestment\s*\n\s*₹?\s*([\d,]+)",
        r"Minimum\s*(?:Lumpsum|Lump\s*sum)\s*[:\s]*₹?\s*([\d,]+)",
        r"Min(?:imum)?\s*Investment\s*[:\n\s]*₹?\s*([\d,]+)",
    ]
    for pat in patterns:
        val = _first(pat, text, re.IGNORECASE)
        if val:
            return f"₹{val}"
    return "Not available"


def _extract_exit_load(text: str) -> str:
    # Groww shows "Exit load" section then the actual text
    patterns = [
        r"Exit\s*[Ll]oad\s*\n\s*([^\n]+(?:\n[^\n]+){0,3})",
        r"Exit\s*[Ll]oad[:\s]+([^\n]+)",
    ]
    for pat in patterns:
        val = _first(pat, text, re.IGNORECASE)
        if val:
            return val.strip()
    return "Not available"


def _extract_stamp_duty(text: str) -> str:
    patterns = [
        r"Stamp\s*[Dd]uty[^:]*:\s*([\d.]+%[^\n]*)",
        r"Stamp\s*[Dd]uty\s*on\s*investment[:\s]*([\d.]+%[^\n]*)",
        r"Stamp\s*[Dd]uty\s*\n\s*([\d.]+%[^\n]*)",
    ]
    for pat in patterns:
        val = _first(pat, text, re.IGNORECASE)
        if val:
            return val.strip()
    return "0.005% (from July 1st, 2020)"   # Groww's standard stamp duty shown


def _extract_tax(text: str) -> str:
    patterns = [
        r"Tax\s*[Ii]mplication\s*\n([\s\S]{10,300}?)(?=\n[A-Z][a-z]|\Z)",
        r"(?:STCG|LTCG|Tax)[^\n]*\n([^\n]{10,200})",
    ]
    for pat in patterns:
        val = _first(pat, text, re.IGNORECASE)
        if val and len(val.strip()) > 5:
            return val.strip()
    return "Not available"


def _extract_risk(text: str) -> str:
    patterns = [
        r"(?:Risk|Riskometer)[:\s\n]*(Very\s*High|High|Moderately\s*High|Moderate|Moderately\s*Low|Low)",
        r"(Very\s*High\s*[Rr]isk|High\s*[Rr]isk|Moderate\s*[Rr]isk|Low\s*[Rr]isk)",
    ]
    for pat in patterns:
        val = _first(pat, text, re.IGNORECASE)
        if val:
            return val.strip()
    return "Not available"


def _extract_returns(text: str) -> dict:
    returns: dict[str, str] = {}
    # Patterns for "1Y\n12.34%" or "1 Year Return 12.34%"
    for period, labels in [
        ("1Y", [r"1\s*Y(?:ear)?\s*\n\s*([+-]?[\d.]+%?)", r"1\s*Year\s*Return[:\s]*([+-]?[\d.]+%?)"]),
        ("3Y", [r"3\s*Y(?:ear)?\s*\n\s*([+-]?[\d.]+%?)", r"3\s*Year\s*Return[:\s]*([+-]?[\d.]+%?)"]),
        ("5Y", [r"5\s*Y(?:ear)?\s*\n\s*([+-]?[\d.]+%?)", r"5\s*Year\s*Return[:\s]*([+-]?[\d.]+%?)"]),
    ]:
        for pat in labels:
            val = _first(pat, text, re.IGNORECASE)
            if val:
                if "%" not in val:
                    val += "%"
                returns[period] = val
                break

    return returns


def _extract_fund_managers(text: str) -> list[str]:
    """
    Groww shows fund managers as: "FirstName LastName\nMon YYYY - Present"
    We pick up lines that look like a person's name followed by a tenure.
    """
    managers: list[str] = []
    # Pattern: name line followed by "Mon YYYY - Present" or "Mon YYYY - Mon YYYY"
    pattern = re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\n\s*"
        r"([A-Z][a-z]{2}\s+\d{4}\s*[-]\s*(?:Present|\w+\s+\d{4}))",
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        entry = f"{m.group(1).strip()} ({m.group(2).strip()})"
        if entry not in managers:
            managers.append(entry)
    return managers[:5]  # cap at 5


def _extract_advanced_ratios(text: str) -> dict:
    ratios: dict[str, str] = {}
    ratio_patterns = {
        "Sharpe Ratio": r"Sharpe\s*[Rr]atio\s*[:\n\s]*([-\d.]+)",
        "Sortino Ratio": r"Sortino\s*[Rr]atio\s*[:\n\s]*([-\d.]+)",
        "Beta": r"\bBeta\b\s*[:\n\s]*([-\d.]+)",
        "Alpha": r"\bAlpha\b\s*[:\n\s]*([-\d.]+)",
        "Standard Deviation": r"(?:Std\.?\s*Dev\.?|Standard\s*Deviation)\s*[:\n\s]*([\d.]+)",
    }
    for name, pat in ratio_patterns.items():
        val = _first(pat, text, re.IGNORECASE)
        if val:
            ratios[name] = val
    return ratios


def _extract_investment_objective(text: str) -> str:
    patterns = [
        r"Investment\s*[Oo]bjective\s*\n([\s\S]{20,800}?)(?=\n[A-Z][a-z]|\Z)",
        r"About.*?\n([\s\S]{20,600}?)(?=Fund\s*[Mm]anager|Holdings|Returns|\Z)",
    ]
    for pat in patterns:
        val = _first(pat, text, re.IGNORECASE)
        if val and len(val.strip()) > 20:
            return val.strip()[:600]
    return "Not available"


def _extract_inception_date(text: str) -> str:
    patterns = [
        r"(?:Launch|Inception|Start)\s*[Dd]ate[:\s\n]*([\d]{1,2}\s+\w+\s+\d{4}|\w+\s+\d{4})",
        r"Since\s+(\w+\s+\d{4})",
    ]
    for pat in patterns:
        val = _first(pat, text, re.IGNORECASE)
        if val:
            return val.strip()
    return "Not available"


# -- Main parse function --------------------------------------------------------
def parse_scheme(scheme_key: str, raw_text: str) -> dict:
    """
    Parses raw page text for a scheme and returns a structured dict
    with all extractable fields.
    """
    meta = SCHEME_METADATA[scheme_key]
    fund_managers = _extract_fund_managers(raw_text)

    parsed = {
        **meta,
        "nav": _extract_nav(raw_text),
        "fund_size": _extract_aum(raw_text),
        "expense_ratio": _extract_expense_ratio(raw_text),
        "min_sip": _extract_min_sip(raw_text),
        "min_lumpsum": _extract_min_lumpsum(raw_text),
        "exit_load": _extract_exit_load(raw_text),
        "stamp_duty": _extract_stamp_duty(raw_text),
        "tax_implication": _extract_tax(raw_text),
        "risk": _extract_risk(raw_text),
        "returns": _extract_returns(raw_text),
        "fund_managers": fund_managers if fund_managers else ["Not available"],
        "advanced_ratios": _extract_advanced_ratios(raw_text),
        "investment_objective": _extract_investment_objective(raw_text),
        "inception_date": _extract_inception_date(raw_text),
    }

    return parsed


def parse_all_from_disk() -> dict[str, dict]:
    """Reads raw text files and parses all schemes."""
    results: dict[str, dict] = {}
    for scheme_key in SCHEME_METADATA:
        raw_path = RAW_DATA_DIR / f"{scheme_key}.txt"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw file not found: {raw_path}. Run scraper first."
            )
        raw_text = raw_path.read_text(encoding="utf-8")
        results[scheme_key] = parse_scheme(scheme_key, raw_text)
        print(f"[Parser] Parsed: {scheme_key}")
    return results


if __name__ == "__main__":
    import json

    data = parse_all_from_disk()
    for key, parsed in data.items():
        print(f"\n{'='*60}")
        print(f"Scheme: {key}")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
