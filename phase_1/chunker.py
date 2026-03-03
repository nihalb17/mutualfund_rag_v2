"""
chunker.py - Phase 1
Converts the structured parsed dict for each scheme into a flat list of
text chunks.  Each chunk is a semantically coherent unit (one field / group
of related fields) with associated metadata for ChromaDB storage.

Metadata keys on every chunk:
  - scheme_name   : Full human-readable fund name
  - source_url    : Groww page URL (used as citation link)
  - field_category: Short snake_case tag (e.g. "nav", "expense_ratio")
"""

import json
from pathlib import Path

CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunks"


def _fmt(value) -> str:
    """Stringify a value neatly."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def build_chunks(parsed: dict) -> list[dict]:
    """
    Given a parsed scheme dict, return a list of chunk dicts each with keys:
      - text      : The plain-text chunk content (what gets embedded)
      - metadata  : dict with scheme_name, source_url, field_category
    """
    name = parsed["scheme_name"]
    url  = parsed["source_url"]

    def meta(field_category: str) -> dict:
        return {
            "scheme_name": name,
            "source_url": url,
            "field_category": field_category,
        }

    chunks: list[dict] = []

    # -- 1. Fund overview / metadata ------------------------------------------
    overview_parts = [
        f"Scheme Name: {name}",
        f"Fund House: {parsed.get('fund_house', 'N/A')}",
        f"Category: {parsed.get('category', 'N/A')}",
        f"Sub-Category: {parsed.get('sub_category', 'N/A')}",
        f"Inception Date: {parsed.get('inception_date', 'N/A')}",
        f"Lock-in Period: {parsed.get('lock_in', 'N/A')}",
        f"Risk Level: {parsed.get('risk', 'N/A')}",
    ]
    chunks.append({
        "text": "\n".join(overview_parts),
        "metadata": meta("overview"),
    })

    # -- 2. NAV ---------------------------------------------------------------
    chunks.append({
        "text": f"Net Asset Value (NAV) of {name}: {parsed.get('nav', 'N/A')}",
        "metadata": meta("nav"),
    })

    # -- 3. Fund size / AUM ---------------------------------------------------
    chunks.append({
        "text": f"Fund Size (AUM) of {name}: {parsed.get('fund_size', 'N/A')}",
        "metadata": meta("fund_size"),
    })

    # -- 4. Expense ratio -----------------------------------------------------
    chunks.append({
        "text": f"Expense Ratio of {name}: {parsed.get('expense_ratio', 'N/A')}",
        "metadata": meta("expense_ratio"),
    })

    # -- 5. Minimum investments -----------------------------------------------
    inv_text = (
        f"Minimum Investments for {name}:\n"
        f"  Minimum SIP Amount: {parsed.get('min_sip', 'N/A')}\n"
        f"  Minimum Lump-sum Investment: {parsed.get('min_lumpsum', 'N/A')}"
    )
    chunks.append({
        "text": inv_text,
        "metadata": meta("minimum_investment"),
    })

    # -- 6. Exit load, stamp duty, tax ----------------------------------------
    charges_text = (
        f"Exit Load, Stamp Duty and Tax for {name}:\n"
        f"  Exit Load: {parsed.get('exit_load', 'N/A')}\n"
        f"  Stamp Duty on Investment: {parsed.get('stamp_duty', 'N/A')}\n"
        f"  Tax Implication: {parsed.get('tax_implication', 'N/A')}"
    )
    chunks.append({
        "text": charges_text,
        "metadata": meta("exit_load_tax"),
    })

    # -- 7. Returns -----------------------------------------------------------
    returns: dict = parsed.get("returns", {})
    if returns:
        return_lines = [f"Returns for {name}:"]
        for period, val in returns.items():
            return_lines.append(f"  {period} Return: {val}")
        chunks.append({
            "text": "\n".join(return_lines),
            "metadata": meta("returns"),
        })

    # -- 8. Fund managers -----------------------------------------------------
    managers = parsed.get("fund_managers", [])
    if managers and managers != ["Not available"]:
        mgr_text = (
            f"Fund Manager(s) of {name}:\n"
            + "\n".join(f"  - {m}" for m in managers)
        )
        chunks.append({
            "text": mgr_text,
            "metadata": meta("fund_manager"),
        })

    # -- 9. Advanced ratios ---------------------------------------------------
    ratios: dict = parsed.get("advanced_ratios", {})
    if ratios:
        ratio_text = (
            f"Advanced Ratios for {name}:\n"
            + "\n".join(f"  {k}: {v}" for k, v in ratios.items())
        )
        chunks.append({
            "text": ratio_text,
            "metadata": meta("advanced_ratios"),
        })

    # -- 10. Investment objective / About -------------------------------------
    objective = parsed.get("investment_objective", "")
    if objective and objective != "Not available":
        chunks.append({
            "text": f"Investment Objective of {name}:\n{objective}",
            "metadata": meta("investment_objective"),
        })

    return chunks


def chunk_all(parsed_data: dict[str, dict]) -> list[dict]:
    """Builds and saves chunks for all schemes."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []

    for scheme_key, parsed in parsed_data.items():
        chunks = build_chunks(parsed)
        all_chunks.extend(chunks)

        # Save per-scheme chunks for debugging
        out_path = CHUNKS_DIR / f"{scheme_key}_chunks.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        print(f"[Chunker] {scheme_key}: {len(chunks)} chunks -> {out_path}")

    # Save combined chunks
    combined_path = CHUNKS_DIR / "all_chunks.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"[Chunker] Total: {len(all_chunks)} chunks -> {combined_path}")
    return all_chunks


if __name__ == "__main__":
    from parser import parse_all_from_disk

    parsed_data = parse_all_from_disk()
    chunks = chunk_all(parsed_data)
    print(f"\nSample chunk:\n{chunks[0]['text']}\n")
