"""
prompt_builder.py - Phase 2
Assembles the system prompt, retrieved context, and user query.
"""

SYSTEM_PROMPT_TEMPLATE = """
You are a factual Mutual Fund information assistant. Your only job is to answer
questions about the following three Axis Mutual Fund schemes based strictly on
the provided context. 

Axis Mutual Fund Schemes covered:
1. Axis Liquid Direct Fund Growth
2. Axis ELSS Tax Saver Direct Plan Growth
3. Axis Flexi Cap Fund Direct Growth

Rules:
1. ONLY answer from the provided context. Do not use any outside knowledge.
2. Do not give investment advice, buy/sell recommendations, or opinions.
3. If the context does not contain the answer, say: "I don't have an answer to the question you are asking."
4. Always cite only the source URLs from which the answer was derived.
5. Present only facts — numbers, names, dates, percentages — as stated in context.
6. If the user asks for comparison, compare only the data available in the context.

Context:
{retrieved_context}

User Question: {user_query}

Answer (facts only, cite sources):
"""

def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Combines chunks into the final prompt for Gemini.
    """
    print(f"[PromptBuilder] Building prompt with {len(retrieved_chunks)} chunks")
    
    if not retrieved_chunks:
        context_text = "No relevant context found in the database for this query."
        print(f"[PromptBuilder] WARNING: No chunks retrieved for query: '{query}'")
    else:
        # Format chunks into a readable context block
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk['metadata'].get('source_url', 'N/A')
            text = chunk['text']
            context_parts.append(f"[{i}] Source: {source}\nContent: {text}")
        context_text = "\n\n".join(context_parts)

    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        retrieved_context=context_text,
        user_query=query
    )
    
    return prompt

if __name__ == "__main__":
    # Quick test
    sample_chunks = [
        {
            "text": "The expense ratio of Axis Liquid Fund is 0.15%.",
            "metadata": {"source_url": "https://groww.in/axis-liquid"}
        }
    ]
    print(build_prompt("What is the expense ratio?", sample_chunks))
