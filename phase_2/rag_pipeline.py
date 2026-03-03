"""
rag_pipeline.py - Phase 2
Main orchestrator for the RAG flow.
"""

from phase_2.retriever import retrieve_context
from phase_2.guardrails import get_guardrail_response
from phase_2.prompt_builder import build_prompt
from phase_2.llm_client import generate_answer
import time

def query_rag_stream(query: str):
    """
    Async generator that yields JSON chunks for streaming. Stateless version.
    """
    from phase_2.llm_client import generate_answer_stream
    import json

    # 1. Input Guardrails
    refusal = get_guardrail_response(query)
    if refusal:
        yield json.dumps({"answer": refusal, "citations": [], "guardrail_triggered": True}) + "\n"
        return

    # 2. Retrieval (No contextualization)
    retrieved_chunks = retrieve_context(query)
    
    # 3. Prompt Construction (No history)
    prompt = build_prompt(query, retrieved_chunks)

    # 4. Stream Answer
    full_answer = []
    for chunk_text in generate_answer_stream(prompt):
        full_answer.append(chunk_text)
        yield json.dumps({"chunk": chunk_text}) + "\n"

    # 5. Final Metadata
    answer = "".join(full_answer)
    unknown_phrases = ["don't have an answer", "do not have an answer", "not mentioned in the context"]
    is_unknown = any(phrase in answer.lower() for phrase in unknown_phrases)
    
    citations = []
    if not is_unknown:
        seen_urls = set()
        for chunk in retrieved_chunks:
            url = chunk['metadata'].get('source_url')
            if url and url not in seen_urls:
                citations.append(url)
                seen_urls.add(url)

    yield json.dumps({
        "citations": citations, 
        "guardrail_triggered": False,
        "done": True
    }) + "\n"

def query_rag(query: str) -> dict:
    """
    Executes the full RAG pipeline (Stateless).
    """
    start_time = time.time()
    
    # 1. Input Guardrails
    refusal = get_guardrail_response(query)
    if refusal:
        return {
            "answer": refusal,
            "citations": [],
            "guardrail_triggered": True
        }

    # 2. Retrieval
    retrieved_chunks = retrieve_context(query)
    
    # 3. Prompt Construction
    prompt = build_prompt(query, retrieved_chunks)

    # 4. LLM Generation
    t1 = time.time()
    answer = generate_answer(prompt)
    t2 = time.time()
    print(f"  [Profile] Generation: {t2 - t1:.2f}s")

    # 5. Unknown handle & Citations
    unknown_phrases = ["don't have an answer", "do not have an answer", "not mentioned in the context"]
    is_unknown = any(phrase in answer.lower() for phrase in unknown_phrases)
    
    citations = []
    if not is_unknown:
        seen_urls = set()
        for chunk in retrieved_chunks:
            url = chunk['metadata'].get('source_url')
            if url and url not in seen_urls:
                citations.append(url)
                seen_urls.add(url)

    print(f"  [Profile] Total Time: {time.time() - start_time:.2f}s")
    return {
        "answer": answer,
        "citations": citations,
        "guardrail_triggered": False
    }

if __name__ == "__main__":
    # Interactive test loop
    print("Mutual Fund RAG Chatbot (Phase 2 REPL)")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        result = query_rag(user_input)
        print(f"\nBot: {result['answer']}")
        if result['citations']:
            print("\nSources:")
            for url in result['citations']:
                print(f"- {url}")
        print("-" * 40)
