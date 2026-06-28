"""
darkclaw/core/rag_extractor.py

Post-query RAG extraction pipeline.

After every successful agent response, this runs asynchronously to:
  1. Extract named entities + key facts from (query, response) text
  2. Ingest them into the agent's memory graph
  3. Add the full response text to the vector store for future retrieval

This builds up the knowledge graph organically from real usage —
every conversation makes the system smarter for the next one.

Design: runs as a fire-and-forget asyncio task; never blocks the
main response path; failures are swallowed silently.
"""
import asyncio
import hashlib
import re
import time
from typing import Optional


# ── Lightweight entity / fact extraction (no LLM needed) ──────────────

# Common stop words to ignore when extracting potential entities
_STOP = frozenset({
    "the","a","an","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","should",
    "could","may","might","shall","can","need","dare","ought",
    "to","of","in","for","on","with","at","by","from","as","into",
    "through","during","including","until","against","among","throughout",
    "what","which","who","this","that","these","those","it","its",
    "they","them","their","he","she","we","you","i","me","my","your",
    "how","why","when","where","if","but","or","and","so","yet","nor",
    "not","no","yes","ok","okay","please","thanks","thank",
})


def _extract_entities(text: str) -> list[str]:
    """
    Fast heuristic entity extraction:
    - CamelCase words
    - UPPER_CASE tokens
    - Quoted strings
    - Multi-word capitalized phrases (up to 3 words)
    """
    entities = set()

    # CamelCase / PascalCase
    for m in re.finditer(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text):
        entities.add(m.group())

    # UPPER_CASE identifiers
    for m in re.finditer(r'\b[A-Z][A-Z_]{2,}\b', text):
        entities.add(m.group())

    # Quoted strings (short ones are likely proper nouns)
    for m in re.finditer(r'"([^"]{3,40})"', text):
        entities.add(m.group(1).strip())

    # Capitalized multi-word phrases
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b', text):
        phrase = m.group(1)
        words = phrase.lower().split()
        if not all(w in _STOP for w in words):
            entities.add(phrase)

    return [e for e in entities if len(e) > 2 and e.lower() not in _STOP]


def _extract_relations(query: str, response: str) -> list[tuple[str, str, str]]:
    """
    Lightweight SVO-style triple extraction from response text.
    Looks for patterns like:
      "<entity> is/are/was <value>"
      "<entity> uses/runs/supports <value>"
    Returns [(subject, predicate, object), ...]
    """
    triples = []
    text = response

    # Pattern: "X is/are Y" (simple IS_A facts)
    for m in re.finditer(
        r'([A-Z][A-Za-z_\s]{2,30})\s+(is|are|was|were)\s+([a-zA-Z][^\.,\n]{3,60})',
        text
    ):
        subj = m.group(1).strip().replace(" ", "_")
        pred = "IS"
        obj  = m.group(3).strip()[:60]
        if len(obj) > 3:
            triples.append((subj, pred, obj))

    # Pattern: "X uses/runs/supports Y"
    for m in re.finditer(
        r'([A-Z][A-Za-z_\s]{2,30})\s+(uses?|runs?|supports?|provides?|has|contains?)\s+([a-zA-Z][^\.,\n]{3,50})',
        text
    ):
        subj = m.group(1).strip().replace(" ", "_")
        pred = m.group(2).upper().rstrip("S").rstrip("E") + "S"
        obj  = m.group(3).strip()[:50]
        if len(obj) > 3:
            triples.append((subj, pred, obj))

    return triples[:10]   # cap at 10 per response


def _response_key(query: str, agent_id: str) -> str:
    return "rag:" + hashlib.md5(f"{agent_id}:{query[:60]}".encode()).hexdigest()[:10]


# ── Main extraction function ───────────────────────────────────────────

async def extract_and_ingest(
    query: str,
    response: str,
    agent_id: str,
    memory,
    delay: float = 0.5,
):
    """
    Fire-and-forget coroutine. Call with asyncio.create_task().

    Extracts entities and facts from the query+response pair and
    ingests them into the agent's memory namespace.
    """
    if not response or len(response) < 20:
        return
    await asyncio.sleep(delay)   # yield so the HTTP response goes out first

    try:
        combined = f"{query}\n\n{response}"
        entities = _extract_entities(combined)
        triples  = _extract_relations(query, response)
        key      = _response_key(query, agent_id)

        # Ingest the full response text for vector retrieval
        memory.ingest_fact(
            agent_id=agent_id,
            subject=key,
            predicate="RAG_RESPONSE",
            object_val=response[:200],
            speaker=agent_id,
            text=combined[:800],
        )

        # Ingest extracted entity mentions
        for entity in entities[:8]:
            ent_key = entity.replace(" ", "_")
            memory.ingest_fact(
                agent_id=agent_id,
                subject=ent_key,
                predicate="MENTIONED_IN",
                object_val=key,
                speaker="rag_extractor",
                text=f"{entity} mentioned in response to: {query[:80]}",
            )

        # Ingest extracted triples
        for subj, pred, obj in triples:
            memory.ingest_fact(
                agent_id=agent_id,
                subject=subj,
                predicate=pred,
                object_val=obj,
                speaker="rag_extractor",
                text=f"{subj} {pred} {obj}",
            )

    except Exception:
        pass   # never crash the main loop


def schedule_rag(query: str, response: str, agent_id: str, memory):
    """Helper — schedules RAG extraction as a background task."""
    asyncio.create_task(
        extract_and_ingest(query, response, agent_id, memory)
    )
