#!/usr/bin/env python3
"""
Multi-Agent Debate Engine
- Clarity scoring + interactive OR non-interactive clarification
- Pro and Con alternate for a fixed number of rounds, uninterrupted
- The Judge only looks at the transcript ONCE, after all rounds are done —
  no interim checkpoints, no early stopping, no going back for more rounds
- The Judge must back one side. No draws, no hedging.
- Dynamic temperature scaling
- Proper Pro/Con message roles (each side sees the other as "the opponent",
  not as its own prior assistant turn)
- Retries + partial-transcript save on LLM failure
- Full transcript saved to Markdown
- Model/base_url configurable for OpenAI-compatible endpoints (LiteLLM, etc.)
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Literal, TypedDict, Optional

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# ======================
# Configuration
# ======================
DEFAULT_ROUNDS = 4          # number of Pro/Con exchanges before the judge weighs in
HARD_ROUND_CAP = 12         # sanity ceiling, not a stopping rule — just a cost guard
CLARITY_THRESHOLD = 0.72
MAX_LLM_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2

# Model / endpoint config. Set via CLI flags or env vars so this can point
# at OpenAI directly or any OpenAI-compatible endpoint (e.g. a local LiteLLM
# proxy) without editing code.
DEFAULT_MODEL = os.environ.get("DEBATE_MODEL", "gpt-4o")
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # None -> official OpenAI API
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY")    # falls back to env automatically if None


# ======================
# Schemas
# ======================
class ClarityAssessment(BaseModel):
    clarity_score: float = Field(ge=0.0, le=1.0)
    is_clear: bool
    issues: List[str]
    clarifying_questions: List[str]
    suggested_refined_topic: Optional[str] = None


class JudgeDecision(BaseModel):
    # No "Draw" option on purpose: the judge is required to back a side.
    winner: Literal["Pro", "Con"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    key_points_pro: List[str]
    key_points_con: List[str]
    summary_table_markdown: str


class TranscriptEntry(TypedDict):
    speaker: Literal["Pro", "Con", "Judge"]
    round: int
    content: str
    rendered: str


class DebateState(TypedDict):
    topic: str
    transcript: List[TranscriptEntry]
    round: int
    total_rounds: int
    model: str
    base_url: Optional[str]
    api_key: Optional[str]
    final_decision: str
    confidence: float
    summary_table: str
    error: Optional[str]


# ======================
# LLM + Temperature
# ======================
def get_llm(temperature: float, model: str = DEFAULT_MODEL,
            base_url: Optional[str] = DEFAULT_BASE_URL,
            api_key: Optional[str] = DEFAULT_API_KEY):
    kwargs = {"model": model, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return ChatOpenAI(**kwargs)


def get_temperature(current_round: int, total_rounds: int) -> float:
    """Higher temperature early, lower later."""
    progress = (current_round - 1) / max(total_rounds - 1, 1)
    temp = 0.85 - (progress * 0.55)
    return round(max(0.30, min(0.85, temp)), 2)


def invoke_with_retry(llm, messages, structured_model: Optional[type] = None):
    """
    Wrap an LLM call with basic retry/backoff so a single transient API
    failure doesn't kill an entire in-progress debate.
    """
    target = llm.with_structured_output(structured_model) if structured_model else llm
    last_err = None
    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            return target.invoke(messages)
        except Exception as e:  # noqa: BLE001 - we want to catch/retry broadly here
            last_err = e
            if attempt < MAX_LLM_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"LLM call failed after {MAX_LLM_RETRIES} attempts: {last_err}") from last_err


# ======================
# Clarity Scoring
# ======================
def assess_topic_clarity(topic: str, model: str, base_url: Optional[str], api_key: Optional[str]) -> ClarityAssessment:
    llm = get_llm(0.2, model, base_url, api_key)
    prompt = f"""Evaluate this debate topic for clarity:
Topic: "{topic}"

Score 0.0-1.0. Below 0.72 means it needs clarifying questions.
Detect: too broad, vague, multiple questions, unclear sides, not debatable.
"""
    result = invoke_with_retry(
        llm,
        [SystemMessage(content="You are a rigorous debate topic evaluator."),
         HumanMessage(content=prompt)],
        structured_model=ClarityAssessment,
    )
    result.is_clear = result.clarity_score >= CLARITY_THRESHOLD
    return result


def refine_topic_with_answers(original: str, questions: List[str], answers: List[str],
                               model: str, base_url: Optional[str], api_key: Optional[str]) -> str:
    llm = get_llm(0.3, model, base_url, api_key)
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(questions, answers))
    prompt = f"""Original: "{original}"

{qa}

Rewrite into one clear, specific, debatable sentence. Return only the refined topic."""
    result = invoke_with_retry(
        llm,
        [SystemMessage(content="Refine debate topics for clarity."),
         HumanMessage(content=prompt)],
    )
    return result.content.strip().strip('"')


# ======================
# Message-history construction
# ======================
def build_context_messages(state: DebateState, self_speaker: str) -> List:
    """
    Build the message list an agent sees, with roles assigned relative to
    that agent: its own prior turns are AIMessage, everything else
    (the opponent's turns) is HumanMessage. This keeps Pro and Con
    genuinely rebutting each other instead of each side continuing its
    own monologue. The Judge never appears mid-transcript here, since it
    only speaks once, at the very end.
    """
    messages: List = [HumanMessage(content=f"Debate Topic: {state['topic']}")]
    for entry in state["transcript"]:
        if entry["speaker"] == self_speaker:
            messages.append(AIMessage(content=entry["rendered"]))
        else:
            messages.append(HumanMessage(content=entry["rendered"]))
    return messages


def make_entry(speaker: str, round_num: int, content: str) -> TranscriptEntry:
    rendered = f"**[{speaker} - Round {round_num}]**\n{content}"
    return {"speaker": speaker, "round": round_num, "content": content, "rendered": rendered}


# ======================
# Agents
# ======================
def pro_agent(state: DebateState):
    temp = get_temperature(state["round"], state["total_rounds"])
    llm = get_llm(temp, state["model"], state["base_url"], state["api_key"])
    system = f"""You are Agent Pro. Topic: {state['topic']}
Argue strongly in favor. Be logical and directly rebut the opponent's most
recent points (shown to you as the other side of the conversation).
180-260 words. Start with a clear thesis."""
    context = build_context_messages(state, "Pro")
    response = invoke_with_retry(llm, [SystemMessage(content=system), *context])
    entry = make_entry("Pro", state["round"], response.content)
    return {"transcript": state["transcript"] + [entry], "round": state["round"]}


def con_agent(state: DebateState):
    temp = get_temperature(state["round"], state["total_rounds"])
    llm = get_llm(temp, state["model"], state["base_url"], state["api_key"])
    system = f"""You are Agent Con. Topic: {state['topic']}
Argue strongly against. Challenge assumptions and directly attack Pro's
most recent points (shown to you as the other side of the conversation).
180-260 words."""
    context = build_context_messages(state, "Con")
    response = invoke_with_retry(llm, [SystemMessage(content=system), *context])
    entry = make_entry("Con", state["round"], response.content)
    return {
        "transcript": state["transcript"] + [entry],
        # Round advances once a full Pro/Con exchange is done. Nothing else
        # reads the transcript until every round has completed — see
        # route_after_con below.
        "round": state["round"] + 1,
    }


def judge_agent(state: DebateState):
    """
    Runs exactly once, after all rounds are complete. Reads the full
    transcript in a single pass and must commit to a winner — Pro or
    Con — with no draw option and no opportunity to ask for more rounds.
    """
    llm = get_llm(0.2, state["model"], state["base_url"], state["api_key"])
    transcript_text = "\n\n".join(e["rendered"] for e in state["transcript"])

    prompt = f"""Neutral debate judge. All {state['total_rounds']} rounds are complete —
this is your only look at the transcript, and your decision is final.

Topic: {state['topic']}

Full transcript:
{transcript_text}

You must back one side. "Pro" or "Con" only — no draws, no 50/50 hedging.
Weigh the strongest points each side actually made and commit to whichever
side made the stronger overall case. Give an honest confidence score: it's
fine for confidence to be modest (e.g. 55-65%) if the case was genuinely
close, as long as you still pick a side."""

    result = invoke_with_retry(
        llm,
        [SystemMessage(content="Precise, decisive debate judge."), HumanMessage(content=prompt)],
        structured_model=JudgeDecision,
    )

    decision = f"""**Winner:** {result.winner}
**Confidence:** {result.confidence:.0%}

**Reasoning:**
{result.reasoning}"""

    entry = make_entry("Judge", state["round"], decision)
    return {
        "final_decision": decision,
        "confidence": result.confidence,
        "summary_table": result.summary_table_markdown,
        "transcript": state["transcript"] + [entry],
    }


# ======================
# Graph
# ======================
def route_after_con(state: DebateState):
    # Round has just been incremented past the exchange that finished.
    # Keep alternating Pro/Con until every round is used up; only then
    # does the judge get a turn. There is no path back from the judge —
    # it evaluates once and the graph ends.
    if state["round"] <= state["total_rounds"]:
        return "pro"
    return "judge"


workflow = StateGraph(DebateState)
workflow.add_node("pro", pro_agent)
workflow.add_node("con", con_agent)
workflow.add_node("judge", judge_agent)
workflow.set_entry_point("pro")
workflow.add_edge("pro", "con")
workflow.add_conditional_edges("con", route_after_con, {"pro": "pro", "judge": "judge"})
workflow.add_edge("judge", END)
app = workflow.compile()


# ======================
# Save
# ======================
def save_transcript(topic, transcript: List[TranscriptEntry], final_decision, summary_table,
                     confidence, total_rounds, partial: bool = False) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)[:50]
    suffix = "_PARTIAL" if partial else ""
    path = Path("debate_transcripts") / f"debate_{ts}_{safe.strip().replace(' ', '_')}{suffix}.md"
    path.parent.mkdir(exist_ok=True)

    md = f"""# Debate Transcript{' (PARTIAL - run failed before completion)' if partial else ''}

**Topic:** {topic}
**Date:** {datetime.now():%Y-%m-%d %H:%M:%S}
**Rounds:** {total_rounds}
**Confidence:** {confidence:.0%}

---

## Full Debate

"""
    for entry in transcript:
        md += entry["rendered"] + "\n\n---\n\n"
    if final_decision:
        md += f"## Final Decision\n\n{final_decision}\n\n## Summary Table\n\n{summary_table}\n"
    path.write_text(md, encoding="utf-8")
    return str(path)


# ======================
# Two-phase clarification API (for embedding behind a UI/API, not just CLI)
# ======================
def get_clarifying_questions(topic: str, model: str = DEFAULT_MODEL,
                              base_url: Optional[str] = DEFAULT_BASE_URL,
                              api_key: Optional[str] = DEFAULT_API_KEY) -> ClarityAssessment:
    """
    Phase 1: score clarity and return questions if needed. Callers embedding
    this engine in a UI/API should call this first, and only prompt the
    human for answers if `is_clear` is False.
    """
    return assess_topic_clarity(topic, model, base_url, api_key)


def resolve_topic(topic: str, assessment: ClarityAssessment, answers: Optional[List[str]],
                   model: str, base_url: Optional[str], api_key: Optional[str]) -> str:
    """
    Phase 2: given answers to the clarifying questions (or None if the topic
    was already clear / running non-interactively), produce the final topic
    to debate.
    """
    if assessment.is_clear or not assessment.clarifying_questions:
        return topic
    if not answers:
        # Non-interactive fallback: proceed with the original topic rather
        # than blocking on input() -- important for API/UI callers that
        # can't service a blocking stdin prompt.
        return topic
    return refine_topic_with_answers(topic, assessment.clarifying_questions, answers, model, base_url, api_key)


def run_debate_graph(topic: str, total_rounds: int, model: str, base_url: Optional[str],
                      api_key: Optional[str], on_update=None) -> DebateState:
    """
    Runs the compiled graph to completion. `on_update`, if given, is called
    with each intermediate state (useful for streaming progress to a UI).
    Saves a partial transcript automatically if the run raises.
    """
    state: DebateState = {
        "topic": topic,
        "transcript": [],
        "round": 1,
        "total_rounds": total_rounds,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "final_decision": "",
        "confidence": 0.0,
        "summary_table": "",
        "error": None,
    }

    final = state
    try:
        for event in app.stream(state, stream_mode="values"):
            final = event
            if on_update:
                on_update(final)
        return final
    except Exception as e:  # noqa: BLE001
        final["error"] = str(e)
        path = save_transcript(
            topic, final.get("transcript", []), final.get("final_decision", ""),
            final.get("summary_table", ""), final.get("confidence", 0.0),
            final.get("total_rounds", total_rounds), partial=True,
        )
        raise RuntimeError(f"Debate failed after round {final.get('round')}. "
                            f"Partial transcript saved to {path}. Original error: {e}") from e


# ======================
# CLI entry point (interactive)
# ======================
def run_debate(topic: str, rounds: int = DEFAULT_ROUNDS,
               model: str = DEFAULT_MODEL, base_url: Optional[str] = DEFAULT_BASE_URL,
               api_key: Optional[str] = DEFAULT_API_KEY, non_interactive: bool = False) -> dict:
    if rounds > HARD_ROUND_CAP:
        print(f"--rounds {rounds} exceeds the sanity cap of {HARD_ROUND_CAP}; clamping.")
        rounds = HARD_ROUND_CAP

    print(f"\nAssessing clarity of: {topic}")
    assessment = get_clarifying_questions(topic, model, base_url, api_key)
    print(f"Clarity Score: {assessment.clarity_score:.2f}")

    answers = None
    if not assessment.is_clear and assessment.clarifying_questions:
        if non_interactive:
            print("Topic is below clarity threshold but running non-interactively; "
                  "proceeding with original topic.")
        else:
            print("\nTopic needs clarification:")
            answers = []
            for i, q in enumerate(assessment.clarifying_questions, 1):
                print(f"\nQ{i}: {q}")
                answers.append(input("Your answer: ").strip() or "No preference")

    topic = resolve_topic(topic, assessment, answers, model, base_url, api_key)
    if answers:
        print(f"\nRefined topic: {topic}")

    print(f"\nDebate started — {rounds} rounds, then a single judge verdict.")

    def on_update(state: DebateState):
        if state.get("transcript"):
            print(state["transcript"][-1]["rendered"][:400])
            print("-" * 40)

    final = run_debate_graph(topic, rounds, model, base_url, api_key, on_update=on_update)

    md_path = save_transcript(
        topic, final["transcript"], final["final_decision"],
        final["summary_table"], final["confidence"], final["total_rounds"],
    )

    print("\nFINAL RESULT")
    print(final["final_decision"])
    print(final["summary_table"])
    print(f"\nSaved: {md_path}")

    return {
        "final_topic": topic,
        "winner_decision": final["final_decision"],
        "confidence": final["confidence"],
        "summary_table": final["summary_table"],
        "total_rounds": final["total_rounds"],
        "transcript_file": md_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                         help="Number of Pro/Con exchanges before the judge evaluates once (default: 4)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name, e.g. gpt-4o")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                         help="OpenAI-compatible base URL, e.g. a LiteLLM proxy")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY,
                         help="API key (falls back to OPENAI_API_KEY env var if omitted)")
    parser.add_argument("--non-interactive", action="store_true",
                         help="Skip clarifying-question prompts; run with original topic if unclear")
    args = parser.parse_args()

    result = run_debate(
        args.topic, args.rounds,
        args.model, args.base_url, args.api_key, args.non_interactive,
    )
    print("\n" + json.dumps(result, indent=2))
