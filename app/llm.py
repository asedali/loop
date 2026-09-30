"""
LLM abstraction layer — calls a model via OpenRouter (OpenAI-compatible API),
so you can point this at any model OpenRouter hosts (Gemini, Claude, GPT,
Llama, DeepSeek, etc.) just by changing OPENROUTER_MODEL, no code changes.

Every call asks for JSON-only output and validates it can be parsed before
returning. Models sometimes ignore "return JSON only" instructions (wrap it
in prose, add markdown fences, etc.) — the retry loop feeds the parse error
back into a follow-up prompt, up to MAX_ATTEMPTS times, before raising. This
keeps every AI touchpoint a structured card/form, never a raw chat reply,
per the product's core "fixed shape, not open chat" rule.
"""
import json
import os
import re

from openai import OpenAI

MAX_ATTEMPTS = 3
# Any OpenRouter model slug works here — see https://openrouter.ai/models
# Override with the OPENROUTER_MODEL env var without touching this file.
DEFAULT_MODEL = "google/gemini-2.0-flash-exp:free"


class LLMError(Exception):
    pass


def _get_client_and_model():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMError(
            "OPENROUTER_API_KEY is not set. Add it as an environment variable "
            "(see .env.example) before using any AI feature."
        )
    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return client, model


def _extract_json(text: str):
    """Strip markdown fences / stray prose and parse the first JSON object/array."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # fall back: find the outermost {...} or [...] in the text
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise json.JSONDecodeError("No JSON found", cleaned, 0)


def call_json(prompt: str) -> dict:
    """Call the configured OpenRouter model, requiring a JSON-parseable response.
    Retries on bad JSON."""
    client, model = _get_client_and_model()
    last_error = None
    current_prompt = prompt
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": current_prompt}],
            )
            text = response.choices[0].message.content
            return _extract_json(text)
        except json.JSONDecodeError as e:
            last_error = e
            current_prompt = (
                f"{prompt}\n\nYour previous response could not be parsed as JSON "
                f"(error: {e}). Respond with ONLY valid JSON, no markdown fences, "
                f"no commentary before or after."
            )
        except Exception as e:
            last_error = e
    raise LLMError(f"OpenRouter call to '{model}' failed after {MAX_ATTEMPTS} attempts: {last_error}")


# ---------------- Phase 1: Idea Discovery ----------------

def extract_ideas(raw_text: str) -> list:
    prompt = f"""You are helping a researcher find commercially viable ideas from their own work.

Below is raw material the researcher provided (CV excerpt, abstract, thesis snippet, or free description).
Extract distinct technical claims, then cluster related ones into 3 to 6 candidate commercial ideas.

For each idea return:
- "title": short punchy name (a few words)
- "commercial_framing": one plain-language sentence on the commercial angle
- "strength_signal": one of "strong", "moderate", "early" — your rough read of commercial promise
- "raw_claims": the specific technical claim(s) this idea is built from, one short sentence

Return ONLY a JSON array of objects with exactly these four keys. No markdown, no commentary.

RESEARCHER MATERIAL:
\"\"\"
{raw_text}
\"\"\"
"""
    result = call_json(prompt)
    if not isinstance(result, list):
        raise LLMError("Expected a JSON array of idea cards")
    return result


# ---------------- Phase 2: Validation Loop ----------------

def generate_todos(idea_title: str, idea_framing: str, bmc_elements: list, cycle_number: int) -> list:
    element_lines = "\n".join(
        f"- {el['element_name']}: {el['status']}" + (f" ({el['notes']})" if el["notes"] else "")
        for el in bmc_elements
    )
    prompt = f"""You are running cycle {cycle_number} of a lean-startup validation loop for this idea:

Idea: {idea_title}
Commercial framing: {idea_framing}

Current Business Model Canvas element status:
{element_lines}

Generate 2 to 4 concrete, real-world validation TODOs targeting the LEAST-validated elements above
(prioritize "untested" and "disconfirmed" over "mixed", and never target elements already "confirmed").
Each TODO must be something the researcher can actually go do this week — an interview, a landing
page test, a smoke test, an expert review, a survey, etc. Be specific to their domain, not generic.

For each TODO return:
- "task_type": one of "interview", "landing_page_test", "survey", "smoke_test", "expert_review"
- "target_element": which BMC element this tests (customer/problem/value_prop/channel/revenue)
- "description": concrete instructions for what to actually do

Return ONLY a JSON array of objects with exactly these three keys. No markdown, no commentary.
"""
    result = call_json(prompt)
    if not isinstance(result, list):
        raise LLMError("Expected a JSON array of TODOs")
    return result


def analyze_cycle(idea_title: str, todos: list, results: list, bmc_elements: list) -> dict:
    todo_result_pairs = "\n".join(
        f"- [{t.get('target_element')}] {t.get('description')}\n  Outcome logged: {r.get('outcome')} "
        f"(sample size: {r.get('sample_size', 'n/a')})"
        for t, r in zip(todos, results)
    )
    element_lines = "\n".join(f"- {el['element_name']}: currently {el['status']}" for el in bmc_elements)
    prompt = f"""You are analyzing validation evidence for this idea: {idea_title}

Current BMC element status:
{element_lines}

This cycle's TODOs and the outcomes the researcher logged:
{todo_result_pairs}

For each of the five BMC elements (customer, problem, value_prop, channel, revenue), decide an updated
status: "untested" (no new evidence), "mixed" (some evidence, contradictory or thin), "confirmed"
(solid real-world evidence supports it), or "disconfirmed" (evidence contradicts it).
Be honest about small sample sizes — do not mark something "confirmed" from one or two data points;
use "mixed" instead and note why.

Then give an overall cycle recommendation: "persevere" (evidence is trending toward validation, but
not all elements are confirmed yet — continue looping), "pivot" (evidence disconfirms a core
assumption; suggest a SPECIFIC pivot direction), or "kill" (evidence is decisively negative and no
pivot is viable).

Return ONLY a JSON object with exactly these keys:
{{
  "elements": [
    {{"element_name": "...", "status": "...", "notes": "one-sentence justification"}}
  ],
  "recommendation": "persevere" | "pivot" | "kill",
  "recommendation_reasoning": "2-3 sentences",
  "pivot_suggestion": "specific pivot description, or null if not applicable"
}}
No markdown, no commentary outside the JSON.
"""
    result = call_json(prompt)
    if "elements" not in result or "recommendation" not in result:
        raise LLMError("Analysis response missing required keys")
    return result


# ---------------- Phase 3: Launch Strategy ----------------

def generate_launch_strategy(idea_title: str, idea_framing: str, bmc_elements: list) -> dict:
    element_lines = "\n".join(
        f"- {el['element_name']}: {el['notes']}" for el in bmc_elements
    )
    prompt = f"""A researcher has validated this idea through a real-world evidence loop:

Idea: {idea_title}
Commercial framing: {idea_framing}

Validated Business Model Canvas evidence:
{element_lines}

Produce a go-to-market package with three parts:

1. "funding_matches": 3-5 funding TYPES/categories that fit this idea's domain and stage
   (e.g. "SBIR/STTR Phase I", "university TTO proof-of-concept fund", "pre-seed deep-tech VC",
   "climate/deep-tech accelerator"). For each: {{"name": "...", "why_it_fits": "one sentence"}}.
   Note this is category-level guidance, not a live grants database.

2. "gtm_channels": 2-3 candidate customer-acquisition channels derived from the validated
   Customer/Channel evidence above, NOT generic startup advice. Each:
   {{"channel": "...", "why_this_fits": "one sentence tied to the actual evidence"}}.

3. "action_plan": 5-8 sequenced, concrete checklist items to reach a first real milestone
   (first grant submitted, first pilot signed, or first paying customer). Each:
   {{"step": "...", "milestone_type": "grant" | "pilot" | "customer"}}.

Return ONLY a JSON object with exactly the keys "funding_matches", "gtm_channels", "action_plan".
No markdown, no commentary outside the JSON.
"""
    result = call_json(prompt)
    for key in ("funding_matches", "gtm_channels", "action_plan"):
        if key not in result:
            raise LLMError(f"Launch strategy response missing '{key}'")
    return result
