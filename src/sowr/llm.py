import json
import os
from dataclasses import dataclass, field

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 3


@dataclass(frozen=True)
class Generator:
    """Any OpenAI-compatible model. `params` go into the request body as-is."""

    model: str
    base_url: str = OPENROUTER_BASE_URL
    api_key_env: str = "OPENROUTER_API_KEY"
    params: dict = field(default_factory=dict)


GENERATORS = {
    "gpt-6-luna": Generator(
        model="openai/gpt-6-luna",
        params={
            "seed": 0,
            "reasoning": {"effort": "minimal"},
            "provider": {"order": ["openai"], "allow_fallbacks": False},
        },
    ),
}


class GenerationFailed(Exception):
    pass


_clients = {}


def chat(generator, messages, schema):
    """Send one request with a JSON schema response format; return the reply text."""
    key = (generator.base_url, generator.api_key_env)
    if key not in _clients:
        _clients[key] = OpenAI(base_url=generator.base_url, api_key=os.getenv(generator.api_key_env), timeout=120, max_retries=5)
    response = _clients[key].chat.completions.create(
        model=generator.model,
        messages=messages,
        response_format={"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": schema}},
        extra_body=generator.params,
    )
    return response.choices[0].message.content


def _with_retries(generator, messages, schema, parse, chat):
    errors = []
    for _ in range(1 + MAX_RETRIES):
        try:
            return parse(json.loads(chat(generator, messages, schema)))
        except (ValueError, KeyError, TypeError) as e:  # json.JSONDecodeError is a ValueError
            errors.append(f"{type(e).__name__}: {e}")
    raise GenerationFailed("; ".join(errors))


# --- Generating sub-decisions -------------------------------------------------

_OPTION = {
    "type": "object",
    "properties": {"label": {"type": "string"}, "description": {"type": "string"}},
    "required": ["label", "description"],
    "additionalProperties": False,
}
SUB_DECISIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["choice", "noul", "score"]},
                    "instructions": {"type": "string"},
                    "options": {"type": "array", "items": _OPTION},
                },
                "required": ["type", "instructions", "options"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["sub_decisions"],
    "additionalProperties": False,
}

GENERATE_PROMPT = """You break a decision into simpler sub-decisions.

You get an input and a chain of decisions, from the root decision down to a parent decision. Write exactly {width} sub-decisions about the same input whose answers would help answer the parent decision.

Rules:
- Every sub-decision must be answerable from the input alone.
- Sub-decisions must cover different aspects. Do not reword each other or the parent.
- Never state, assume or hint at the answer to the parent or to any decision in the chain.
- Pick the type of each sub-decision:
  - "noul": a yes/no question. Give exactly two options, labelled "true" and "false", each describing when it applies.
  - "choice": one of two or more named options. Give each a short label and a description.
  - "score": a position on an ordered scale. Give two or more options, ordered from lowest to highest.
- Write in the language of the input."""


def to_question(sub):
    """Turn a Generator sub-decision into a Decider question, or raise ValueError."""
    kind, options = sub["type"], sub["options"]
    labels = [o["label"] for o in options]
    if not sub["instructions"].strip():
        raise ValueError("empty instructions")
    if len(options) < 2 or len(set(labels)) != len(labels):
        raise ValueError(f"{kind} needs two or more distinct options, got {labels}")
    if kind == "noul":
        if sorted(labels) != ["false", "true"]:
            raise ValueError(f"noul options must be true/false, got {labels}")
        criteria = {o["label"]: o["description"] for o in options}
    elif kind == "choice":
        criteria = {o["label"]: o["description"] for o in options}
    else:
        criteria = [f"{o['label']}: {o['description']}" for o in options]
    return {"type": kind, "instructions": sub["instructions"], "criteria": criteria}


def generate_children(generator, state, chain, width, chat=chat):
    """Write `width` sub-decisions for the last decision in `chain` (root first)."""
    messages = [
        {"role": "system", "content": GENERATE_PROMPT.format(width=width)},
        {"role": "user", "content": json.dumps({"input": state, "decision_chain": chain}, ensure_ascii=False)},
    ]

    def parse(reply):
        subs = reply["sub_decisions"]
        if len(subs) != width:
            raise ValueError(f"expected {width} sub-decisions, got {len(subs)}")
        return [to_question(s) for s in subs]

    return _with_retries(generator, messages, SUB_DECISIONS_SCHEMA, parse, chat)


def build_tree(generator, state, root_question, width, depth, chat=chat):
    """Build the decision tree breadth-first, one Generator call per sibling group."""
    root = {"id": "r", "question": root_question, "children": []}
    level = [(root, [root_question])]
    for _ in range(depth):
        next_level = []
        for node, chain in level:
            for i, question in enumerate(generate_children(generator, state, chain, width, chat)):
                child = {"id": f"{node['id']}_{i}", "question": question, "children": []}
                node["children"].append(child)
                next_level.append((child, chain + [question]))
        level = next_level
    return root


# --- Generator-only baseline ----------------------------------------------------

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

ANSWER_PROMPT = """Answer the decision about the input. Reply with exactly one of these values for "answer": {allowed}"""


def allowed_answers(question):
    criteria = question.get("criteria")
    if question["type"] == "noul":
        return ["true", "false"]
    if question["type"] == "choice":
        return list(criteria)
    return [str(c) for c in criteria]


def answer_directly(generator, state, question, chat=chat):
    """The Generator answers a decision itself. Returns {"answer": ...}."""
    allowed = allowed_answers(question)
    messages = [
        {"role": "system", "content": ANSWER_PROMPT.format(allowed=json.dumps(allowed, ensure_ascii=False))},
        {"role": "user", "content": json.dumps({"input": state, "decision": question}, ensure_ascii=False)},
    ]

    def parse(reply):
        if reply["answer"] not in allowed:
            raise ValueError(f"answer {reply['answer']!r} not in {allowed}")
        return {"answer": reply["answer"]}

    return _with_retries(generator, messages, ANSWER_SCHEMA, parse, chat)
