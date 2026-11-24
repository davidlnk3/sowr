"""Run the baselines and the width x depth grid over a dataset.

A dataset is a directory with:
- root.json: the hand-written root decision, as a Decider question
- items.jsonl: one {"id", "state", "label"} per line, where state is a dict
"""

import json
from pathlib import Path

from . import llm
from .decision import DECIDER, answer_questions
from .tree import answer_tree, truncate


def load_dataset(path):
    path = Path(path)
    root = json.loads((path / "root.json").read_text())
    items = [json.loads(line) for line in (path / "items.jsonl").read_text().splitlines() if line.strip()]
    return path.name, root, items


def load_or_build_tree(tree_file, generator, state, root, width, depth, chat):
    """Each tree is generated once and then reused by every condition and rerun."""
    if tree_file.exists():
        return json.loads(tree_file.read_text())
    tree = llm.build_tree(generator, state, root, width, depth, chat)
    tree_file.parent.mkdir(parents=True, exist_ok=True)
    tree_file.write_text(json.dumps(tree, ensure_ascii=False, indent=2))
    return tree


def run(dataset_dir, generator_name, widths, max_depth, limit, out_dir,
        baseline_repeats=1, decider=DECIDER, decide=None, chat=llm.chat):
    decide = decide or (lambda questions, state: answer_questions(questions, state, decider))
    name, root, items = load_dataset(dataset_dir)
    generator = llm.GENERATORS[generator_name]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.jsonl", "a") as results:
        for item in items[:limit]:
            state = item["state"]

            def record(condition, width=None, depth=None, carry=None, repeat=None, answer=None, calls=None, error=None):
                row = {
                    "dataset": name, "item": item["id"], "label": item["label"], "condition": condition,
                    "decider": decider, "generator": generator_name, "width": width, "depth": depth,
                    "carry": carry, "repeat": repeat, "answer": answer and answer.get("answer"),
                    "confidence": answer and answer.get("confidence"), "calls": calls, "error": error,
                }
                results.write(json.dumps(row, ensure_ascii=False) + "\n")
                results.flush()

            for r in range(baseline_repeats):
                try:
                    answers, calls = answer_tree({"id": "r", "question": root, "children": []}, state, decide)
                    record("baseline", depth=0, repeat=r, answer=answers["r"], calls=calls)
                except Exception as e:
                    record("baseline", depth=0, repeat=r, error=repr(e))

            try:
                record("generator_only", answer=llm.answer_directly(generator, state, root, chat))
            except Exception as e:
                record("generator_only", error=repr(e))

            for width in widths:
                tree_file = out_dir / "trees" / name / generator_name / str(item["id"]) / f"w{width}.json"
                conditions = [(d, "children") for d in range(1, max_depth + 1)] + [(max_depth, "subtree")] * (max_depth >= 2)
                try:
                    tree = load_or_build_tree(tree_file, generator, state, root, width, max_depth, chat)
                except Exception as e:
                    for depth, carry in conditions:
                        record("tree", width, depth, carry, error=repr(e))
                    continue
                for depth, carry in conditions:
                    try:
                        answers, calls = answer_tree(truncate(tree, depth), state, decide, carry)
                        record("tree", width, depth, carry, answer=answers["r"], calls=calls)
                    except Exception as e:
                        record("tree", width, depth, carry, error=repr(e))
