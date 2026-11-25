"""Run the baselines and the width x depth grid over a dataset.

A dataset is a directory with:
- root.json: the hand-written root decision, as a Decider question
- items.jsonl: the evaluation items, one {"id", "state", "label"} per line, where state is a dict
- validation.jsonl: held-out items in the same format; their inputs are the Generator's examples
"""

import json
from pathlib import Path

from . import llm
from .decision import DECIDER, answer_questions
from .tree import answer_tree, considerations, truncate

N_EXAMPLES = 3


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_dataset(path):
    path = Path(path)
    root = json.loads((path / "root.json").read_text())
    return path.name, root, read_jsonl(path / "items.jsonl"), read_jsonl(path / "validation.jsonl")


def load_or_build_tree(tree_file, generator, examples, root, width, depth, seed, chat):
    """Each tree is written once and then reused for every input, condition and rerun."""
    if tree_file.exists():
        return json.loads(tree_file.read_text())
    tree = llm.build_tree(generator, examples, root, width, depth, seed, chat)
    tree_file.parent.mkdir(parents=True, exist_ok=True)
    tree_file.write_text(json.dumps(tree, ensure_ascii=False, indent=2))
    return tree


def run(dataset_dir, generator_name, widths, max_depth, limit, out_dir,
        baseline_repeats=1, seeds=(0, 1, 2), decider=DECIDER, decide=None, chat=llm.chat):
    decide = decide or (lambda questions, state: answer_questions(questions, state, decider))
    name, root, items, validation = load_dataset(dataset_dir)
    examples = [v["state"] for v in validation[:N_EXAMPLES]]
    generator = llm.GENERATORS[generator_name]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trees = {}  # (width, seed) -> tree, or the exception that stopped it
    for width in widths:
        for seed in seeds:
            tree_file = out_dir / "trees" / name / generator_name / f"w{width}-s{seed}.json"
            try:
                trees[width, seed] = load_or_build_tree(tree_file, generator, examples, root, width, max_depth, seed, chat)
            except Exception as e:
                trees[width, seed] = e

    conditions = [c for d in range(1, max_depth + 1) for c in (("tree", d, "children"), ("context_only", d, None))]
    conditions += [("tree", max_depth, "subtree")] * (max_depth >= 2)

    with open(out_dir / "results.jsonl", "a") as results:
        for item in items[:limit]:
            state = item["state"]

            def record(condition, width=None, seed=None, depth=None, carry=None, repeat=None,
                       answer=None, calls=None, error=None):
                row = {
                    "dataset": name, "item": item["id"], "label": item["label"], "condition": condition,
                    "decider": decider, "generator": generator_name, "width": width, "seed": seed,
                    "depth": depth, "carry": carry, "repeat": repeat, "answer": answer and answer.get("answer"),
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

            for (width, seed), tree in trees.items():
                for condition, depth, carry in conditions:
                    if isinstance(tree, Exception):
                        record(condition, width, seed, depth, carry, error=repr(tree))
                        continue
                    shallow = truncate(tree, depth)
                    try:
                        if condition == "context_only":
                            answers = decide({"r": root}, {**state, "considerations": considerations(shallow)})
                            record(condition, width, seed, depth, answer=answers["r"], calls=1)
                        else:
                            answers, calls = answer_tree(shallow, state, decide, carry)
                            record(condition, width, seed, depth, carry, answer=answers["r"], calls=calls)
                    except Exception as e:
                        record(condition, width, seed, depth, carry, error=repr(e))
