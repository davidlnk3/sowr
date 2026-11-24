import json

import pytest

from sowr import llm
from sowr.decision import normalize
from sowr.experiment import run
from sowr.tree import answer_tree, truncate

ROOT = {"type": "choice", "instructions": "root?", "criteria": {"a": "A", "b": "B"}}
GEN = llm.Generator(model="fake")


def fake_chat(width):
    """A Generator that always returns `width` valid yes/no sub-decisions."""
    counter = iter(range(10_000))

    def chat(generator, messages, schema):
        if schema is llm.ANSWER_SCHEMA:
            return json.dumps({"answer": "a"})
        subs = [{"type": "noul", "instructions": f"q{next(counter)}?",
                 "options": [{"label": "true", "description": "yes"}, {"label": "false", "description": "no"}]}
                for _ in range(width)]
        return json.dumps({"sub_decisions": subs})

    return chat


class FakeDecider:
    def __init__(self):
        self.calls = []

    def __call__(self, questions, state):
        self.calls.append((list(questions), state))
        return {qid: {"answer": f"ans-{qid}", "confidence": 0.9} for qid in questions}


def count_nodes(node):
    return 1 + sum(count_nodes(c) for c in node["children"])


@pytest.mark.parametrize("width,depth", [(1, 1), (2, 2), (3, 2)])
def test_build_tree_shape(width, depth):
    tree = llm.build_tree(GEN, {"x": 1}, ROOT, width, depth, fake_chat(width))
    assert count_nodes(tree) == sum(width**k for k in range(depth + 1))
    assert tree["question"] == ROOT


def test_generator_gets_chain_from_root_to_parent():
    seen = []
    inner = fake_chat(2)

    def chat(generator, messages, schema):
        seen.append(json.loads(messages[1]["content"])["decision_chain"])
        return inner(generator, messages, schema)

    llm.build_tree(GEN, {"x": 1}, ROOT, 2, 2, chat)
    assert [len(c) for c in seen] == [1, 2, 2]
    assert all(c[0] == ROOT for c in seen)


def test_truncate_keeps_top_levels():
    tree = llm.build_tree(GEN, {}, ROOT, 3, 2, fake_chat(3))
    shallow = truncate(tree, 1)
    assert count_nodes(shallow) == 4
    assert [c["question"] for c in shallow["children"]] == [c["question"] for c in tree["children"]]
    assert count_nodes(truncate(tree, 0)) == 1


def test_depth_zero_is_one_root_call():
    decide = FakeDecider()
    answers, calls = answer_tree({"id": "r", "question": ROOT, "children": []}, {"x": 1}, decide)
    assert calls == 1 and decide.calls == [(["r"], {"x": 1})]
    assert answers["r"]["answer"] == "ans-r"


def test_call_pattern_and_children_carry():
    tree = llm.build_tree(GEN, {}, ROOT, 3, 2, fake_chat(3))
    decide = FakeDecider()
    answers, calls = answer_tree(tree, {"x": 1}, decide)
    # 3 leaf sibling groups + 3 middle nodes + root
    assert calls == 7
    leaf_groups = [ids for ids, _ in decide.calls if len(ids) == 3]
    assert leaf_groups == [["r_0_0", "r_0_1", "r_0_2"], ["r_1_0", "r_1_1", "r_1_2"], ["r_2_0", "r_2_1", "r_2_2"]]
    assert all("carried_decisions" not in s for ids, s in decide.calls if len(ids) == 3)
    ids, root_state = decide.calls[-1]
    assert ids == ["r"]
    assert [c["answer"] for c in root_state["carried_decisions"]] == ["ans-r_0", "ans-r_1", "ans-r_2"]
    assert root_state["x"] == 1
    assert len(answers) == 13


def test_subtree_carry_includes_grandchildren():
    tree = llm.build_tree(GEN, {}, ROOT, 2, 2, fake_chat(2))
    decide = FakeDecider()
    answer_tree(tree, {}, decide, carry="subtree")
    assert len(decide.calls[-1][1]["carried_decisions"]) == 6


def test_invalid_generator_output_is_retried_then_fails():
    replies = iter([
        "not json",
        json.dumps({"sub_decisions": []}),  # wrong count
        json.dumps({"sub_decisions": [{"type": "noul", "instructions": "q?",
                                       "options": [{"label": "yes", "description": ""}, {"label": "no", "description": ""}]}]}),
        json.dumps({"sub_decisions": [{"type": "score", "instructions": "q?",
                                       "options": [{"label": "low", "description": "l"}, {"label": "high", "description": "h"}]}]}),
    ])
    children = llm.generate_children(GEN, {}, [ROOT], 1, lambda *a: next(replies))
    assert children == [{"type": "score", "instructions": "q?", "criteria": ["low: l", "high: h"]}]

    with pytest.raises(llm.GenerationFailed):
        llm.generate_children(GEN, {}, [ROOT], 1, lambda *a: "not json")


def test_to_question_shapes():
    opts = [{"label": "x", "description": "X"}, {"label": "y", "description": "Y"}]
    assert llm.to_question({"type": "choice", "instructions": "q", "options": opts})["criteria"] == {"x": "X", "y": "Y"}
    with pytest.raises(ValueError):
        llm.to_question({"type": "choice", "instructions": "q", "options": opts[:1]})


def test_normalize_noul():
    assert normalize({"type": "noul", "noul": 0.2}) == {"answer": "false", "confidence": 0.8, "raw": {"type": "noul", "noul": 0.2}}
    assert normalize({"type": "choice", "choice": "a", "confidence": 0.7, "probabilities": {}})["answer"] == "a"


def test_run_writes_all_conditions_and_reuses_trees(tmp_path):
    out = tmp_path / "runs"
    args = ("datasets/example", "gpt-6-luna", [1, 2], 2, 10, out)
    run(*args, decide=FakeDecider(), chat=fake_chat_for_any_width())
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    # per item: baseline + generator_only + per width (depth 1, depth 2, depth 2 subtree)
    assert len(rows) == 2 * (2 + 2 * 3)
    assert not any(r["error"] for r in rows)
    trees = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.json"))
    assert trees == [f"trees/example/gpt-6-luna/{i}/w{w}.json" for i in ("1", "2") for w in (1, 2)]

    # second run must not call the Generator for trees again
    def no_tree_chat(generator, messages, schema):
        assert schema is llm.ANSWER_SCHEMA
        return json.dumps({"answer": "Estético"})

    run(*args, decide=FakeDecider(), chat=no_tree_chat)


def fake_chat_for_any_width():
    def chat(generator, messages, schema):
        if schema is llm.ANSWER_SCHEMA:
            return json.dumps({"answer": "Estético"})
        width = int(messages[0]["content"].split("Write exactly ")[1].split()[0])
        return fake_chat(width)(generator, messages, schema)

    return chat
