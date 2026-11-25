"""Decision trees: truncating them and having the Decider answer them bottom-up.

A tree node is a dict: {"id": str, "question": <Decider question>, "children": [nodes]}.
"""


def truncate(node, depth):
    """Return a copy of the tree with only `depth` levels of sub-decisions."""
    children = [truncate(c, depth - 1) for c in node["children"]] if depth > 0 else []
    return {**node, "children": children}


def descendants(node):
    for child in node["children"]:
        yield child
        yield from descendants(child)


def considerations(tree):
    """The instructions of every sub-decision, for the Context-only baseline."""
    return [n["question"]["instructions"] for n in descendants(tree)]


def answer_tree(tree, state, decide, carry="children"):
    """Answer every decision in the tree, deepest level first.

    Each sibling group at the deepest level is answered in one call. Every decision
    above it gets its own call, with its carried decisions added to `state`:
    its direct children's (carry="children") or all its descendants' (carry="subtree").

    `decide(questions, state)` returns {id: {"answer", "confidence", ...}}.
    Returns (answers by node id, number of Decider calls).
    """
    answers = {}
    calls = 0

    def ask(nodes, node_state):
        nonlocal calls
        calls += 1
        answers.update(decide({n["id"]: n["question"] for n in nodes}, node_state))

    def carried(n):
        a = answers[n["id"]]
        return {"decision": n["question"].get("instructions"), "answer": a["answer"], "confidence": a["confidence"]}

    def solve(node):
        children = node["children"]
        if not children:  # only reached for a depth-0 tree: the root alone
            ask([node], state)
            return
        if children[0]["children"]:
            for child in children:
                solve(child)
        else:
            ask(children, state)
        sources = children if carry == "children" else list(descendants(node))
        ask([node], {**state, "carried_decisions": [carried(n) for n in sources]})

    solve(tree)
    return answers, calls
