# sowr

System One With Reasoning. Research on whether a fast decision model answers a decision more accurately when it first answers a reusable tree of generated sub-decisions.

## Language

### Models

**Decider**:
The decision model under test. It answers every decision in the tree and the root decision.
_Avoid_: decision model, judge, classifier

**Generator**:
A lightweight LLM that writes the sub-decisions. It never answers them.
_Avoid_: LLM, question writer, planner

### Tree

**Root decision**:
The question whose answer is actually being measured.
_Avoid_: main decision, main question, target

**Sub-decision**:
Any question in the tree other than the root decision, which the Decider answers before its parent.
_Avoid_: sub-question, child question, node question

**Decision tree**:
The root decision together with all its sub-decisions. It is written once from the root decision and a few unlabeled example inputs, never the inputs it will be used on, and reused for every input of that root decision.
_Avoid_: sub-decision tree, question tree

**Answer type**:
The form of a decision's answer: a choice between named options, a yes/no, or a score on an ordered scale. The root decision's answer type comes from the dataset. The Generator picks the answer type of each sub-decision.
_Avoid_: output type, question type, format

**Parent / Children / Siblings**:
The relationships between decisions in a decision tree. Siblings share a parent.
_Avoid_: father, son

**Width**:
The number of children every decision has, apart from the deepest level.
_Avoid_: breadth, branching

**Depth**:
The number of levels of sub-decisions below the root decision. Depth 0 means the root decision alone.
_Avoid_: height, levels

### Evaluation

**Carried decision**:
A child's answer together with the Decider's confidence in it, which the Decider receives when it answers that child's parent. A decision receives only its direct children's carried decisions.
_Avoid_: extra information, context, propagated answer

**Baseline**:
The Decider answering the root decision alone, without a decision tree, on the same input.
_Avoid_: control, vanilla

**Generator-only baseline**:
The Generator answering the root decision directly, on the same input.

**Context-only baseline**:
The Decider answering the root decision with the tree's sub-decisions given as context but not answered. It separates the value of the decomposition from the value of answering it.
_Avoid_: unanswered tree

### Reuse

**Tree library**:
The stored decision trees, found by how similar their root decision is to a new one.
_Avoid_: tree database, cache, RAG store

**Hit**:
A new root decision whose closest stored root decision is at least as similar as the similarity threshold, so that tree's sub-decisions are reused under the new root decision. Otherwise it is a miss, and a new tree is written and added to the library.
_Avoid_: match, cache hit

**Similarity threshold**:
The minimum similarity between root decisions for a hit.
_Avoid_: cutoff
