# sowr

Research on whether a decision model answers a question more accurately when it first answers a tree of generated sub-decisions about the same input.

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
The root decision together with all its sub-decisions, generated for a single input.
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
