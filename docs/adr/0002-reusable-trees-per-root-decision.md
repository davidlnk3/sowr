# Reusable decision trees, written once per root decision

A tree built for every input puts a Generator call in the path of every answer, which costs about as much as letting the Generator answer directly. So the Generator now writes each decision tree from the root decision plus three unlabeled example inputs from the held-out validation items, never from the evaluation items. That tree is reused for every input of that root decision. The Decider still answers it bottom-up for each input. This supersedes ADR 0001. It also removes most of the risk that the Generator leaks the answer into the tree.

## Considered Options

- **One tree per input (ADR 0001).** Rejected: it costs a Generator call per input, and the pilot trees restated facts from the input.
- **A tree library searched by input similarity.** Rejected: it keeps the tie to particular inputs.

## Consequences

The work is split into two stages. Stage 1 keeps one tree per root decision, with no search, and tests whether answering a tree improves the Decider at all. Each (root decision, width) gets three trees, one per Generator seed, and results report the mean and the spread across trees, so that a single bad tree cannot decide a result. Stage 1 is compared against the Decider alone, the Generator alone, and the Decider with the tree's sub-decisions given as context but not answered. Stage 2 is built only if Stage 1 shows gains. It adds a tree library searched by root-decision similarity: embeddings of each stored root decision, brute-force cosine search, and a similarity threshold. On a hit, the stored tree's sub-decisions are reused under the new root decision. On a miss, a new tree is written and added to the library. Stage 2 needs datasets with many distinct but related root decisions, each with labeled inputs.
