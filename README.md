<p align="center">
  <img src="assets/icon.png" alt="sowr icon" width="160">
</p>


<div align="center">
  
  # sowr
  **System One With Reasoning**

</div>


A research project on whether a fast decision model (the *Decider*) gets better when it first answers simpler sub-decisions.

A light LLM (the *Generator*) writes a tree of sub-decisions under a root decision once, from the root decision and a few unlabeled example inputs. The tree has a set **width** (children per node) and **depth** (levels below the root). It is then reused for every input: the Decider answers it bottom-up, so answering stays on fast Decider calls only. Each parent is answered with its children's answers and confidences passed in as extra context, and the root decision is answered last. The experiment compares this against the Decider alone, the Generator alone, and the Decider with the sub-decisions given as context but not answered, across a width × depth grid.

See [GLOSSARY.md](GLOSSARY.md) for terms and [docs/adr/](docs/adr/) for design decisions.

## Usage

Put both API keys in `.env` (it is gitignored). `uv run --env-file .env` loads them:

```sh
OPENROUTER_API_KEY=...   # Generator
PERPLEXITY_API_KEY=...   # Decider
```

Then run:

```sh
uv run --env-file .env sowr datasets/example --baseline-repeats 3
uv run pytest
```

A dataset is a directory with `root.json` (the root decision), `items.jsonl` (evaluation items, `{"id", "state", "label"}` per line) and `validation.jsonl` (held-out items in the same format, whose inputs the Generator sees as examples). Results go to `runs/results.jsonl`, and trees are cached under `runs/trees/`, one per width and Generator seed.

## Status

Stage 1 runs end to end. The example dataset only checks that the pipeline works: it has two easy items, and every condition gets both right. Comparing conditions needs a real labeled dataset that is hard enough for the Decider alone to make mistakes, plus an analysis of accuracy and calls per condition, with the mean and spread across the three trees.
