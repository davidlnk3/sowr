<p align="center">
  <img src="assets/icon.png" alt="sowr icon" width="160">
</p>


<div align="center">
  
  # sowr
  **System One With Reasoning**

</div>


A research project on whether a fast decision model (the *Decider*) gets better when it first answers simpler sub-decisions.

For each input, a light LLM (the *Generator*) builds a tree of sub-decisions under the root decision, with a set **width** (children per node) and **depth** (levels below the root). The Decider answers the tree bottom-up. Each parent is answered with its children's answers and confidences passed in as extra context, and the root decision is answered last. The experiment compares this against the Decider alone and the Generator alone, across a width × depth grid.

See [GLOSSARY.md](GLOSSARY.md) for terms and [docs/adr/](docs/adr/) for design decisions.

## Usage

Needs `OPENROUTER_API_KEY` (Generator) and `PERPLEXITY_API_KEY` (Decider) in the environment or in `.env`.

```sh
uv run --env-file .env sowr datasets/example --baseline-repeats 3
uv run pytest
```

A dataset is a directory with `root.json` (the root decision) and `items.jsonl` (`{"id", "state", "label"}` per line). Results go to `runs/results.jsonl`, and generated trees are cached under `runs/trees/`.
