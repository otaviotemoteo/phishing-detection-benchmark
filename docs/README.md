# Documentation

Technical documentation for the Phishing Detection Benchmark.

## Contents

- **[RESULTS.md](RESULTS.md)**: every result, every figure, and what they mean.
  The place to start if you want to know what came out.
- **[SETUP.md](SETUP.md)**: installing, fetching the datasets, running the
  pipeline whole or one phase at a time, and the evidence behind the
  reproducibility claim.
- **[DEVELOPMENT.md](DEVELOPMENT.md)**: the full development guide. Roadmap,
  stack, conventions, reproducibility procedures, cost measurement, templates.
  The authoritative reference for anyone working on the project.
- **[DECISIONS.md](DECISIONS.md)**: decision records for every non-trivial
  methodological choice, including the phase that was deliberately dropped.
- **[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)**: chronological lab journal of the
  experiment sessions.

## When to read what

| If you want to | Read |
|---|---|
| Know what the project found | [`RESULTS.md`](RESULTS.md) |
| Run it yourself | [`SETUP.md`](SETUP.md) |
| Understand the project from the outside | [`../README.md`](../README.md) |
| Contribute code, or follow the conventions | [`DEVELOPMENT.md`](DEVELOPMENT.md) |
| Understand why a specific choice was made | [`DECISIONS.md`](DECISIONS.md) |
| See what happened in a given session | [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) |

Before making a change, read [`DEVELOPMENT.md`](DEVELOPMENT.md) first. It is
binding: a deviation from it needs a justification, and the justification is
recorded in [`DECISIONS.md`](DECISIONS.md) rather than left in a commit message.
