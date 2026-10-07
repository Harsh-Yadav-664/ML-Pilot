# 0005. A typed feature spec first, free SQL second

**Status:** Accepted. The spec, its validation and its compiler are implemented (#100, [reference](../features.md), `test_feature_ir.py`). The baseline generator (#55) and the LLM proposer (#56) are `planned`.

## Context

Free-form SQL from an LLM is flexible but hard to guard and hard to trust. Most useful relational features are one of a few shapes: an aggregate over a related table, in a time window, with an optional filter ("orders in the last 30 days where status is paid").

## Decision

The first feature language is a small typed spec (an intermediate representation): aggregate, column, window, filter. It compiles to SQL that respects the cutoff by construction. The LLM proposes specs as structured output. Free SQL is a fallback for features the spec cannot express, and it always goes through the guard in 0004.

## Consequences

- Most features are cutoff-safe without analysis, and are easy to read, compare and deduplicate.
- The spec limits what can be proposed; extending it is a deliberate change with tests.
- Two paths to maintain (spec compiler and SQL guard), but the spec path is the common one.

## Alternatives considered

- **Free SQL only.** More expressive, but every feature needs full analysis and is harder to review.
- **Only fixed DFS-style features, no LLM.** Safe and useful as a baseline (#55), but gives up the main idea of the product.

## Issues

#100 (IR), #55 (baseline features), #56 (LLM proposer).
