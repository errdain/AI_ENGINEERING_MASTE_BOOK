# CANONICAL AI ENGINEERING GLOSSARY

## Agent

An AI system in which a model participates in deciding
what actions to take toward a goal within defined
constraints.

Do not use "agent" to describe every LLM application.

---

## Tool

An external capability exposed to an AI system that can
be invoked using structured inputs.

Examples:

database query
API request
calculator
search
file operation

---

## Context

Information available to the model during a particular
inference request.

Do not automatically call persistent information
"context."

---

## Memory

Information stored and later retrieved to influence
future interactions or reasoning.

---

## Token

A discrete unit produced by a tokenizer and represented
by an integer token ID before being mapped to a learned
representation.

---

## Embedding

A learned numerical vector representation intended to
capture useful properties or relationships of an input.

---

## Retrieval

Selecting relevant information from an external
knowledge source for a query.

---

## RAG

Retrieval-Augmented Generation.

Retrieval supplies external evidence/context to a
generative model before or during answer generation.

---

## Workflow

A controlled sequence/graph of processing steps.

A workflow may contain LLM calls without being an
autonomous agent.

---

## Tool Calling

A mechanism through which a model/application selects
an external function or capability and produces
structured arguments for its invocation.

---

## Hallucination

Model-generated content that is unsupported, incorrect
or fabricated relative to the relevant evidence or
ground truth.

---

## Grounding

Connecting a model response to supplied or retrieved
evidence.

---

## Evaluation

Systematic measurement of an AI component or system
against defined criteria.

---

## Observability

Ability to understand system behavior through signals
such as traces, metrics and logs.

---

## Guardrail

A control used to constrain, validate, filter or govern
inputs, outputs or actions.

---

## Model Gateway

A service abstraction between applications and one or
more model providers.

---

## MCP

Model Context Protocol.

Treat protocol details as version-sensitive and verify
against the current specification before publication.

---

## A2A / Agent Interoperability

Mechanisms allowing independent agentic systems to
discover capabilities, exchange tasks/messages and
coordinate work.

Treat implementation-specific standards as
version-sensitive.
