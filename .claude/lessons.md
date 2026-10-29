# Lessons

## State the triage call out loud

- When ~/code/CLAUDE.md applies, classify the task (feature / small / trivial) at the start and say whether a Workflow will be used and why. Don't silently run feature-sized work single-agent.
- Delegate research reads (model cards, reference code) to subagents so file dumps stay out of the main context.

## Tests must not restate the implementation

- A test whose expected values come from the same formula as the code (e.g. an index helper copying the layout) proves nothing. Use an independent oracle: a golden fixture from a different implementation (numpy reshape/permute, a jinja render of the real chat template, ORT CPU output).
- Put pure state machines (decode-loop mode switching) behind a port so they can be unit-tested with a fake model, rather than testing them only end-to-end.
