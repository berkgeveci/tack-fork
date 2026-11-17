# Plans

A plan is written before work that has decisions in it: a change of API,
work that spans several pull requests, or a choice between designs that
reviewers should weigh in on before the code exists. Bug fixes and small
features do not need one.

Each plan is a page here, proposed in a pull request of its own, so that
reviewers can comment on any line of it and suggest changes. Merging the pull
request accepts the plan. The pull requests that carry it out then link the
plan, and their review asks two questions: is this what the plan called for,
and is the risk acceptable? When the work is done, its plan records how it
came out (the measurements, what changed on the way) and its status says so.

A plan opens with its status, its issue and its date:

- **Proposed**: in review, in its own pull request.
- **Accepted**: merged; the work is under way.
- **Done**: carried out; the plan records the outcome.
- **Superseded**: replaced, by the plan it names.

Each decision in a plan is numbered (D1, D2, ...) and carries a
recommendation, so a review can answer "yes", "no" or "other" per decision.

## Plans

| Plan | Issue | Status |
|---|---|---|
| [Cut filters' memory](cut-filters-memory.md) | [#318](https://github.com/Kitware/tack/issues/318) | Accepted |
| [Inspecting device-class objects](inspect-template.md) | [#324](https://github.com/Kitware/tack/issues/324) | Done |
