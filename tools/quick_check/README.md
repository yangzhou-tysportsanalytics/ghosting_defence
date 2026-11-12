# Quick check (gdq-1)

Offline single-page tool for the quick check of matchups and events (decision D-036). It is a
template: `scripts/make_quick_check.py` writes one page per annotator with the items embedded
(`/*__GDQ_DATA__*/null` is replaced) under `data/annotations/quick/gdq_v1/` (git-ignored: the
pages contain coordinates). Sample: `scripts/select_quick_check.py`; evaluation:
`scripts/quick_check_eval.py`.

- Part 1: whom each defender guards at two moments of a possession (no model output shown).
- Part 2: short clips with one defender (help) or two (screen) ringed: did the help or the
  switch happen? Some clips are decoys without a detected event.
- Keys: Space play / pause, R replay, arrows step, 1-4 answer and Enter next (Part 2).
- Progress is kept in the browser's local storage; "Download my answers" writes
  `<code>_gdq_v1_answers.json`. Active time per item is recorded (idle > 90 s not counted).
