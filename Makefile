# Everything runs inside the uv environment. Without make: bash scripts/pipeline_all.sh
PY := uv run python
export PYTHONUTF8=1

.PHONY: sensitivity-second-chance setup lint test pipeline pipeline-all figures release-data release-check

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

test:
	uv run pytest -q

# Per-game steps; GAME_SET = tiny | small | all.
GAME_SET ?= small
WORKERS ?= 1
pipeline:
	$(PY) scripts/phase1_possessions.py --game-set $(GAME_SET) --workers $(WORKERS)
	$(PY) scripts/phase2_matchups.py --game-set $(GAME_SET)
	$(PY) scripts/phase2_events.py --game-set $(GAME_SET)
	$(PY) scripts/phase2_screen_events.py --game-set $(GAME_SET)
	$(PY) scripts/phase2_screen_rates.py --game-set $(GAME_SET)
	$(PY) scripts/phase4_rule_ghost.py --game-set $(GAME_SET)

# Season-level analyses (game set "all" only); about 3 h on a 16-core laptop CPU.
pipeline-all: GAME_SET = all
pipeline-all: pipeline
	$(PY) scripts/phase4_xfg_data.py
	$(PY) scripts/phase4_xfg.py
	$(PY) scripts/phase4_ghost_points.py
	$(PY) scripts/analysis_team_screen_defense.py --game-set all
	$(PY) scripts/analysis_team_validity.py
	$(PY) scripts/analysis_help_timing.py
	$(PY) scripts/analysis_help_value.py
	$(PY) scripts/phase4_phase_context.py --defs A
	$(PY) scripts/phase4_phase_context.py --defs B
	$(PY) scripts/phase4_reliability.py --game-set all
	$(PY) scripts/phase4_reliability.py --game-set all --context phases
	$(PY) scripts/phase4_hier.py
	$(PY) scripts/phase4_hier.py --context phases
	$(PY) scripts/phase4_breakdowns.py
	$(PY) scripts/phase4_breakdowns.py --quantile 0.95
	$(PY) scripts/phase5_predictive_validity.py
	$(PY) scripts/phase5_predictive_validity.py --context phases
	$(PY) scripts/phase2_em_init_sensitivity.py --game-set all
	$(PY) scripts/make_figures.py

figures:
	$(PY) scripts/make_figures.py

release-data:
	$(PY) scripts/build_derived_release.py

release-check:
	$(PY) scripts/check_release.py data/derived_release

# Second-chance sensitivity (D-018) in runs/sc_sensitivity; needs pipeline-all first (~1 h).
sensitivity-second-chance:
	bash scripts/second_chance_sensitivity.sh
