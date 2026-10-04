# Contributing to Sky Beats

Thank you for your interest in contributing to **Sky Beats**! This document
describes how to propose changes, report problems, and get your contribution
merged. By participating you agree to follow our
[Code of Conduct](CONDUCT.md).

## Ways to contribute

- **Report a bug** — something crashes, a workflow rule fails, a plot looks wrong.
- **Suggest an enhancement** — a new research question, a better statistical
  method, clearer documentation.
- **Improve the documentation** — README, docstrings, notebooks.
- **Submit code** — bug fixes, new workflow components, tests.

## Reporting bugs and suggesting features

We use the [issue tracker](https://gitup.uni-potsdam.de/mammadov/sky-beats/-/issues)
for all discussion of bugs and new ideas.

Before opening an issue, please search the existing issues (open *and* closed)
to avoid duplicates. When you open one:

- Use a clear, descriptive title.
- For bugs: describe the exact steps to reproduce, what you expected, and what
  happened instead. Include the command you ran, the error output, and your
  Python/OS versions.
- For features: explain the motivation — what problem does it solve?
- Add appropriate labels and, if you plan to work on it yourself, assign it to
  yourself so work is not duplicated.

## Development setup

1. Fork/clone the repository and create a virtual environment:

   ```bash
   git clone https://gitup.uni-potsdam.de/mammadov/sky-beats.git
   cd sky-beats
   python3 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Obtain the input data (see [data/README.md](data/README.md)) and run the
   workflow from the project root to check everything works:

   ```bash
   snakemake --cores 1
   ```

3. Run the test suite:

   ```bash
   python -m pytest tests/
   ```

## Making changes

1. **Open or claim an issue** describing what you want to change.
2. **Create a feature branch** from `main` — never commit to `main` directly:

   ```bash
   git checkout -b <type>/<short-description> main
   ```

   e.g. `feature/rq4-quantile-plot`, `fix/rq2-date-parsing`, `docs/data-readme`.
3. **Make small, focused commits.** Follow the project's commit message
   convention (`type(scope): imperative summary`):

   ```
   feat(rq3): add genre shift bar plot
   fix(rq8): anchor delta-R2 annotation to bar tops
   test(rq1): cover empty-season edge case
   docs: update README run instructions
   refactor(workflow): migrate RQ5 to workflow/ layout
   chore: update project dependencies
   ```

4. **Follow the existing code style:**
   - every script is a command-line tool with `argparse`, a module docstring
     and a usage example;
   - scripts log to their component's `logs/` directory and validate their
     inputs, exiting with an `ERROR: ...` message on bad input;
   - configuration lives in `config/config.yaml`, not hard-coded in scripts;
   - plots use the shared style via `src/common/utils.py::set_plot_params`.
5. **Add or update tests** in `tests/` for any component you touch. Use small,
   hand-checkable synthetic DataFrames as in the existing tests.
6. **Verify before pushing:** the test suite passes and the affected workflow
   rule(s) run end to end via `snakemake --cores 1 <target>`.

## Submitting a merge request

1. Push your branch and open a **merge request** against `main` on Git.UP.
2. In the description: say *what* changed and *why*, and reference the issue it
   resolves (e.g. `Closes #12`).
3. Keep merge requests small and self-contained — one topic per MR.
4. A team member reviews the MR; please respond to review comments with
   follow-up commits rather than force-pushes, so the discussion stays
   readable.
5. Once approved and the pipeline is green, the MR is merged and the issue is
   closed.

## What not to commit

- Generated files: everything under `results/` and intermediate files in
  `data/` (the three small example inputs listed in
  [data/README.md](data/README.md) are the deliberate exception).
- Large raw datasets (`data/raw/`), virtual environments, log files — see
  `.gitignore`.
- Credentials of any kind (e.g. Kaggle API tokens).

## Questions?

Open an issue on the
[issue tracker](https://gitup.uni-potsdam.de/mammadov/sky-beats/-/issues) —
we're happy to help.
