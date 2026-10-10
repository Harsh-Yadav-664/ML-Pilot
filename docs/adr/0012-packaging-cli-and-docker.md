# 0012. Packaging: a command line over the same services, a wheel, one Docker image

**Status:** Accepted. Implemented (#68). CI job `pip-install` runs the wheel; the Docker image and the release workflow have not been run by anyone yet (see Consequences).

## Context

The five-minute path is "install, point at a database, run". The code was only runnable from a clone with `python start.py`, and the API is the only front door. A second front door must not be a second implementation: AGENTS.md wants one SQL guard, one point-in-time guard and one job runner.

## Decision

- **The `mlpilot` command (Typer, MIT) calls the services in `backend/app/services/` directly**, in the same process, through the same database sessions and the same job runner (`mlpilot run` starts the in-process `JobWorker`). There is no HTTP between the command and the work, and no second code path to keep in step. A refusal that a service raises as an `HTTPException` becomes a message and exit code 1.
- **State is shared with the web UI.** The command uses the same metadata database and the same "newest project" as the UI, so `mlpilot run` followed by `mlpilot ui` shows the run. State lives in `~/.mlpilot` (`MLPILOT_HOME`), because an installed package must not write into `site-packages` or the current folder. The password of a connection is only ever taken from an environment variable.
- **A thin `mlpilot` package, not a move.** `app/`, `ai/` and `ml/` stay where they are and are shipped as top-level packages of the same wheel; `mlpilot/` holds the command line, the Alembic revisions (moved from `backend/migrations/`) and the built UI. The version is `mlpilot.__version__`, the one place a release tag is checked against.
- **The UI is one self-contained file** built for the same origin (`/api/v1`) by `scripts/build_ui.py`, shipped as package data and served by FastAPI at `/`. The access token is not baked in: `mlpilot ui` prints the address with the token after a `#` (a fragment never reaches a server log), and the page moves it into local storage and out of the address bar.
- **One Docker image** (API and built UI), built from the same wheel. Inside the container the server must listen on all interfaces for Docker to forward the port; what keeps it off the network is publishing it as `-p 127.0.0.1:8000:8000`, which the compose file and the README do. Everything is on a non-root user with state in `/data`.
- **Releases** are tags (`v<mlpilot.__version__>`): a workflow publishes the sdist and wheel to PyPI (trusted publishing) and the image to GHCR.

## Consequences

- `app`, `ai`, `ml` and `mlpilot` become top-level names in the environment of whoever installs the wheel. They are generic; the price of not moving 300 files in this issue. A later namespace move (`mlpilot.app`, ...) is possible and mechanical, but touches every import and test.
- The old CSV sample (`backend/datasets/`) is not in the wheel; the legacy "sample data" button fails loudly in an installed copy.
- `requirements.txt` is the one list of runtime dependencies for CI, `start.py` and the wheel (read by setuptools); test tools moved to `requirements-dev.txt` so the wheel does not depend on pytest.
- Not verified by anyone yet: the Docker image (the CI job `docker-image` is its first run), the release workflow, and installing from PyPI (the project name is not reserved and nothing has been published).

## Alternatives considered

- **A CLI that talks HTTP to a running API.** Needs a server and a token for the simplest use, and tests the HTTP layer instead of the product. Rejected for local use; the API stays for the UI.
- **Move `app/`, `ai/`, `ml/` under `mlpilot/`.** The clean end state, a very large diff for no new behaviour. Deferred.
- **Bake a token into the UI build or serve it from `/`.** Anyone who can reach the port would get it. Rejected for the fragment.

## Evidence and issues

`backend/tests/integration/test_cli.py`; CI jobs `pip-install` and `docker-image`. Issue: #68.
