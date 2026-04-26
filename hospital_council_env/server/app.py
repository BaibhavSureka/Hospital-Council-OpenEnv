# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI server for the Hospital Council OpenEnv environment."""

import os

from fastapi.responses import RedirectResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import HospitalCouncilAction, HospitalCouncilObservation
    from .hospital_council_env_environment import HospitalCouncilEnvironment
except ImportError:
    from models import HospitalCouncilAction, HospitalCouncilObservation
    from server.hospital_council_env_environment import HospitalCouncilEnvironment


# Create the app with web interface and README integration.
app = create_app(
    HospitalCouncilEnvironment,
    HospitalCouncilAction,
    HospitalCouncilObservation,
    env_name="hospital_council_env",
    max_concurrent_envs=4,
)


@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/web", status_code=307)


@app.get("/status", include_in_schema=False)
def status() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "hospital-council-openenv",
        "ui": "/web",
        "api_health": "/health",
        "mode": "openenv-fastapi",
    }


def main() -> None:
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m hospital_council_env.server.app

    For production deployments, consider using uvicorn directly with multiple workers:
        uvicorn hospital_council_env.server.app:app --workers 4
    """
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
