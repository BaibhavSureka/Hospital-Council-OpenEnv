# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI server for the Hospital Council OpenEnv environment."""

import os

from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from fastapi import FastAPI

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
def root() -> HTMLResponse:
    """Root endpoint returning HTML landing page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hospital Council OpenEnv</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                   margin: 0; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   min-height: 100vh; }
            .container { max-width: 900px; margin: 0 auto; background: white; 
                        padding: 40px; border-radius: 10px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
            h1 { color: #333; margin-top: 0; }
            p { color: #666; line-height: 1.6; }
            .links { margin-top: 30px; }
            a { display: inline-block; padding: 12px 24px; margin: 8px 8px 8px 0; 
                background: #667eea; color: white; text-decoration: none; border-radius: 5px; 
                transition: background 0.3s; }
            a:hover { background: #764ba2; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; 
                       border-left: 4px solid #667eea; font-family: monospace; }
            .status { color: #27ae60; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 Hospital Council OpenEnv</h1>
            <p><span class="status">✓ Service Running</span></p>
            <p>A long-horizon, multi-agent environment for medical decision-making and coordination.</p>
            
            <h2>Quick Links</h2>
            <div class="links">
                <a href="/web">→ Web Interface</a>
                <a href="/docs">→ API Documentation</a>
                <a href="/status">→ Status</a>
            </div>
            
            <h2>Available Endpoints</h2>
            <div class="endpoint"><strong>POST /env/reset</strong> - Reset environment</div>
            <div class="endpoint"><strong>POST /env/step</strong> - Execute action</div>
            <div class="endpoint"><strong>GET /status</strong> - Service status</div>
            <div class="endpoint"><strong>GET /health</strong> - Health check</div>
            <div class="endpoint"><strong>GET /docs</strong> - Swagger UI</div>
            <div class="endpoint"><strong>GET /web</strong> - Interactive web interface</div>
            
            <h2>About</h2>
            <p>The agent acts as a hospital council coordinator managing:</p>
            <ul>
                <li>Multi-agent negotiation across 5 stakeholders</li>
                <li>Patient state and clinical outcomes</li>
                <li>Long-horizon episode planning</li>
                <li>Real-world medical scenarios</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/robots.txt", include_in_schema=False)
def robots() -> str:
    """Return robots.txt to prevent crawlers from indexing."""
    return "User-agent: *\nDisallow: /"


@app.get("/health", include_in_schema=False)
def health() -> dict[str, object]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "hospital-council-openenv",
        "version": "0.2.0"
    }


@app.get("/status", include_in_schema=False)
def status() -> dict[str, object]:
    """Service status endpoint."""
    return {
        "status": "ok",
        "service": "hospital-council-openenv",
        "ui": "/web",
        "api_docs": "/docs",
        "health": "/health",
        "mode": "openenv-fastapi",
        "endpoints": {
            "reset": "POST /env/reset",
            "step": "POST /env/step",
            "status": "GET /status",
            "health": "GET /health",
            "docs": "GET /docs",
            "ui": "GET /web"
        }
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
