# Quick Start
This project uses uv for lightning-fast dependency management.

1. Install uv
If you don't have uv installed:

# Pythonic Way (CMD)
`pip install uv`

# macOS/Linux
`curl -LsSf https://astral.sh/uv/install.sh | sh`


# Windows
`powershell -ExecutionPolicy ByRef -c "irm https://astral.sh/uv/install.ps1 | iex"`


2. Setup and Run
You don't even need to create a virtual environment manually. Just run:

`uv run fastapi dev main.py`

uv will automatically create a .venv, install the exact dependencies from uv.lock, and start the server.