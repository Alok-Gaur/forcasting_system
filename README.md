## Quick Start

This project uses **[uv](https://github.com/astral-sh/uv)** for lightning-fast dependency management and environment handling.

---

### 1. Install `uv`

If you don't have `uv` installed, use one of the following methods:

#### **Pythonic Way (PIP)**

```bash
pip install uv

```

#### **macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```

#### **Windows**

```powershell
powershell -ExecutionPolicy ByRef -c "irm https://astral.sh/uv/install.ps1 | iex"

```

---

### 2. Setup and Run

You do **not** need to create a virtual environment manually. `uv` handles everything with a single command:

```bash
uv run fastapi dev main.py

```

> **Note:** When you run this, `uv` will automatically:
> * Create a `.venv` (virtual environment).
> * Install the exact dependencies from the `uv.lock` file.
> * Start the FastAPI development server.
> 
>
