# CivicLens — PySustain 2026

CivicLens is a small command-line assistant for finding relevant public
assistance schemes in a local JSON dataset. Search is performed locally, and
the response formatter does not send user messages or profile data to an
external service.

## Setup

Use Python 3.11 or newer:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

The default data file is `data/schemes.json`. Set `CIVICLENS_DATA_DIR` in
`.env` to use a different directory. See [data/README.md](data/README.md) for
the expected JSON format.

## Tests

Run the test suite with:

```powershell
python -m unittest discover -s tests -v
```
