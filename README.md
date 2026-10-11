# CivicLens — PySustain 2026

CivicLens is a prototype public-benefits navigator for students in Tamil Nadu.
It searches a local, reviewed JSON catalogue, checks recorded eligibility rules
deterministically, and uses the OpenAI API to extract explicit profile facts and
explain results. A potential match is preliminary screening only, never an
official eligibility decision or guarantee of an award.

## Setup (Windows PowerShell)

Use Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env` to enable AI responses. The
app remains safe when these are missing: it reports that the AI response is
unavailable rather than fabricating one. Do not enter Aadhaar numbers, OTPs,
passwords, bank details, or identity documents. User messages and relevant
profile details are sent to the configured OpenAI API; its data policies apply.

Start the Streamlit app:

```powershell
python -m streamlit run app.py
```

The default catalogue is `data/schemes.json`. Read [data/README.md](data/README.md)
for its schema, verification status, limitations, and update guidance.

## Tests

Install the project dependencies above, then run:

```powershell
python -m pytest -q
```

Tests use synthetic fixtures and do not require API credentials or network
access.
