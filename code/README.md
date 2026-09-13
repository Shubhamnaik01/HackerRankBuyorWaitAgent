# Buy or Wait? Solution

This package contains the runnable solution for the HackerRank Orchestrate September 2026 **Buy or Wait?** challenge.

## Requirements

- Python 3.11 or newer
- Dependencies listed in `code/requirements.txt`
- `OPENAI_API_KEY` set in the environment for the final evidence-enabled run

From the repository root, install dependencies with:

```text
python -m pip install -r code/requirements.txt
```

Never place an API key in source code or include a `.env` file in `code.zip`.

## Run

Generate the final submission artifacts from the repository root:

```text
python code/main.py
```

This processes `dataset/requests.csv` and creates the linked artifacts:

- `output.csv` at the repository root
- `code/evaluation/usage_report.md`

For development calculations against the solved samples without writing submission artifacts:

```text
python code/main.py --samples
```

Validate a generated submission independently:

```text
python code/evaluation/submission_validator.py output.csv
```

Check the files that are safe to include in `code.zip`:

```text
python code/evaluation/package_check.py
```

## Architecture

Deterministic Python is authoritative for data loading, currency conversion, recurrence inference, 90-day forecasting, minimum-balance safety, payment-plan construction, ranking, validation, and final decisions. OpenAI is used only when required to extract constrained financial facts from relevant unstructured messages or linked images; model output is validated before it reaches the financial engine.

The final run creates an isolated evidence cache and does not require or consume a pre-populated development cache.
