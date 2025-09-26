# HeadAgency

Simple CLI wrapper that sends your instructions to an LLM while keeping the base prompt and secrets easy to manage.

## Project layout

- `prompts/head_agency_prompt.txt` – system prompt HeadAgency uses. Edit this to shape the agency's personality and policy.
- `secrets/llm_api_key.txt.example` – optional fallback store for your LLM API key if you prefer not to use environment variables.
- `src/head_agency/` – Python package with the CLI entry point and helper classes.

## Getting started

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -e .
   ```
3. Export your API key (recommended):
   ```bash
   export HEAD_AGENCY_API_KEY="sk-your-real-key"
   ```
   The CLI reads the `HEAD_AGENCY_API_KEY` variable by default. For shells like fish or PowerShell, use the appropriate syntax.
4. (Optional) If you cannot use environment variables, copy the template file and add your key as a fallback:
   ```bash
   cp secrets/llm_api_key.txt.example secrets/llm_api_key.txt
   echo "sk-your-real-key" > secrets/llm_api_key.txt
   ```

## Usage

- Edit the system prompt in `prompts/head_agency_prompt.txt` whenever you want to change the agency's behaviour.
- Run the CLI once with a single message:
  ```bash
  head-agency "오늘 일정 정리해줘"
  ```
- Or launch the interactive shell (default when no message is provided):
  ```bash
  head-agency
  ```

### Options

- `--prompt PATH` – point to an alternative prompt file.
- `--key-env NAME` – change the environment variable that stores the API key (default `HEAD_AGENCY_API_KEY`).
- `--key-file PATH` – fallback key file if the environment variable is not set.
- `--model MODEL_ID` – override the default `gpt-4o-mini` model.
- `--temperature VALUE` – adjust sampling temperature (0–2).

## Notes

- The project uses the official `openai` Python package. Install updates as required by your chosen model.
- The actual fallback key file `secrets/llm_api_key.txt` remains ignored by Git via `.gitignore`.

