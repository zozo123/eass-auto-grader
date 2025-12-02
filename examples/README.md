# Example Inputs (sanitized)

This folder contains sanitized example input files showing the expected structure for inputs used by the grader. These files contain placeholder values only and do not include any private or real student data.

Files:
- `submission.csv` — CSV with headers and anonymized rows.
- `codex_prompt.example.txt` — Example template for Codex prompts (placeholders).
- `gemini_prompt.example.txt` — Example template for Gemini prompts (placeholders).
- `gemini_settings.example.json` — Example `.gemini/settings.json` with MCP servers and Gemini 3 preview enabled.

Use these files as templates when creating your own `submission.csv` or prompt files. Do not commit real student data to the repository.

## Gemini 3 & MCP Setup

To enable Gemini 3 and MCP servers:

1. Copy `gemini_settings.example.json` to `.gemini/settings.json` in your project root.
2. Set `"previewFeatures": true` to enable Gemini 3.
3. Ensure `GITHUB_TOKEN` env var is set for the GitHub MCP server.
4. Run `gemini` CLI — it should show "Using: X MCP servers".

If you see `Connection failed for 'filesystem'`, check that Node.js and npx are installed, and that the path in args is valid (use `.` for current directory).

If you need additional example inputs, copy a file here and replace real data with placeholders.
