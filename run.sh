#!/bin/bash
# EASS Student Project Evaluator
# Supports: Gemini CLI (default) or Codex CLI

set -e

cd "$(dirname "$0")"

LIMIT=""
TIMEOUT=300
NO_CLEAN=""
AI_PROVIDER="gemini"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ai)
            AI_PROVIDER="$2"
            if [[ "$AI_PROVIDER" != "gemini" && "$AI_PROVIDER" != "codex" ]]; then
                echo "Error: --ai must be 'gemini' or 'codex'"
                exit 1
            fi
            shift 2
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --timeout)
            TIMEOUT=$2
            shift 2
            ;;
        --no-clean)
            NO_CLEAN="--no-clean"
            shift
            ;;
        -h|--help)
            echo "Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --ai <gemini|codex>  AI provider (default: gemini)"
            echo "  --limit N            Process first N submissions"
            echo "  --timeout S          Timeout per command (default: 300)"
            echo "  --no-clean           Keep work directory"
            echo "  -h, --help           Show this help"
            echo ""
            echo "Examples:"
            echo "  ./run.sh --limit 5"
            echo "  ./run.sh --ai codex --limit 5"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ "$AI_PROVIDER" == "gemini" ]]; then
    PROMPT_FILE="$(pwd)/gemini_prompt.txt"
    AI_COMMAND="cat \"$PROMPT_FILE\" | gemini --include-directories \"{repo_dir}\" --output-format json > \"{artifacts_dir}/gemini.json\""
    SCORE_FILE="gemini.json"
    AI_TOOL="gemini"
else
    PROMPT_FILE="$(pwd)/codex_prompt.txt"
    AI_COMMAND="codex --approval-mode full-auto --quiet \"{repo_dir}\" -p \"$PROMPT_FILE\" --output \"{artifacts_dir}/codex.json\""
    SCORE_FILE="codex.json"
    AI_TOOL="codex"
fi

echo ""
echo "EASS Student Project Evaluator"
echo "=============================="
echo "Provider: ${AI_PROVIDER}"
echo "Timeout:  ${TIMEOUT}s"
[ -n "$LIMIT" ] && echo "Limit:    ${LIMIT#--limit }"
[ -z "$NO_CLEAN" ] && echo "Clean:    yes"
echo ""

uv run python scripts/run_evaluation.py \
    $NO_CLEAN \
    $LIMIT \
    --keep-clones \
    --format-check \
    --format-command "/opt/homebrew/bin/ruff format --check ." \
    --gemini-command "$AI_COMMAND" \
    --score-file "$SCORE_FILE" \
    --score-key final_score \
    --timeout-seconds "$TIMEOUT"

echo ""
echo "Complete. Results:"
echo "  results/REPORT.md       - Full analysis"
echo "  results/index.md        - Ranking overview"
echo "  results/ranked_list.csv - Spreadsheet data"
echo "  results/evaluation.json - Raw JSON data"
echo ""
