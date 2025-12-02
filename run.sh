#!/bin/bash
# E2E Evaluation Pipeline - Comprehensive Project Evaluation
# Supports: Gemini CLI (default) or OpenAI Codex CLI
# Uses: uv, ruff, tree

set -e

cd "$(dirname "$0")"

# Default values
LIMIT=""
TIMEOUT=300
NO_CLEAN=""
AI_PROVIDER="gemini"

# Parse arguments
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
            echo "  --ai <gemini|codex>  AI provider to use (default: gemini)"
            echo "  --limit N            Process only first N submissions"
            echo "  --timeout S          Timeout per command in seconds (default: 300)"
            echo "  --no-clean           Don't clean work directory before starting"
            echo "  -h, --help           Show this help"
            echo ""
            echo "Examples:"
            echo "  ./run.sh --limit 5              # Test with 5 submissions using Gemini"
            echo "  ./run.sh --ai codex --limit 5   # Use Codex instead"
            echo "  ./run.sh                        # Run all submissions"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set up AI-specific configuration
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
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           🎓 EASS Student Project Evaluator v2.0             ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  AI Provider: ${AI_PROVIDER}                                         ║"
echo "║  Tools: uv · ruff · ${AI_TOOL} · tree                            ║"
echo "║  Timeout: ${TIMEOUT}s per command                                  ║"
[ -n "$LIMIT" ] && echo "║  Limit: first ${LIMIT#--limit } submissions                               ║"
[ -z "$NO_CLEAN" ] && echo "║  Clean: work directory will be cleaned                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
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
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ✅ Pipeline Complete!                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  📊 REPORT.md       - Full analysis with insights            ║"
echo "║  📋 index.md        - Quick ranking overview                 ║"
echo "║  📈 ranked_list.csv - Sortable spreadsheet data              ║"
echo "║  📁 evaluation.json - Raw data for processing                ║"
echo "║  🔍 ${SCORE_FILE}     - Detailed AI evaluation per project     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "👉 Open results/REPORT.md for the detailed report"
echo ""
