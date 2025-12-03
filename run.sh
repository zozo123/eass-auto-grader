#!/bin/bash
# EASS Student Project Evaluator
# =============================================================================
# Supports multiple AI providers:
#   gemini-cli  - Gemini CLI tool (uses OAuth, no API key needed)
#   gemini-api  - Direct Gemini API via HTTP (requires GEMINI_API_KEY)
#   codex       - OpenAI Codex CLI
#   local       - Local LLM via Ollama/llama-server
# =============================================================================

set -e

cd "$(dirname "$0")"

# =============================================================================
# Load environment variables from .env
# =============================================================================
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# =============================================================================
# Default configuration (can be overridden by .env or command line)
# =============================================================================
LIMIT=""
TIMEOUT="${AI_TIMEOUT:-600}"
NO_CLEAN=""
AI_PROVIDER="${AI_PROVIDER:-gemini-cli}"
GEMINI_CONFIG_DIR="$(pwd)/.gemini"

# Model configuration from .env (with sensible defaults)
GEMINI_CLI_MODEL="${GEMINI_CLI_MODEL:-gemini-2.5-flash}"
GEMINI_API_MODEL="${GEMINI_API_MODEL:-gemini-2.5-flash}"
GEMINI_API_ENDPOINT="${GEMINI_API_ENDPOINT:-https://generativelanguage.googleapis.com/v1beta}"
# CODEX_MODEL: leave unset to use user's ~/.codex/config.toml default, or set to override
CODEX_MODEL="${CODEX_MODEL:-}"
AI_FALLBACK_CHAIN="${AI_FALLBACK_CHAIN:-codex}"

# =============================================================================
# Parse command line arguments
# =============================================================================
PARALLEL=""
WORKERS="${WORKERS:-4}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ai)
            AI_PROVIDER="$2"
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
        --parallel|-p)
            PARALLEL="yes"
            shift
            ;;
        --workers|-w)
            WORKERS="$2"
            shift 2
            ;;
        --model)
            # Override the model for the selected provider
            case "$AI_PROVIDER" in
                gemini-cli) GEMINI_CLI_MODEL="$2" ;;
                gemini-api) GEMINI_API_MODEL="$2" ;;
                codex) CODEX_MODEL="$2" ;;
            esac
            shift 2
            ;;
        -h|--help)
            cat << 'EOF'
EASS Student Project Evaluator
==============================

Usage: ./run.sh [OPTIONS]

AI Providers:
  gemini-cli   Gemini CLI tool (uses OAuth, no API key needed) [DEFAULT]
  gemini-api   Direct Gemini API via HTTP (requires GEMINI_API_KEY)
  codex        OpenAI Codex CLI

Options:
  --ai <provider>    AI provider (see above)
  --model <model>    Model to use (overrides .env)
  --limit N          Process first N submissions
  --timeout S        Timeout per command (default: 600)
  --parallel, -p     Run evaluations in parallel
  --workers N, -w N  Number of parallel workers (default: 4)
  --no-clean         Keep work directory
  -h, --help         Show this help

Environment Variables (from .env):
  AI_PROVIDER          Default provider (gemini-cli, gemini-api, codex)
  
  # Gemini CLI (OAuth-based, no key needed)
  GEMINI_CLI_MODEL     Model for CLI (default: gemini-2.5-flash)
  
  # Gemini API (requires API key)
  GEMINI_API_KEY       API key from aistudio.google.com
  GEMINI_API_MODEL     Model for API (default: gemini-2.5-flash)
  
  # Codex CLI
  CODEX_MODEL          Model (default: from ~/.codex/config.toml)

Examples:
  ./run.sh                              # Use Gemini CLI (default)
  ./run.sh --ai gemini-api --limit 5    # Use Gemini API with key
  ./run.sh --ai codex --model o3        # Use Codex with o3 model
  ./run.sh --parallel --workers 4       # Run 4 evaluations in parallel
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate provider
case "$AI_PROVIDER" in
    gemini-cli|gemini-api|codex) ;;
    gemini) AI_PROVIDER="gemini-cli" ;;  # Legacy alias
    *)
        echo "Error: Invalid AI provider '$AI_PROVIDER'"
        echo "Valid options: gemini-cli, gemini-api, codex"
        exit 1
        ;;
esac

# =============================================================================
# Quota/availability checks
# =============================================================================
run_quota_checks() {
    echo "Preflight checks:"
    echo ""

    # Gemini CLI check
    if command -v gemini >/dev/null 2>&1; then
        local g_out g_log
        g_out=$(mktemp)
        g_log=$(mktemp)
        
        if GEMINI_CONFIG_DIR="$GEMINI_CONFIG_DIR" gemini -m "$GEMINI_CLI_MODEL" \
            --output-format json -p 'Return JSON {"status":"ok"}' >"$g_out" 2>"$g_log"; then
            echo "  ✓ Gemini CLI ($GEMINI_CLI_MODEL): OK"
        else
            echo "  ✗ Gemini CLI ($GEMINI_CLI_MODEL): FAIL"
            [[ -s "$g_log" ]] && echo "    $(tail -n 1 "$g_log")"
        fi
        rm -f "$g_out" "$g_log"
    else
        echo "  - Gemini CLI: not installed"
    fi

    # Gemini API check (if key is set)
    if [[ -n "$GEMINI_API_KEY" ]]; then
        local api_response
        api_response=$(curl -s --connect-timeout 5 \
            "${GEMINI_API_ENDPOINT}/models/${GEMINI_API_MODEL}?key=${GEMINI_API_KEY}" 2>/dev/null)
        if echo "$api_response" | grep -q '"name"'; then
            echo "  ✓ Gemini API ($GEMINI_API_MODEL): OK"
        else
            echo "  ✗ Gemini API ($GEMINI_API_MODEL): FAIL"
            echo "$api_response" | grep -o '"message":"[^"]*"' | head -1 | sed 's/"message":"//;s/"$//' | xargs -I{} echo "    {}"
        fi
    else
        echo "  - Gemini API: no GEMINI_API_KEY set"
    fi

    # Codex check
    if command -v codex >/dev/null 2>&1; then
        local c_out c_log model_arg
        c_out=$(mktemp)
        c_log=$(mktemp)
        model_arg=""
        [[ -n "$CODEX_MODEL" ]] && model_arg="-c model=\"$CODEX_MODEL\""
        if echo 'Return {"status":"ok"}' | codex exec --full-auto --color never --skip-git-repo-check \
            $model_arg -c model_reasoning_effort="low" \
            --output-last-message "$c_out" >/dev/null 2>"$c_log"; then
            echo "  ✓ Codex CLI (${CODEX_MODEL:-<config>}): OK"
        else
            echo "  ✗ Codex CLI (${CODEX_MODEL:-<config>}): FAIL"
            [[ -s "$c_log" ]] && echo "    $(tail -n 1 "$c_log")"
        fi
        rm -f "$c_out" "$c_log"
    else
        echo "  - Codex CLI: not installed"
    fi

    echo ""
}

run_quota_checks

# =============================================================================
# Validate selected provider is available
# =============================================================================
validate_provider() {
    case "$AI_PROVIDER" in
        gemini-cli)
            if ! command -v gemini >/dev/null 2>&1; then
                echo "Error: Gemini CLI not installed"
                echo "Install with: npm install -g @google/gemini-cli"
                exit 1
            fi
            ;;
        gemini-api)
            if [[ -z "$GEMINI_API_KEY" ]]; then
                echo "Error: GEMINI_API_KEY not set"
                echo "Get your key from: https://aistudio.google.com/apikey"
                exit 1
            fi
            ;;
        codex)
            if ! command -v codex >/dev/null 2>&1; then
                echo "Error: Codex CLI not installed"
                exit 1
            fi
            ;;
    esac
}

validate_provider

# =============================================================================
# Build AI commands based on provider
# =============================================================================
PROMPT_FILE="$(pwd)/gemini_prompt.txt"
CODEX_PROMPT_FILE="$(pwd)/codex_prompt.txt"

# Define reusable command templates
build_gemini_cli_cmd() {
    echo "GEMINI_CONFIG_DIR=\"$GEMINI_CONFIG_DIR\" gemini -m \"$GEMINI_CLI_MODEL\" --include-directories \"{repo_dir}\" --output-format json < \"$PROMPT_FILE\" > \"{artifacts_dir}/gemini.json\""
}

build_gemini_api_cmd() {
    echo "GEMINI_API_KEY=\"$GEMINI_API_KEY\" GEMINI_CONFIG_DIR=\"$GEMINI_CONFIG_DIR\" gemini -m \"$GEMINI_API_MODEL\" --include-directories \"{repo_dir}\" --output-format json < \"$PROMPT_FILE\" > \"{artifacts_dir}/gemini.json\""
}

build_codex_cmd() {
    local model_arg=""
    if [[ -n "$CODEX_MODEL" ]]; then
        model_arg="-c model=\"$CODEX_MODEL\""
    fi
    echo "codex exec --full-auto --color never --skip-git-repo-check --cd \"{repo_dir}\" \
        $model_arg \
        -c model_reasoning_effort=\"high\" \
        -c mcpServers.filesystem.command=\"npx\" \
        -c 'mcpServers.filesystem.args=[\"-y\",\"@modelcontextprotocol/server-filesystem\",\"{repo_dir}\"]' \
        --output-last-message \"{artifacts_dir}/codex.json\" < \"$CODEX_PROMPT_FILE\""
}

# Determine fallback command based on AI_FALLBACK_CHAIN
get_fallback_cmd() {
    local fallback="$1"
    case "$fallback" in
        codex)
            if command -v codex >/dev/null 2>&1; then
                build_codex_cmd
            fi
            ;;
        gemini-cli)
            if command -v gemini >/dev/null 2>&1; then
                build_gemini_cli_cmd
            fi
            ;;
        gemini-api)
            if [[ -n "$GEMINI_API_KEY" ]] && command -v gemini >/dev/null 2>&1; then
                build_gemini_api_cmd
            fi
            ;;
    esac
}

case "$AI_PROVIDER" in
    gemini-cli)
        # Gemini CLI - uses OAuth, no API key needed
        AI_COMMAND="$(build_gemini_cli_cmd)"
        SCORE_FILE="gemini.json"
        PRIMARY_ARG=(--gemini-command "$AI_COMMAND")
        
        # Fallback to Codex if configured
        FALLBACK_CMD="$(get_fallback_cmd "$AI_FALLBACK_CHAIN")"
        if [[ -n "$FALLBACK_CMD" ]]; then
            FALLBACK_ARGS=(--fallback-codex --codex-command "$FALLBACK_CMD")
        else
            FALLBACK_ARGS=()
        fi
        ;;
        
    gemini-api)
        # Gemini API - uses HTTP with API key
        AI_COMMAND="$(build_gemini_api_cmd)"
        SCORE_FILE="gemini.json"
        PRIMARY_ARG=(--gemini-command "$AI_COMMAND")
        
        # Fallback chain
        FALLBACK_CMD="$(get_fallback_cmd "$AI_FALLBACK_CHAIN")"
        if [[ -n "$FALLBACK_CMD" ]]; then
            FALLBACK_ARGS=(--fallback-codex --codex-command "$FALLBACK_CMD")
        else
            FALLBACK_ARGS=()
        fi
        ;;
        
    codex)
        # Codex CLI - use as primary with Gemini fallback
        AI_COMMAND="$(build_codex_cmd)"
        SCORE_FILE="codex.json"
        
        # For Codex primary, we use it as the "gemini" slot and optionally fallback to actual Gemini
        PRIMARY_ARG=(--gemini-command "$AI_COMMAND")
        
        # Fallback to Gemini if configured
        if [[ "$AI_FALLBACK_CHAIN" == *"gemini"* ]]; then
            if [[ -n "$GEMINI_API_KEY" ]] && command -v gemini >/dev/null 2>&1; then
                FALLBACK_CMD="$(build_gemini_api_cmd)"
                FALLBACK_ARGS=(--fallback-codex --codex-command "$FALLBACK_CMD")
            elif command -v gemini >/dev/null 2>&1; then
                FALLBACK_CMD="$(build_gemini_cli_cmd)"
                FALLBACK_ARGS=(--fallback-codex --codex-command "$FALLBACK_CMD")
            else
                FALLBACK_ARGS=()
            fi
        else
            FALLBACK_ARGS=()
        fi
        ;;
esac

# =============================================================================
# Display configuration
# =============================================================================
echo "EASS Student Project Evaluator"
echo "=============================="
echo "Provider: ${AI_PROVIDER}"
case "$AI_PROVIDER" in
    gemini-cli) echo "Model:    ${GEMINI_CLI_MODEL}" ;;
    gemini-api) echo "Model:    ${GEMINI_API_MODEL}" ;;
    codex)      echo "Model:    ${CODEX_MODEL:-<from config>}" ;;
esac
echo "Timeout:  ${TIMEOUT}s"
[[ -n "$LIMIT" ]] && echo "Limit:    ${LIMIT#--limit }"
[[ -z "$NO_CLEAN" ]] && echo "Clean:    yes"
[[ -n "$PARALLEL" ]] && echo "Parallel: ${WORKERS} workers"
[[ -n "${FALLBACK_ARGS[*]}" ]] && echo "Fallback: ${AI_FALLBACK_CHAIN}"
echo ""

# =============================================================================
# Run the evaluation
# =============================================================================
if [[ -n "$PARALLEL" ]]; then
    # Parallel mode
    uv run python scripts/run_evaluation_parallel.py \
        $NO_CLEAN \
        $LIMIT \
        --keep-clones \
        --format-check \
        --format-command "/opt/homebrew/bin/ruff format --check ." \
        --score-file "$SCORE_FILE" \
        --score-key final_score \
        --timeout-seconds "$TIMEOUT" \
        --workers "$WORKERS" \
        "${PRIMARY_ARG[@]}" \
        "${FALLBACK_ARGS[@]}"
else
    # Sequential mode
    uv run python scripts/run_evaluation_parallel.py \
        $NO_CLEAN \
        $LIMIT \
        --keep-clones \
        --format-check \
        --format-command "/opt/homebrew/bin/ruff format --check ." \
        --score-file "$SCORE_FILE" \
        --score-key final_score \
        --timeout-seconds "$TIMEOUT" \
        --workers 1 \
        "${PRIMARY_ARG[@]}" \
        "${FALLBACK_ARGS[@]}"
fi

# =============================================================================
# Done
# =============================================================================
echo ""
echo "Complete. Results:"
echo "  results/REPORT.md       - Full analysis"
echo "  results/index.md        - Ranking overview"
echo "  results/ranked_list.csv - Spreadsheet data"
echo "  results/evaluation.json - Raw JSON data"
echo ""
