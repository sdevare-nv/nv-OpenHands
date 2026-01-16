#!/usr/bin/env bash
set -eo pipefail

# Remove this to allow running without version control
# source "evaluation/utils/version_control.sh"

checkout_eval_branch() {
    if [ -z "$COMMIT_HASH" ]; then
        echo "Commit hash not specified, use current git commit"
        return 0
    fi

    if git diff --quiet $COMMIT_HASH HEAD; then
        echo "The given hash is equivalent to the current HEAD"
        return 0
    fi

    echo "Start to checkout openhands version to $COMMIT_HASH, but keep current evaluation harness"
    if ! git diff-index --quiet HEAD --; then
        echo "There are uncommitted changes, please stash or commit them first"
        exit 1
    fi
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    echo "Current version is: $current_branch"
    echo "Check out OpenHands to version: $COMMIT_HASH"
    if ! git checkout $COMMIT_HASH; then
        echo "Failed to check out to $COMMIT_HASH"
        exit 1
    fi

    echo "Revert changes in evaluation folder"
    git checkout $current_branch -- evaluation

    # Trap the EXIT signal to checkout original branch
    trap checkout_original_branch EXIT

}


checkout_original_branch() {
    if [ -z "$current_branch" ]; then
        return 0
    fi
    echo "Checkout back to original branch $current_branch"
    git checkout $current_branch
}

get_openhands_version() {
    # IMPORTANT: Because Agent's prompt changes fairly often in the rapidly evolving codebase of OpenHands
    # We need to track the version of Agent in the evaluation to make sure results are comparable
    OPENHANDS_VERSION=v$(poetry run python -c "from openhands import get_version; print(get_version())")
}


MODEL_CONFIG=$1
COMMIT_HASH=$2
AGENT=$3
EVAL_LIMIT=$4
MAX_ITER=$5
NUM_WORKERS=$6
DATASET=$7
SPLIT=$8
EVAL_OUTPUT_DIR=${9}
SELECTED_ID=${10}
INSTANCE_DICT_PATH=${11}
CONFIG_FILE=${12}
N_RUNS=${13}
MODE=${14}

if [ -z "$NUM_WORKERS" ]; then
  NUM_WORKERS=1
  echo "Number of workers not specified, use default $NUM_WORKERS"
fi
checkout_eval_branch

if [ -z "$AGENT" ]; then
  echo "Agent not specified, use default CodeActAgent"
  AGENT="CodeActAgent"
fi

if [ -z "$MAX_ITER" ]; then
  echo "MAX_ITER not specified, use default 100"
  MAX_ITER=100
fi

if [ -z "$RUN_WITH_BROWSING" ]; then
  echo "RUN_WITH_BROWSING not specified, use default false"
  RUN_WITH_BROWSING=false
fi


if [ -z "$DATASET" ]; then
  echo "DATASET not specified, use default princeton-nlp/SWE-bench_Lite"
  DATASET="princeton-nlp/SWE-bench_Lite"
fi

if [ -z "$SPLIT" ]; then
  echo "SPLIT not specified, use default test"
  SPLIT="test"
fi

if [ -z "$MODE" ]; then
  MODE="swe"
  echo "MODE not specified, use default $MODE"
fi

if [ -n "$EVAL_CONDENSER" ]; then
  echo "Using Condenser Config: $EVAL_CONDENSER"
else
  echo "No Condenser Config provided via EVAL_CONDENSER, use default (NoOpCondenser)."
fi

if [ -z "$CONFIG_FILE" ]; then
  echo "CONFIG_FILE not specified, use default config.toml"
  CONFIG_FILE="config.toml"
fi

export RUN_WITH_BROWSING=$RUN_WITH_BROWSING
echo "RUN_WITH_BROWSING: $RUN_WITH_BROWSING"

get_openhands_version

echo "AGENT: $AGENT"
echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "DATASET: $DATASET"
echo "SPLIT: $SPLIT"
echo "MAX_ITER: $MAX_ITER"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "COMMIT_HASH: $COMMIT_HASH"
echo "MODE: $MODE"
echo "EVAL_CONDENSER: $EVAL_CONDENSER"
echo "EVAL_OUTPUT_DIR: $EVAL_OUTPUT_DIR"
echo "SELECTED_ID: $SELECTED_ID"
echo "INSTANCE_DICT_PATH: $INSTANCE_DICT_PATH"

# Default to NOT use Hint
if [ -z "$USE_HINT_TEXT" ]; then
  export USE_HINT_TEXT=false
fi
echo "USE_HINT_TEXT: $USE_HINT_TEXT"
EVAL_NOTE="$OPENHANDS_VERSION"
# if not using Hint, add -no-hint to the eval note
if [ "$USE_HINT_TEXT" = false ]; then
  EVAL_NOTE="$EVAL_NOTE-no-hint"
fi

if [ "$RUN_WITH_BROWSING" = true ]; then
  EVAL_NOTE="$EVAL_NOTE-with-browsing"
fi

if [ -n "$EXP_NAME" ]; then
  EVAL_NOTE="$EVAL_NOTE-$EXP_NAME"
fi
# if mode != swe, add mode to the eval note
if [ "$MODE" != "swe" ]; then
  EVAL_NOTE="${EVAL_NOTE}-${MODE}"
fi
# Add condenser config to eval note if provided
if [ -n "$EVAL_CONDENSER" ]; then
  EVAL_NOTE="${EVAL_NOTE}-${EVAL_CONDENSER}"
fi

function run_eval() {
  local eval_note="${1}"
  COMMAND="poetry run python evaluation/benchmarks/swe_bench/run_infer.py \
    --agent-cls $AGENT \
    --llm-config $MODEL_CONFIG \
    --max-iterations $MAX_ITER \
    --eval-num-workers $NUM_WORKERS \
    --eval-note $eval_note \
    --dataset $DATASET \
    --split $SPLIT \
    --mode $MODE"

  if [ -n "$EVAL_OUTPUT_DIR" ]; then
    COMMAND="$COMMAND --eval-output-dir $EVAL_OUTPUT_DIR"
  fi

  if [ -n "$EVAL_LIMIT" ]; then
    echo "EVAL_LIMIT: $EVAL_LIMIT"
    COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
  fi

  if [ -n "$SELECTED_ID" ]; then
    echo "SELECTED_ID: $SELECTED_ID"
    COMMAND="$COMMAND --selected-id \"$SELECTED_ID\""
  fi

  if [ -n "$INSTANCE_DICT_PATH" ]; then
    echo "INSTANCE_DICT: Using provided instance dictionary"
    COMMAND="$COMMAND --instance-dict-path $INSTANCE_DICT_PATH"
  fi

  if [ -n "$CONFIG_FILE" ]; then
    echo "CONFIG_FILE: $CONFIG_FILE"
    COMMAND="$COMMAND --config-file $CONFIG_FILE"
  fi

  # Run the command
  eval $COMMAND
}

unset SANDBOX_ENV_GITHUB_TOKEN # prevent the agent from using the github token to push
if [ -z "$N_RUNS" ]; then
  N_RUNS=1
  echo "N_RUNS not specified, use default $N_RUNS"
fi

# Skip runs if the run number is in the SKIP_RUNS list
# read from env variable SKIP_RUNS as a comma separated list of run numbers
SKIP_RUNS=(${SKIP_RUNS//,/ })
for i in $(seq 1 $N_RUNS); do
  if [[ " ${SKIP_RUNS[@]} " =~ " $i " ]]; then
    echo "Skipping run $i"
    continue
  fi
  current_eval_note="$EVAL_NOTE-run_$i"
  echo "EVAL_NOTE: $current_eval_note"
  run_eval $current_eval_note
done

checkout_original_branch
