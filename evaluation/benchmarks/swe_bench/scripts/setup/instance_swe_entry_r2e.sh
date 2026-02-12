#!/usr/bin/env bash

source ~/.bashrc
SWEUTIL_DIR=/swe_util

# FIXME: Cannot read SWE_INSTANCE_ID from the environment variable
# SWE_INSTANCE_ID=django__django-11099
if [ -z "$SWE_INSTANCE_ID" ]; then
    echo "Error: SWE_INSTANCE_ID is not set." >&2
    exit 1
fi

# Install jq if not present (it is not installed by default in the r2e environments)
if ! command -v jq >/dev/null 2>&1
then
    curl https://github.com/jqlang/jq/releases/download/jq-1.8.1/jq-linux-amd64 -Lo /bin/jq
    chmod +x /bin/jq
fi

# Read the swe-bench-test-lite.json file and extract the required item based on instance_id
item=$(jq --arg INSTANCE_ID "$SWE_INSTANCE_ID" '.[] | select(.instance_id == $INSTANCE_ID)' $SWEUTIL_DIR/eval_data/instances/swe-bench-instance.json)

if [[ -z "$item" ]]; then
  echo "No item found for the provided instance ID."
  exit 1
fi


WORKSPACE_NAME=$(echo "$item" | jq -r '(.repo | tostring) + "__" + (.version | tostring) | gsub("/"; "__")')

echo "WORKSPACE_NAME: $WORKSPACE_NAME"

mkdir -p /workspace

# Remove existing workspace if present
if [ -d /workspace/$WORKSPACE_NAME ]; then
    rm -rf /workspace/$WORKSPACE_NAME
fi

# Use cp with hard links (-al) for near-instant copy
# Falls back to regular copy if hard links fail
if ! cp -al /testbed /workspace/$WORKSPACE_NAME 2>/dev/null; then
    echo "Hard link copy failed, falling back to regular copy..."
    cp -r /testbed /workspace/$WORKSPACE_NAME
fi

# Activate instance-specific environment
source /testbed/.venv/bin/activate
