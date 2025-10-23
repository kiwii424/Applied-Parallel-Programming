#!/bin/bash

# Set default root directory
ROOT_DIR=${1:-build}
NSYS_DIR=${2:-profile_nsys}

# Example: listing executables in the root directory
echo "Using root directory: $ROOT_DIR"

# If needed, check if the directory exists
if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Directory '$ROOT_DIR' does not exist."
    exit 1
fi

if [ ! -d "$NSYS_DIR" ]; then
    mkdir $NSYS_DIR
fi

# list all executable files in the root directory
find "$ROOT_DIR" -maxdepth 1 -type f -executable -exec echo "Found executable: {}" \;

# Exit script on any error
set -e

# Run nsys
nsys profile -o $NSYS_DIR/matmul_pageable --force-overwrite true $ROOT_DIR/matmul_pageable
# nsys profile -o $NSYS_DIR/matmul_pinned --force-overwrite true $ROOT_DIR/matmul_pinned
