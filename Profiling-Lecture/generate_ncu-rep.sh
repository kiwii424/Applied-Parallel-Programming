#!/bin/bash

# Set default root directory
ROOT_DIR=${1:-build}
NCU_DIR=${2:-profile_ncu-rep}

# Example: listing executables in the root directory
echo "Using root directory: $ROOT_DIR"

# If needed, check if the directory exists
if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Directory '$ROOT_DIR' does not exist."
    exit 1
fi

if [ ! -d "$NCU_DIR" ]; then
    mkdir $NCU_DIR
fi

# list all executable files in the root directory
find "$ROOT_DIR" -maxdepth 1 -type f -executable -exec echo "Found executable: {}" \;

# Exit script on any error
set -e

# Run ncu
ncu -o $NCU_DIR/matmul_pageable --set full -f  --profile-from-start off $ROOT_DIR/matmul_pageable --profile
