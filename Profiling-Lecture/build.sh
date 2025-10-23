#!/bin/bash
#!/bin/bash

# Exit script on any error
set -e

# Define directories
SOURCE_DIR=$(pwd)
BUILD_DIR="$SOURCE_DIR/build"

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir "$BUILD_DIR"
fi

# Navigate to the build directory
cd "$BUILD_DIR"

# Run cmake from the source directory
echo "Running cmake..."
cmake "$SOURCE_DIR"

# Compile with make
echo "Building project..."
make