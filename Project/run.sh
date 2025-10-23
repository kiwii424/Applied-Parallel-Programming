#!/bin/bash

clean() {
    echo "Cleaning up..."
    ./cleanfile.sh
    rm -rf ./m1_cpu ./m1_gpu ./m2_unroll ./m2 ./m3 *.out *.err outfile
}

build() {
    echo "Building the project..."
    ./cleanfile.sh
    module load cuda/12.4
    cmake ./project/ && make -j8
    ./cleanfile.sh
}


case "$1" in
    clean) clean ;;
    build) build ;;
    *) echo "Usage: $0 {clean|build}" ;;
esac
