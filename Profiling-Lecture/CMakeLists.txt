cmake_minimum_required(VERSION 3.18)
project(KernelComparison CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Find CUDA toolkit (for CURAND)
find_package(CUDAToolkit REQUIRED)

# # Add cutlass
# add_subdirectory(cutlass EXCLUDE_FROM_ALL)
# set_property(DIRECTORY cutlass PROPERTY EXCLUDE_FROM_ALL YES)


# Function to define a CUDA target with correct flags
function(build_cuda_target target_name target_srcs)
    add_executable(${target_name} ${target_srcs})

    # Link against CURAND
    target_link_libraries(${target_name} PRIVATE CUDA::curand)

    # Link against CUBLAS
    target_link_libraries(${target_name} PRIVATE CUDA::cublas)

    # Link against nvToolsExt
    target_link_libraries(${target_name} PRIVATE CUDA::nvToolsExt)

    # Enable separable compilation
    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Set SM - if you have a GPU system, you can set it to native to ensure best performance
    set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES 86)
    # set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES native)

    # Add compiler flags for CUDA and C++
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info --use_fast_math --relocatable-device-code=true --expt-extended-lambda --expt-relaxed-constexpr -O3>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    )
endfunction()

# Function to define a CUTLASS target with correct flags
function(build_cutlass_target target_name target_srcs)
    add_executable(${target_name} ${target_srcs})

    # Link against CURAND
    target_link_libraries(${target_name} PRIVATE CUDA::curand)

    # Link against CUBLAS
    target_link_libraries(${target_name} PRIVATE CUDA::cublas)

    # Enable separable compilation
    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Set SM
    set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES native)

    cutlass_apply_standard_compile_options(${target_name})
endfunction()

# Build targets
build_cuda_target(matmul_pageable matmul_pageable.cu)
