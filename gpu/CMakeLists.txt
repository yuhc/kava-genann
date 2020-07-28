project(genann-gpu C CXX CUDA)
cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
 set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/mnist
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math --ptxas-options=-v")

# Build libgenann.so
# set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
add_library(genann SHARED
  ${PROJECT_SOURCE_DIR}/genann_backprop_edits.cu
)
set_property(TARGET genann PROPERTY CUDA_ARCHITECTURES 61)

function(add_cuda_target name sources)
  add_executable(${name} ${sources})
  target_link_libraries(${name}
    genann
  )
endfunction()

add_cuda_target(example1
  ${PROJECT_SOURCE_DIR}/example1.c
)
add_cuda_target(example2
  ${PROJECT_SOURCE_DIR}/example2.c
)
add_cuda_target(example3
  ${PROJECT_SOURCE_DIR}/example3.c
)
add_cuda_target(example4
  ${PROJECT_SOURCE_DIR}/example4.c
)
set(mnist_src
  ${PROJECT_SOURCE_DIR}/mnist/runmnist.c
  ${PROJECT_SOURCE_DIR}/mnist/mnist.c
)
add_cuda_target(runmnist
  "${mnist_src}"
)
