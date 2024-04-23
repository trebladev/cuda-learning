cmake_minimum_required(VERSION 3.0)

project(cuda_achieve LANGUAGES CXX)

# set(CUDA_TOOLKIT_ROOT_DIR "/home/xuan/cuda/cuda-11.7")
# set(CUDA_CUDART_LIBRARY "/home/xuan/cuda/cuda-11.7/lib64/libcudart.so")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
include(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake)

#set(CUDA_HEADERS "${PROJECT_SOURCE_DIR}/3rdparty/cuda-samples/Common")
include_directories(
        ${CUDA_INCLUDE_DIRS}
        #${CUDA_HEADERS}
        ./include
)

add_subdirectory(src/demo)
