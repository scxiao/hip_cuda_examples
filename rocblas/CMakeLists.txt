cmake_minimum_required(VERSION 3.18)
project (Test_rocblas_matmul)
find_package(ROCM REQUIRED)

add_definitions(-D__HIP_PLATFORM_AMD__)
#include(ROCMSetupVersion)
#rocm_setup_version(VERSION 2.1)

#list(APPEND PACKAGE_DEPENDS ACKAGE rocblas)
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip /opt/rocm/hcc /opt/rocm-5.0.0/rocblas)
#find_package(rocblas, REQUIRED PATHS /opt/rocm/rocblas)

#set (CMAKE_CXX_COMPILER /opt/rocm/hip/bin/hipcc)
set (CMAKE_CXX_STANDARD 11)

include_directories(/opt/rocm/hip/include ../common /opt/rocm/rocblas/include)
link_directories(/opt/rocm/hip/lib /opt/rocm/rocblas/lib)

message("cmake_current_src_dir = " ${CMAKE_SOURCE_DIR})
message("cmake_current_bin_dir = " ${CMAKE_BINARY_DIR})

file(GLOB util_SRC "*.cpp")
file(GLOB test_examples test_*.cpp)
file(GLOB util_HDR "*.hpp")
list(REMOVE_ITEM util_SRC ${test_examples})
set(lib_SRC ${util_SRC} ../common/timer.cpp)

message("lib_src = " ${lib_SRC})

#set(test_examples test_mm_mul.cpp test_vec_add.cpp)
set(test_examples test_rocblas_matmul.cpp)
add_library(test_rocblas SHARED ${lib_SRC})
target_link_libraries(test_rocblas amdhip64 rocblas)

foreach(filepath_name ${test_examples})
    get_filename_component(src_name ${filepath_name} NAME)
    get_filename_component(bin_name ${src_name} NAME_WE)
    message("source file: " ${src_name} " ---> bin: " ${bin_name})
    add_executable(${bin_name} ${src_name})
    target_link_libraries(${bin_name} test_rocblas amdhip64 pthread)
endforeach(filepath_name)

