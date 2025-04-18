cmake_minimum_required(VERSION 3.16)
project (Test_CUDA_HIP_Examples)

set(CMAKE_PREFIX /opt/rocm)

find_package(hip REQUIRED)

#add_definitions(-D__HIP_PLATFORM_AMD__)

set (CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
set (CMAKE_CXX_STANDARD 14)
# set (CMAKE_CXX_FLAGS  "-save-temps")
#set (CMAKE_BUILD_TYPE Debug)
set(CMAKE_VERBOSE_MAKEFILE on)

include_directories(${HIP_INCLUDE_DIR} ../common)
link_directories(${HIP_LIBRARY_DIR})

message("cmake_current_src_dir = " ${CMAKE_SOURCE_DIR})
message("cmake_current_bin_dir = " ${CMAKE_BINARY_DIR})

file(GLOB util_SRC "*.cpp")
file(GLOB test_examples test_*.cpp)
file(GLOB util_HDR "*.hpp")
list(REMOVE_ITEM util_SRC ${test_examples})
set(lib_SRC 
    ${util_SRC} 
    ../common/timer.cpp
    ../common/layernorm_fuse.cpp)

set(test_examples 
    test_mm_mul.cpp 
    test_vec_add.cpp 
    test_reduction.cpp 
    test_sqrt.cpp 
    test_layernorm.cpp
    test_atomic_add.cpp
    test_cross_lane.cpp
    test_opt_atomic_add.cpp
    test_prefix_sum.cpp
    test_prefix_sum_one_wave.cpp
    test_exec_mask.cpp)

add_library(test_util SHARED ${lib_SRC})
#target_link_libraries(test_util ${HIP_LIBRARY})
target_link_libraries(test_util amdhip64)

foreach(filepath_name ${test_examples})
    get_filename_component(src_name ${filepath_name} NAME)
    get_filename_component(bin_name ${src_name} NAME_WE)
    message("source file: " ${src_name} " ---> bin: " ${bin_name})
    add_executable(${bin_name} ${src_name})
    target_link_libraries(${bin_name} test_util amdhip64 pthread)
endforeach(filepath_name)

