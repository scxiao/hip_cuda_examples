cmake_minimum_required(VERSION 3.16)
project (Test_CUDA_HIP_Examples)

set(CMAKE_PREFIX /opt/rocm)

find_package(hip REQUIRED)

add_definitions(-D__HIP_PLATFORM_AMD__)

set (CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
set (CMAKE_CXX_STANDARD 14)
#set (CMAKE_CXX_FLAGS  "-save-temps")

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

message("lib_src = " ${lib_SRC})

#set(test_examples 
#    test_mm_mul.cpp 
#    test_vec_add.cpp 
#    test_reduction.cpp 
#    test_sqrt.cpp 
#    test_layernorm.cpp)
#
set(test_examples 
    test_vec_add.cpp)


add_library(test_util SHARED ${lib_SRC})
target_link_libraries(test_util ${HIP_LIBRARY})

foreach(filepath_name ${test_examples})
    get_filename_component(src_name ${filepath_name} NAME)
    get_filename_component(bin_name ${src_name} NAME_WE)
    message("source file: " ${src_name} " ---> bin: " ${bin_name})
    add_executable(${bin_name} ${src_name})
    target_link_libraries(${bin_name} test_util amdhip64 pthread)
endforeach(filepath_name)

