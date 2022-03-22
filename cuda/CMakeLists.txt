cmake_minimum_required(VERSION 3.18)
project (Test_Reduction)

add_definitions(-D__NVCC__)
enable_language(CUDA)
#set (CMAKE_CXX_STANDARD 11)

include_directories(/usr/local/cuda-11.3/include ../common)
link_directories(/usr/local/cuda-11.3/lib64)

message("cmake_current_src_dir = " ${CMAKE_SOURCE_DIR})
message("cmake_current_bin_dir = " ${CMAKE_BINARY_DIR})

file(GLOB util_SRC "*.cpp")
file(GLOB cu_SRC "*.cu")
file(GLOB test_examples test_*.cpp)
file(GLOB util_HDR "*.hpp")
list(REMOVE_ITEM util_SRC ${test_examples})
set(lib_SRC ${util_SRC} ${cu_SRC} ../common/timer.cpp)

message("lib_src = " ${lib_SRC})

set(test_examples test_mm_mul.cpp test_vec_add.cpp test_reduction.cpp)
add_library(test_util SHARED ${lib_SRC})
target_link_libraries(test_util cudart)
#set_property(TARGET test_util PROPERTY CUDA_ARCHITECTURES 61 70 75)
set_property(TARGET test_util PROPERTY CUDA_ARCHITECTURES 70)

foreach(filepath_name ${test_examples})
    get_filename_component(src_name ${filepath_name} NAME)
    get_filename_component(bin_name ${src_name} NAME_WE)
    message("source file: " ${src_name} " ---> bin: " ${bin_name})
    add_executable(${bin_name} ${src_name})
    set_property(TARGET ${bin_name} PROPERTY CUDA_STANDARD 11)
    target_link_libraries(${bin_name} test_util pthread cudart)
endforeach(filepath_name)

