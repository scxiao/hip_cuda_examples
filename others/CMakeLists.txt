cmake_minimum_required(VERSION 3.16)
project (Test_CUDA_HIP_Examples)

set (CMAKE_CXX_STANDARD 14)
#set (CMAKE_BUILD_TYPE Debug)

message("cmake_current_src_dir = " ${CMAKE_SOURCE_DIR})
message("cmake_current_bin_dir = " ${CMAKE_BINARY_DIR})

#file(GLOB util_SRC "*.cpp")
#file(GLOB test_examples test_*.cpp)
#file(GLOB util_HDR "*.hpp")
#list(REMOVE_ITEM util_SRC ${test_examples})
#set(lib_SRC 
#    ${util_SRC} 
#    ../common/timer.cpp
#    ../common/layernorm_fuse.cpp)

set(test_examples 
    test_swizzle.cpp)

#add_library(test_util SHARED ${lib_SRC})
#target_link_libraries(test_util ${HIP_LIBRARY})
#target_link_libraries(test_util amdhip64)

foreach(filepath_name ${test_examples})
    get_filename_component(src_name ${filepath_name} NAME)
    get_filename_component(bin_name ${src_name} NAME_WE)
    message("source file: " ${src_name} " ---> bin: " ${bin_name})
    add_executable(${bin_name} ${src_name})
endforeach(filepath_name)

