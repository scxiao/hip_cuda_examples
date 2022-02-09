cmake_minimum_required(VERSION 3.18)
project (Test_Matmul)

add_definitions(-D__HIP_PLATFORM_AMD__)

set (CMAKE_CXX_COMPILER /opt/rocm/hip/bin/hipcc)
set (CMAKE_CXX_STANDARD 11)

#add_custom_target(
#    copylibs
#	COMMAND cp ${MIGRAPHX_BUILD}/src/onnx/libmigraphx_onnx.so          ${MIGRAPHX_LIB_PATH}/.
#	COMMAND cp ${MIGRAPHX_BUILD}/src/libmigraphx.so                    ${MIGRAPHX_LIB_PATH}/.
#	COMMAND cp ${MIGRAPHX_BUILD}/src/targets/cpu/libmigraphx_cpu.so    ${MIGRAPHX_LIB_PATH}/.
#	COMMAND cp ${MIGRAPHX_BUILD}/src/targets/gpu/libmigraphx_gpu.so    ${MIGRAPHX_LIB_PATH}/.
#	COMMAND cp ${MIGRAPHX_BUILD}/src/targets/gpu/libmigraphx_device.so ${MIGRAPHX_LIB_PATH}/.)

include_directories(/opt/rocm/hip/include ../common)
link_directories(/opt/rocm/hip/lib)

message("cmake_current_src_dir = " ${CMAKE_SOURCE_DIR})
message("cmake_current_bin_dir = " ${CMAKE_BINARY_DIR})

file(GLOB util_SRC "*.cpp")
#file(GLOB cu_SRC "*.cu")
file(GLOB test_examples test_*.cpp)
file(GLOB util_HDR "*.hpp")
list(REMOVE_ITEM util_SRC ${test_examples})
set(lib_SRC ${util_SRC} ../common/timer.cpp)

message("lib_src = " ${lib_SRC})

#set(test_examples test_mm_mul.cpp test_vec_add.cpp)
set(test_examples test_mm_mul.cpp test_vec_add.cpp)
add_library(test_util SHARED ${lib_SRC})
target_link_libraries(test_util amdhip64)

foreach(filepath_name ${test_examples})
    get_filename_component(src_name ${filepath_name} NAME)
    get_filename_component(bin_name ${src_name} NAME_WE)
    message("source file: " ${src_name} " ---> bin: " ${bin_name})
    add_executable(${bin_name} ${src_name})
    target_link_libraries(${bin_name} test_util amdhip64 pthread)
endforeach(filepath_name)

