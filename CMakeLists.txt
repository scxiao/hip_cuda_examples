cmake_minimum_required(VERSION 3.18)
project (Test_Matmul)

#add_definitions(-D__HIP_PLATFORM_HCC__)

#set (DEPS_FOLDER deps_miopen2.2)
#set (MIGRAPHX_BUILD ${MIGRAPHX_FOLDER}/build_rocm3.0)
#set (MIGRAPHX_BUILD ${MIGRAPHX_FOLDER}/build_debug)
#set (MIGRAPHX_BUILD ${MIGRAPHX_FOLDER}/build)
#set (MIGRAPHX_DEPS /home/scxiao/Workplace/projects/MIGraph/${DEPS_FOLDER}/lib
#                   /opt/rocm/lib)
#set (CMAKE_CXX_COMPILER /usr/local/cuda-11.3/bin/nvcc)
enable_language(CUDA)
#find_package(nvcc)
#set (CMAKE_CXX_STANDARD 11)

#add_custom_target(
#    copylibs
#	COMMAND cp ${MIGRAPHX_BUILD}/src/onnx/libmigraphx_onnx.so          ${MIGRAPHX_LIB_PATH}/.
#	COMMAND cp ${MIGRAPHX_BUILD}/src/libmigraphx.so                    ${MIGRAPHX_LIB_PATH}/.
#	COMMAND cp ${MIGRAPHX_BUILD}/src/targets/cpu/libmigraphx_cpu.so    ${MIGRAPHX_LIB_PATH}/.
#	COMMAND cp ${MIGRAPHX_BUILD}/src/targets/gpu/libmigraphx_gpu.so    ${MIGRAPHX_LIB_PATH}/.
#	COMMAND cp ${MIGRAPHX_BUILD}/src/targets/gpu/libmigraphx_device.so ${MIGRAPHX_LIB_PATH}/.)
#
#
#include_directories(${MIGRAPHX_FOLDER}/src/include
#                    ${MIGRAPHX_FOLDER}/src/api/include
#                    ${MIGRAPHX_FOLDER}/src/targets/gpu/include
#                    ${MIGRAPHX_FOLDER}/src/targets/cpu/include
#                    ${MIGRAPHX_FOLDER}/src/targets/ref/include
#                    ${MIGRAPHX_FOLDER}/test/include
#                    /opt/rocm/include
#                    /home/scxiao/Workplace/projects/MIGraph/${DEPS_FOLDER}/include)

include_directories(/usr/local/cuda-11.3/include)
link_directories(/usr/local/cuda-11.3/lib64)

message("cmake_current_src_dir = " ${CMAKE_SOURCE_DIR})
message("cmake_current_bin_dir = " ${CMAKE_BINARY_DIR})

file(GLOB util_SRC "*.cpp")
file(GLOB cu_SRC "*.cu")
file(GLOB test_examples test_*.cpp)
file(GLOB util_HDR "*.hpp")
list(REMOVE_ITEM util_SRC ${test_examples})
set(lib_SRC ${util_SRC} ${cu_SRC})

message("lib_src = " ${lib_SRC})

set(test_examples test_mm_mul.cpp test_vec_add.cpp)
add_library(test_util SHARED ${lib_SRC})
target_link_libraries(test_util cudart)

foreach(filepath_name ${test_examples})
    get_filename_component(src_name ${filepath_name} NAME)
    get_filename_component(bin_name ${src_name} NAME_WE)
    message("source file: " ${src_name} " ---> bin: " ${bin_name})
    add_executable(${bin_name} ${src_name})
    set_property(TARGET ${bin_name} PROPERTY CUDA_STANDARD 11)
    target_link_libraries(${bin_name} test_util pthread cudart)
endforeach(filepath_name)

