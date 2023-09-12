cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
# set(CMAKE_C_COMPILER "/usr/bin/gcc")
project(Test_Inpainting VERSION 0.1.0)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_BUILD_TYPE DEBUG_POSTFIX)
# cuda
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.7" CACHE STRING "select cuda dir path")
set(cudalibpath ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
link_directories(${cudalibpath})
find_package(CUDA)

# TRT
set(TRT_ROOT_DIR "/data1/huangjiamian/envs/TensorRT-8.6.1.6/targets/x86_64-linux-gnu" CACHE STRING "select tensorrt dir path")
include_directories(${TRT_ROOT_DIR}/include)
set(TRT_LIB_PATH "${TRT_ROOT_DIR}/lib")
file(GLOB TRT_LIBS_TOP "${TRT_LIB_PATH}/*.so")
file(GLOB_RECURSE TRT_LIBS_SUB "${TRT_LIB_PATH}/stubs/*.so")
set(TRT_LIBS 
    ${TRT_LIBS_TOP} 
    ${TRT_LIBS_SUB}
)
message(STATUS "TensorRT library status:")
message(STATUS "    libraries: ${TRT_LIBS}" )

# set(CMAKE_CXX_FLAGS "-D TIXML_USE_STL -O3 -w -fPIC -g2 -ggdb")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -DTIXML_USE_STL -DDEBUG -DOCR -O0 -w -fPIC -g2 -ggdb")
# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -D TIXML_USE_STL -DOCR -O0 -w -fPIC -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -D TIXML_USE_STL -DOCR -O3 -w -fPIC")
include_directories(${PROJECT_SOURCE_DIR}/include)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
add_subdirectory(src)
link_directories(${PROJECT_SOURCE_DIR}/lib)

add_executable(main_class main.cpp)
target_link_libraries(main_class ${CUDA_LIBRARIES} ${TRT_LIBS} test)
set_property(TARGET main_class PROPERTY CXX_STANDARD 14)

