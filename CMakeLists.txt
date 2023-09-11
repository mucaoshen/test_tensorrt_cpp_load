cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
# set(CMAKE_C_COMPILER "/usr/bin/gcc")
project(Test_Inpainting VERSION 0.1.0)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_BUILD_TYPE DEBUG_POSTFIX)
# cuda
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.7)
set(cudalibpath ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
link_directories(${cudalibpath})
find_package(CUDA)

# # Torch
# set(Torch_DIR "/usr/local/libtorch/share/cmake/Torch")
# find_package(Torch REQUIRED)
# message(STATUS "Torch library status:")
# # message(STATUS "    version: ${OpenCV_VERSION}")
# message(STATUS "    libraries: ${TORCH_LIBRARIES} ${TORCH_LIBRARY}" )
# message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}")
# include_directories(${TORCH_INCLUDE_DIRS})

# # TorchVison
# find_package(TorchVision REQUIRED)
# message(STATUS "TorchVision library status:")
# # message(STATUS "    version: ${OpenCV_VERSION}")
# message(STATUS "    libraries: ${TorchVision_LIBRARY}" )
# message(STATUS "    include path: ${TorchVision_INCLUDE_DIR}")
# include_directories(${TorchVision_INCLUDE_DIR})

# TRT
include_directories(/data1/huangjiamian/envs/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/include)
set(TRT_LIB_PATH "/data1/huangjiamian/envs/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib")
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

add_executable(Test_Inpainting main.cpp)
target_link_libraries(Test_Inpainting ${CUDA_LIBRARIES} ${TRT_LIBS} test)
set_property(TARGET Test_Inpainting PROPERTY CXX_STANDARD 14)

