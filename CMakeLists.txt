cmake_minimum_required(VERSION 3.10)
project(yolov7)

set(CMAKE_CXX_STANDARD 11)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# cuda
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/usr/local/TensorRT-7.2.2.3/include/)
link_directories(/usr/local/TensorRT-7.2.2.3/lib/)

# project
include_directories(${PROJECT_SOURCE_DIR}/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_DIR)
cuda_add_library(yolov7 SHARED ${SRC_DIR} preprocess.cu )
target_link_libraries(yolov7 nvinfer cudart)

cuda_add_library(myplugins SHARED decodelayer.cu)
target_link_libraries(myplugins nvinfer cudart)

# ===== demo =====
cuda_add_executable(convert ${PROJECT_SOURCE_DIR}/convert.cpp)
target_link_libraries(convert yolov7 myplugins ${OpenCV_LIBS})

#add_definitions(-O2 -pthread)
