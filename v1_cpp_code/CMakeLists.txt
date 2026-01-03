cmake_minimum_required(VERSION 3.10)
project(PPE_Detection_V2)

set(CMAKE_CXX_STANDARD 17)

set(ONNX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.23.2")

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn)

include_directories(
    . 
    ${OpenCV_INCLUDE_DIRS} 
    ${ONNX_DIR}/include
)

link_directories(${ONNX_DIR}/lib)

add_executable(ppe_detector inference.cpp) 

target_link_libraries(ppe_detector 
    ${OpenCV_LIBRARIES} 
    onnxruntime
)