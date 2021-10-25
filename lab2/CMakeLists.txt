cmake_minimum_required(VERSION 3.10)
project(lab2)

set(CMAKE_CXX_STANDARD 17)

add_executable(lab2 main.cpp)
include_directories(/usr/include/opencv4)
#target_link_directories(lab2 /usr/lib/aarch64-linux-gnu)
target_link_libraries(lab2 opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc)