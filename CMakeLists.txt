cmake_minimum_required(VERSION 3.14)
project(kalman_test)

set(CMAKE_CXX_STANDARD 17)

find_package(PythonLibs 2.7)

add_executable(kalman_tests main.cpp)
target_include_directories(kalman_tests PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(kalman_tests ${PYTHON_LIBRARIES})
