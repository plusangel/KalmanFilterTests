cmake_minimum_required(VERSION 3.14)
project(kalman_test)

set(CMAKE_CXX_STANDARD 17)

find_package(PythonLibs 2.7)

# add_executable(kalman_tests7 part7.cpp)
# target_include_directories(kalman_tests7 PRIVATE ${PYTHON_INCLUDE_DIRS})
# target_link_libraries(kalman_tests7 ${PYTHON_LIBRARIES})

add_executable(kalman_tests9 src/part9.cpp)
target_include_directories(kalman_tests9 PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(kalman_tests9 ${PYTHON_LIBRARIES})