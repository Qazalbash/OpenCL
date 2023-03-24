cmake_minimum_required(VERSION 3.0)

project(HelloWorld LANGUAGES C)

add_executable(HelloWorld hello.c)

target_compile_options(HelloWorld PRIVATE -Wall -Wextra)

target_compile_features(HelloWorld PRIVATE c_std_99)

find_package(OpenCL REQUIRED)
target_link_libraries(HelloWorld PRIVATE OpenCL::OpenCL)