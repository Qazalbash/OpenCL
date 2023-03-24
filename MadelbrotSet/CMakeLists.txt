cmake_minimum_required(VERSION 3.0)

project(MadelbortSet LANGUAGES C)

add_executable(MadelbortSet set.c)

target_compile_options(MadelbortSet PRIVATE -Wall -Wextra)

target_compile_features(MadelbortSet PRIVATE c_std_99)

find_package(OpenCL REQUIRED)
target_link_libraries(MadelbortSet PRIVATE OpenCL::OpenCL)