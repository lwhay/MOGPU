cmake_minimum_required(VERSION 3.12)
project(MOGPU)
enable_language(C CXX CUDA)

find_package(Threads)

add_executable(sandbox ./test/inputVerify.cpp)