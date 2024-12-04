cmake_minimum_required(VERSION "3.14")
project(MLINTERPY LANGUAGES C CXX VERSION "0.1.0")

option(SKBUILD "Should be ON of being build by skbuild, 
and OFF of being build by regular cmake" OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_CXX_STANDARD 11)

add_subdirectory(mlinterpy)