# Require at least CMake 3.25, because in 3.24 and 3.25 there were a couple of
# FetchContent-related changes
cmake_minimum_required(VERSION 3.25)
project(dtcc-3dbag-roofer)

# Global CMake variables are set here We use C++20, with the assumption that we
# only implement features that are supported by GCC, Clang, MSVC, Apple Clang
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Don't use extensions, because they might affect compiler compatibility
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/roofer/cmake/")

find_package(fmt REQUIRED)
find_package(
    CGAL 6.0 QUIET
    COMPONENTS Core
    REQUIRED)

if(MSVC)
    # windows.h breaks std::min/std::max, fix by define
    add_definitions(-DNOMINMAX)

    # enable permissive compiling and/or statements
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive-")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:precise")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
endif()

# set(PYBIND11_FINDPYTHON ON)
# set(Python_FIND_VIRTUALENV ONLY)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message("Python3_EXECUTABLE " ${Python3_EXECUTABLE})

# # Find pybind11
execute_process(COMMAND python -c "import site; print([s for s in site.getsitepackages() if 'site-packages' in s][0])" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
message("PYTHON_SITE_PACKAGES " ${PYTHON_SITE_PACKAGES})
set(pybind11_DIR ${PYTHON_SITE_PACKAGES}/pybind11/share/cmake/pybind11)

message("pybind11_DIR " ${pybind11_DIR})
find_package(pybind11 REQUIRED)
message("pybind11_INCLUDE_DIRS " ${pybind11_INCLUDE_DIRS})

set(ROOFER_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/roofer/include")
include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})

add_subdirectory(roofer/src)
add_subdirectory(roofer/rooferpy)

install(TARGETS rooferpy DESTINATION dtcc_3dbag_roofer)
