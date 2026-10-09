cmake_minimum_required(VERSION 3.18)
project(fxcore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(fxcore STATIC
  cpp/src/indicators.cpp
  cpp/src/labeling.cpp
  cpp/src/backtest.cpp)
target_include_directories(fxcore PUBLIC cpp/include)
if(MSVC)
  target_compile_options(fxcore PRIVATE /W4)
else()
  target_compile_options(fxcore PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_executable(test_fxcore cpp/tests/test_fxcore.cpp)
target_link_libraries(test_fxcore PRIVATE fxcore)
add_test(NAME fxcore_unit_tests COMMAND test_fxcore)

# Optional: build the Python extension with CMake instead of `pip install -e .`
option(FXCORE_BUILD_PYTHON "Build the pybind11 Python module" OFF)
if(FXCORE_BUILD_PYTHON)
  # Pass -DPython_EXECUTABLE=<path to .venv python> to build for the project's environment.
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  # get_cmake_dir() returns the path unquoted (`-m pybind11 --cmakedir` quotes paths with spaces).
  execute_process(COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
                  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(pybind11 CONFIG REQUIRED)
  # NO_EXTRAS skips pybind11's post-build strip, which breaks on folder names containing "(".
  pybind11_add_module(_fxcore NO_EXTRAS cpp/bindings/module.cpp)
  target_link_libraries(_fxcore PRIVATE fxcore)
  # The generator expression stops multi-config generators (Visual Studio, Xcode)
  # from adding a Release/ or Debug/ sub-folder, so Python finds the module.
  set_target_properties(_fxcore PROPERTIES LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_SOURCE_DIR}/fxquant>")
endif()
