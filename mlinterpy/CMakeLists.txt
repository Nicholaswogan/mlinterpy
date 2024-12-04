
add_library(mlinterp_c_api mlinterp_c_api.cpp)
target_include_directories(mlinterp_c_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (NOT SKBUILD)
  if (NOT DEFINED SKBUILD_CMAKE_MODULE_DIR)
    # Here, we try to find scikit-build cmake modules
    find_package(Python COMPONENTS Development)
    set(SKBUILD_CMAKE_MODULE_DIR "${Python_LIBRARY_DIRS}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/skbuild/resources/cmake")
  endif()
  if (EXISTS ${SKBUILD_CMAKE_MODULE_DIR})
    message(STATUS "Scikit-build CMake modules: ${SKBUILD_CMAKE_MODULE_DIR}")
  else()
    message(FATAL_ERROR "Failed to find scikit-build CMake modules in directory: ${SKBUILD_CMAKE_MODULE_DIR}")
  endif()
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${SKBUILD_CMAKE_MODULE_DIR})
endif()

find_package(PythonExtensions REQUIRED)
find_package(NumPy REQUIRED)
find_package(Cython REQUIRED)

add_cython_target(_mlinterpy _mlinterpy.pyx)
add_library(_mlinterpy MODULE ${_mlinterpy})
python_extension_module(_mlinterpy)

target_link_libraries(_mlinterpy mlinterp_c_api)
target_include_directories(_mlinterpy PUBLIC ${NumPy_INCLUDE_DIRS})

if (SKBUILD)
  install(TARGETS _mlinterpy DESTINATION mlinterpy)
else()
  install(TARGETS _mlinterpy DESTINATION ${CMAKE_SOURCE_DIR}/mlinterpy)
endif()

