cmake_minimum_required (VERSION 3.12)
project (HDF5Test CXX)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

# set (LIB_TYPE STATIC) # or SHARED
# string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)
list(APPEND CMAKE_PREFIX_PATH $ENV{LIBHDF5_ROOT})
find_package (HDF5) # NAMES hdf5 COMPONENTS C ${SEARCH_TYPE}

# set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIR}")
include_directories("${HDF5_INCLUDE_DIR}"
                     ${PROJECT_SOURCE_DIR}/include)

# HDF5 api for the dataset
add_library(hdf5_dataset STATIC ${PROJECT_SOURCE_DIR}/src/hdf5_api/hdf5_dataset.cpp)
target_link_libraries(hdf5_dataset hdf5-static hdf5_cpp-static)
set_target_properties(hdf5_dataset PROPERTIES CXX_STANDARD 17)

# HDF5 library test
add_executable(hdf5_test ${PROJECT_SOURCE_DIR}/src/hdf5_api/hdf5_test.cpp)
target_link_libraries(hdf5_test hdf5_dataset)
# add_executable(example ${PROJECT_SOURCE_DIR}/src/test.cpp)
# target_link_libraries(example hdf5-static hdf5_cpp-static)

# ROS stuff
