cmake_minimum_required (VERSION 3.12)
project (HDF5Test CXX)

# set (LIB_TYPE STATIC) # or SHARED
# string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)
list(APPEND CMAKE_PREFIX_PATH $ENV{LIBHDF5_ROOT})
find_package (HDF5) # NAMES hdf5 COMPONENTS C ${SEARCH_TYPE}

set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIR}")

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(test ${PROJECT_SOURCE_DIR}/src/main.cpp
                    ${PROJECT_SOURCE_DIR}/src/hdf5_dataset.cpp)
target_link_libraries(test hdf5-static hdf5_cpp-static)


# add_executable(example ${PROJECT_SOURCE_DIR}/src/test.cpp)
# target_link_libraries(example hdf5-static hdf5_cpp-static)