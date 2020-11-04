cmake_minimum_required (VERSION 3.12)
# cmake_policy(SET CMP0048 NEW)
project(mass_data_collector)

# cmake flags ---------------------------------------------------------------
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS YES)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# required for stuff and things ---------------------------------------------
find_package(Threads REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package (Eigen3 3 REQUIRED)

# CARLA related stuff -------------------------------------------------------
if (NOT DEFINED ENV{CARLA_CXX_API})
	message(FATAL_ERROR "environment variable CARLA_CXX_API is not defined. \
						 export it before compiling")
endif ()
FILE(GLOB CARLA_LIBRARIES $ENV{CARLA_CXX_API}/lib/*.a)
# catkin packages -----------------------------------------------------------
find_package(catkin REQUIRED COMPONENTS
	roscpp
	roslib
	# cv_bridge
	# image_transport
	# sensor_msgs
	freicar_map
)
# HDF5 lib ------------------------------------------------------------------
# CMAKE_PREFIX_PATH should only be appended after catkin find_package (idk)
list(APPEND CMAKE_PREFIX_PATH $ENV{LIBHDF5_ROOT})
find_package (HDF5) # NAMES hdf5 COMPONENTS C ${SEARCH_TYPE}

# more catkin stuff ---------------------------------------------------------
catkin_package(CATKIN_DEPENDS 
	roscpp
	roslib
	# cv_bridge
	# image_transport
	# sensor_msgs
	freicar_map
)

# include dirs --------------------------------------------------------------
include_directories(
	"${HDF5_INCLUDE_DIR}"
	${PROJECT_SOURCE_DIR}/include
	${catkin_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	$ENV{CARLA_CXX_API}/include
	$ENV{CARLA_CXX_API}/include/system
	# ${Open3D_INCLUDE_DIRS} not necessary
)

# HDF5 api for the dataset --------------------------------------------------
add_library(hdf5_dataset STATIC 
	${PROJECT_SOURCE_DIR}/src/hdf5_api/hdf5_dataset.cpp
)
target_link_libraries(hdf5_dataset hdf5-static hdf5_cpp-static)
set_target_properties(hdf5_dataset PROPERTIES CXX_STANDARD 17)

# HDF5 api test -------------------------------------------------------------
# add_executable(hdf5_test ${PROJECT_SOURCE_DIR}/src/hdf5_api/hdf5_test.cpp)
# target_link_libraries(hdf5_test hdf5_dataset)
# add_executable(example ${PROJECT_SOURCE_DIR}/src/test.cpp)
# target_link_libraries(example hdf5-static hdf5_cpp-static)

# data collection executable ------------------------------------------------
add_executable(${PROJECT_NAME}
	src/main.cpp
	src/mass_agent/mass_agent.cpp
)

target_link_libraries(${PROJECT_NAME}
	${catkin_LIBRARIES}
	${CARLA_LIBRARIES}
	${OpenCV_LIBS}
	$ENV{CARLA_CXX_API}/lib/libDetourCrowd.a
	$ENV{CARLA_CXX_API}/lib/libDetour.a
	$ENV{CARLA_CXX_API}/lib/librpc.a
	hdf5_dataset
	yaml-cpp
	Open3D
	Open3D_3rdparty_jsoncpp
	pthread
)