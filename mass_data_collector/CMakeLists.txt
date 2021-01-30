cmake_minimum_required (VERSION 3.12)
# cmake_policy(SET CMP0048 NEW)
project(mass_data_collector)

# cmake flags ---------------------------------------------------------------
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS YES)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_FIND_DEBUG_MODE ON)

# required for stuff and things ---------------------------------------------
find_package(Threads REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package (Eigen3 3 REQUIRED)
find_package(PCL 1.3 REQUIRED COMPONENTS common io)
find_package(nanoflann)
find_package(OpenMP REQUIRED)
# find_package(Open3D HINTS /export/repos/Open3D/build/install/lib/cmake/Open3D)
# CARLA related stuff -------------------------------------------------------
if (NOT DEFINED ENV{CARLA_910_CXX_API})
	message(FATAL_ERROR "environment variable CARLA_910_CXX_API is not defined. \
						 export it before compiling")
endif ()
# message(WARNING "CARLA API: $ENV{CARLA_910_CXX_API}")
FILE(GLOB CARLA_LIBRARIES $ENV{CARLA_910_CXX_API}/lib/*.a)
# catkin packages -----------------------------------------------------------
find_package(catkin REQUIRED COMPONENTS
	roscpp
	roslib
	# cv_bridge
	# image_transport
	# sensor_msgs
	# freicar_map
)
# HDF5 ----------------------------------------------------------------------
# CMAKE_PREFIX_PATH should only be appended after catkin find_package (idk)
list (APPEND CMAKE_PREFIX_PATH $ENV{LIBHDF5_ROOT})
find_package (HDF5) # NAMES hdf5 COMPONENTS C ${SEARCH_TYPE}
# message(WARNING "${CMAKE_MODULE_PATH}")

# more catkin stuff ---------------------------------------------------------
catkin_package(CATKIN_DEPENDS 
	roscpp
	roslib
	# cv_bridge
	# image_transport
	# sensor_msgs
	# freicar_map
)

# include dirs --------------------------------------------------------------
include_directories(
	"/usr/lib/gcc/x86_64-linux-gnu/9/include;"
	${PROJECT_SOURCE_DIR}/include
	${HDF5_INCLUDE_DIR}
	${catkin_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	${PCL_INCLUDE_DIRS}
	${OPEN3D_INCLUDE_DIRS}
	$ENV{CARLA_910_CXX_API}/include
	$ENV{CARLA_910_CXX_API}/include/system
)
# HDF5 lib for the dataset --------------------------------------------------
add_library(hdf5_dataset STATIC 
	${PROJECT_SOURCE_DIR}/src/hdf5_api/hdf5_dataset.cpp
)
target_link_libraries(hdf5_dataset hdf5-static hdf5_cpp-static)
set_target_properties(hdf5_dataset PROPERTIES CXX_STANDARD 17)

# HDF5 api test -------------------------------------------------------------
add_executable(hdf5_test ${PROJECT_SOURCE_DIR}/src/hdf5_api/hdf5_test.cpp)
target_link_libraries(hdf5_test hdf5_dataset)
# add_executable(example ${PROJECT_SOURCE_DIR}/src/test.cpp)
# target_link_libraries(example hdf5-static hdf5_cpp-static)

# data collection executable ------------------------------------------------
add_executable(${PROJECT_NAME}
	src/main.cpp
	src/geometry/camera_geometry.cpp
	src/geometry/semantic_cloud.cpp
	src/mass_agent/mass_agent.cpp
	src/mass_agent/sensors.cpp
	src/config/geom_config.cpp
	src/config/agent_config.cpp
)
target_link_libraries(${PROJECT_NAME}
	$ENV{CARLA_910_CXX_API}/lib/libDetourCrowd.a
	$ENV{CARLA_910_CXX_API}/lib/libDetour.a
	$ENV{CARLA_910_CXX_API}/lib/librpc.a
	${catkin_LIBRARIES}
	${CARLA_LIBRARIES}
	${PCL_LIBRARIES}
	${OpenCV_LIBS}
	# Open3D::Open3D
	nanoflann::nanoflann
	OpenMP::OpenMP_CXX
	Eigen3::Eigen
	hdf5_dataset
	yaml-cpp
)