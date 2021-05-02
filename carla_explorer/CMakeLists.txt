cmake_minimum_required (VERSION 3.12)
# cmake_policy(SET CMP0048 NEW)
project(carla_explorer)

# cmake flags ---------------------------------------------------------------
set(CMAKE_CXX_FLAGS "-O0 -g -Wall -Wextra")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS YES)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# required for stuff and things ---------------------------------------------
find_package(Eigen3 3 REQUIRED)
# CARLA related stuff -------------------------------------------------------
if (NOT DEFINED ENV{CARLA_910_CXX_API})
	message(FATAL_ERROR "environment variable CARLA_910_CXX_API is not defined.")
endif ()
FILE(GLOB CARLA_LIBRARIES $ENV{CARLA_910_CXX_API}/lib/*.a)
# catkin packages -----------------------------------------------------------
find_package(catkin REQUIRED COMPONENTS
	roscpp
	roslib
	sensor_msgs
)
# more catkin stuff ---------------------------------------------------------
catkin_package(CATKIN_DEPENDS 
	roscpp
	roslib
	sensor_msgs
)

# include dirs --------------------------------------------------------------
include_directories(
	"/usr/lib/gcc/x86_64-linux-gnu/9/include;"
	${PROJECT_SOURCE_DIR}/include
	${catkin_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	$ENV{CARLA_910_CXX_API}/include
	$ENV{CARLA_910_CXX_API}/include/system
)
# data collection executable ------------------------------------------------
add_executable(${PROJECT_NAME}
	src/main.cpp
	src/transform_conversions.cpp
)
target_link_libraries(${PROJECT_NAME}
	$ENV{CARLA_910_CXX_API}/lib/libDetourCrowd.a
	$ENV{CARLA_910_CXX_API}/lib/libDetour.a
	$ENV{CARLA_910_CXX_API}/lib/librpc.a
	${catkin_LIBRARIES}
	${CARLA_LIBRARIES}
)