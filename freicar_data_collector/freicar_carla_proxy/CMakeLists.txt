# building compiler database for ccls
# cmake -H. -BDebug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=YES
# ln -s Debug/compile_commands.json .
cmake_minimum_required(VERSION 2.8.3)
project(freicar_carla_proxy)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


## Find catkin macros and libraries
find_package(Threads REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package (Eigen3 3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS)

# Carla related stuff
if (NOT DEFINED ENV{CARLA_CXX_API})
    message(FATAL_ERROR "environment variable CARLA_CXX_API is not defined. export it before compiling")
endif ()
FILE(GLOB CARLA_LIBRARIES $ENV{CARLA_CXX_API}/lib/*.a)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	roslib
	std_msgs
	nav_msgs
	geometry_msgs
	tf2
	tf2_ros
	cv_bridge
	image_transport
	freicar_common
	raiscar_msgs
	sensor_msgs
)

catkin_package(CATKIN_DEPENDS 
	roscpp
	roslib
	std_msgs
	nav_msgs
	geometry_msgs
	tf2
	tf2_ros
	cv_bridge
	image_transport
	freicar_common
	raiscar_msgs
	sensor_msgs
)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	$ENV{CARLA_CXX_API}/include
	$ENV{CARLA_CXX_API}/include/system
	${SUBDIRS}
	${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# aiscar_airsim_agent node ---------------------------------------------
add_executable(${PROJECT_NAME}_node src/main.cpp
									src/freicar_carla_proxy.cpp
)
add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(${PROJECT_NAME}_node ${CMAKE_THREAD_LIBS_INIT}
										   ${catkin_LIBRARIES}
										   ${CARLA_LIBRARIES}
										   $ENV{CARLA_CXX_API}/lib/libDetourCrowd.a
										   $ENV{CARLA_CXX_API}/lib/libDetour.a
										   $ENV{CARLA_CXX_API}/lib/librpc.a
										   Eigen3::Eigen
										   yaml-cpp
										   pthread
										   ${PCL_LIBRARIES}
										   ${OpenCV_LIBS}
										   # png
										   # jpeg
										   # tiff
)
