#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <opencv2/core/types.hpp>
#include <thread>
#include <unistd.h>
#include <csignal>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <ros/ros.h>
#include <ros/package.h>

#include "mass_agent/mass_agent.h"
#include "ros/init.h"

#define RANDOM_SEED 135

using namespace std::chrono_literals;

void inter(int signo) {
	(void)signo;
	std::cout << "shutting down" << std::endl;
	ros::shutdown();
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "mass_data_collector");
	std::shared_ptr<ros::NodeHandle> node_handle = std::make_shared<ros::NodeHandle>();
	signal(SIGINT, inter);

	// reading command line args
	size_t number_of_agents = 1;
	size_t max_data_count = 1;
	size_t data_count = 0;
	if (argc != 2) {
		std::cout << "use: rosrun mass_data_collector <number of agents>" << std::endl;
	} else {
		number_of_agents = std::atoi(argv[1]); // NOLINT
	}
	srand(RANDOM_SEED);
	ROS_INFO("starting data collection...");
	agent::MassAgent agents[number_of_agents];
	boost::shared_ptr<carla::client::Waypoint> random_pose;
	while (ros::ok()) {
		if (++data_count > max_data_count) {
			break;
		}
		// chain randoming poses
		std::cout << "moving agent " << 0 << std::endl;
		random_pose = agents[0].SetRandomPose();
		for (size_t i = 1; i < number_of_agents; ++i) {
			std::cout << "moving agent " << i << std::endl;
			random_pose = agents[i].SetRandomPose(random_pose);
		}
		for (auto& agent : agents) {
			agent.CaptureOnce();
			agent.GenerateDataPoint();
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		ros::spinOnce();
	}
	std::cout << "\ndone" << std::endl;
	// agent.~MassAgent(); 
	return 0;
}
