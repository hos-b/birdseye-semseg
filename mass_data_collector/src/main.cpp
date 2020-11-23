#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unistd.h>
#include <csignal>
#include <sstream>
#include <vector>

#include <ros/ros.h>
#include <ros/package.h>

#include "mass_agent/mass_agent.h"

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
	size_t number_of_agents = 0;
	size_t max_data_count = 1;
	size_t data_count = 0;
	if (argc != 2) {
		number_of_agents = 1;
		std::cout << "use: rosrun mass_data_collector <number of agents>" << std::endl;
	} else {
		number_of_agents = std::atoi(argv[1]); // NOLINT
	}
	srand(RANDOM_SEED);
	ROS_INFO("starting data collection...");
	agent::MassAgent agent;
	while (ros::ok()) {
		if (data_count++ < max_data_count) {
			agent.CaptureOnce();
			agent.GenerateDataPoint();
			// agent.SetRandomPose();
		}
	}
	// ros::spin();
	// agent.~MassAgent(); 
	return 0;
}
