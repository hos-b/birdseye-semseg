#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <csignal>
#include <sstream>
#include <vector>

#include <ros/ros.h>
#include <ros/package.h>

#include "mass_agent/mass_agent.h"

#define RANDOM_SEED 135
#define MAP_INIT_SLEEP 2.0
#define MAP_DENSIFICATION_LIMIT 0.22f
#define CALRA_PORT 2000

void InitializeFreicarMap(freicar::map::ThriftMapProxy& proxy);
int main(int argc, char **argv)
{
	ros::init(argc, argv, "mass_data_collector");
	std::shared_ptr<ros::NodeHandle> node_handle = std::make_shared<ros::NodeHandle>();
	size_t number_of_agents = 0;
	// reading command line args
	if (argc != 2) {
		number_of_agents = 1;
		std::cout << "use: rosrun mass_data_collector <number of agents>" << std::endl;
	} else {
		number_of_agents = std::atoi(argv[1]); // NOLINT
	}
	// thrift map intialization
	freicar::map::ThriftMapProxy map_proxy;
	InitializeFreicarMap(map_proxy);
	ROS_INFO("starting data collection...");
	srand(RANDOM_SEED);
	std::vector<MassAgent*> agents;
	for (size_t i = 0; i < number_of_agents; ++i) {
		agents.emplace_back(new MassAgent()); // NOLINT
		agents.back()->ActivateCarlaAgent("127.0.0.1", CALRA_PORT);
		agents.back()->SetRandomPose();
	}
	ros::spin();
	return 0;
}

void InitializeFreicarMap(freicar::map::ThriftMapProxy& proxy) {
	std::string map_path = ros::package::getPath("freicar_map") + "/maps/thriftmap_fix.aismap";
	// if should wait for network or the map can't be loaded
	if (!proxy.LoadMapFromFile(map_path)) {
		ROS_ERROR("could not find thriftmap file: %s", map_path.c_str());
	}
	ros::Duration(MAP_INIT_SLEEP).sleep();
	freicar::map::Map::GetInstance().PostProcess(MAP_DENSIFICATION_LIMIT);
}