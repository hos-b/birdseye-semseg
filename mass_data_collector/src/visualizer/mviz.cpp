#include <thread>
#include <chrono>
#include <csignal>
#include <future>

#include <ros/ros.h>
#include <ros/package.h>

#include "config/agent_config.h"
#include "hdf5_api/hdf5_dataset.h"
#include "mass_agent/mass_agent.h"
#include "data_collection.h"

using namespace std::chrono_literals;
using agent::MassAgent;

bool EnableSyncMode();

int main(int argc, char **argv)
{
	// ros init, signal handling ------------------------------------------------------
	ros::init(argc, argv, "mass_viz");
	std::shared_ptr<ros::NodeHandle> node_handle = std::make_shared<ros::NodeHandle>();
	// reading configs ----------------------------------------------------------------
	auto& config = CollectionConfig::GetConfig();
	size_t number_of_agents = config.maximum_cars;
    std::vector<unsigned int> not_shuffled(number_of_agents);
    std::iota(not_shuffled.begin(), not_shuffled.end(), 0);
	srand(config.random_seed);
	// random distribution & shuffling necessities ------------------------------------
	std::mt19937 random_gen(config.random_seed);
	// CARLA stuff --------------------------------------------------------------------
	auto& agents = MassAgent::agents();
    for (size_t i = 0; i < number_of_agents; ++i) {
        new MassAgent(random_gen, config::GetRestrictedRoads(3));
    }
	boost::shared_ptr<carla::client::Waypoint> random_pose;
	// data collection loop -----------------------------------------------------------
	// timing
	auto batch_start_t = std::chrono::high_resolution_clock::now();
	// chain randoming poses for the chosen ones, reset if placement fails
	do {
		random_pose = agents[0]->SetRandomPose();
		for (size_t i = 1; i < number_of_agents; ++i) {
			random_pose = agents[i]->SetRandomPose(random_pose, 32, not_shuffled, i);
		}
	} while (random_pose == nullptr);
	// Ticking() the simulator
	try {
		MassAgent::carla_client()->GetWorld().Tick(5s);
	} catch (carla::client::TimeoutException& ex) {
		std::cout << "connection to simulator timed out while Tick()ing" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	for (size_t caps = 0; caps < 5; ++caps) {
		for (size_t i = 0; i < number_of_agents; ++i) {
			agents[i]->CaptureOnce();
		}
		// Ticking() the simulator. for some reason once doesn't suffice
		try {
			MassAgent::carla_client()->GetWorld().Tick(5s);
			MassAgent::carla_client()->GetWorld().Tick(5s);
		} catch (carla::client::TimeoutException& ex) {
			std::cout << "connection to simulator timed out while Tick()ing" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}
	std::string dir = "/home/mass/data/";
	auto map = agents[0]->GetMap();
	agents[0]->SaveFullCloud(dir + "full_cloud.pcl");
	agents[0]->SaveMaskedClouds(dir + "front_mask.pcl", dir + "full_mask.pcl");
	MassAgent::SaveMultiAgentCloud(dir + "multiagent_cloud.pcl", 0);
	auto [rgb, semantic, mask] = agents[0]->GetBEVSample();
	cv::imwrite(dir + "front_rgb.png", rgb);
	cv::imwrite(dir + "bev_semantics.png", semantic);
	cv::imwrite(dir + "bev_mask.png", mask);
	cv::imwrite(dir + "map.png", map);

	std::cout << "exiting" << std::endl;
	for (size_t i = 0; i < agents.size(); ++i) {
		delete agents[i];
	}
	return 0;
}

/* set the simulator to synchronous mode */
bool EnableSyncMode() {
	// get sensor tick from sensors yaml
	std::string yaml_path = ros::package::getPath("mass_data_collector");
	yaml_path += "/param/sensors.yaml";
	YAML::Node base = YAML::LoadFile(yaml_path);
	YAML::Node front_msk_node = base["camera-mask"];
	double sensor_tick = front_msk_node["sensor_tick"].as<double>();
	auto settings = MassAgent::carla_client()->GetWorld().GetSettings();
	if (!settings.synchronous_mode || settings.fixed_delta_seconds != sensor_tick) {
		settings.synchronous_mode = true;
		settings.fixed_delta_seconds = sensor_tick;
		try {
			MassAgent::carla_client()->GetWorld().ApplySettings(settings, 5s);
		} catch (carla::client::TimeoutException& e) {
			std::cout << "switch to sync mode failed. connection timed out" << std::endl;
			return false;
		}
		std::cout << "synchronus mode enabled" << std::endl;
	}
	return true;
}