#include <chrono>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <thread>
#include <csignal>
#include <iomanip>
#include <future>
#include <ros/ros.h>
#include <ros/package.h>

#include "config/agent_config.h"
#include "hdf5_api/hdf5_dataset.h"
#include "mass_agent/mass_agent.h"

#define RANDOM_SEED 135

using namespace std::chrono_literals;
void inter(int signo);
void StatusThreadCallback(size_t* data_count, size_t max_data_count, float* avg_batch_time, bool* update);
void AssertDataDimensions();

int main(int argc, char **argv)
{
	AssertDataDimensions();
	ros::init(argc, argv, "mass_data_collector");
	std::shared_ptr<ros::NodeHandle> node_handle = std::make_shared<ros::NodeHandle>();
	signal(SIGINT, inter);
	bool debug_mode = false;
	// reading command line args --------------------------------------------------------------------------------
	size_t number_of_agents = 0;
	size_t max_data_count = 0;
	size_t data_count = 0;
	if (argc == 2) {
		if (std::strcmp(argv[1], "--debug") == 0) {
			number_of_agents = 1;
			max_data_count = 1;
			debug_mode = true;
		} else {
			number_of_agents = 1;
			max_data_count = 100;
		}
	} else if (argc == 3) {
		number_of_agents = std::atoi(argv[2]);
		max_data_count = 100;
	} else if (argc ==  4) {
		number_of_agents = std::atoi(argv[2]);
		max_data_count = std::atoi(argv[3]);
	} else {
		std::cout << "use: rosrun mass_data_collector dataset_name <number of agents> <data count>" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	std::string dset_name(argv[1]);
	dset_name += ".hdf5";
	srand(RANDOM_SEED);
	ROS_INFO("starting data collection with %zu agent(s) for %zu iterations", number_of_agents, max_data_count);
	// dataset --------------------------------------------------------------------------------------------------
	HDF5Dataset* dataset = nullptr;
	if (!debug_mode) {
		dataset = new HDF5Dataset("/home/hosein/" + dset_name, "dataset_1", mode::FILE_TRUNC | mode::DSET_CREAT,
								  compression::ZLIB | 6, 1, max_data_count + 32, 32, number_of_agents);
	}
	// CARLA setup ----------------------------------------------------------------------------------------------
	agent::MassAgent agents[number_of_agents];
	boost::shared_ptr<carla::client::Waypoint> random_pose;
	auto world = agent::MassAgent::carla_client()->GetWorld();
	// some timing stuff ----------------------------------------------------------------------------------------
	float avg_batch_time = 0.0f;
	bool update = false;
	std::thread *time_thread = nullptr;
	if (!debug_mode) {
		time_thread = new std::thread(StatusThreadCallback, &data_count, 
									  max_data_count, &avg_batch_time, &update);
	}
	// debugging ------------------------------------------------------------------------------------------------
	if (debug_mode) {
		std::cout << "creating uniform pointcloud" << std::endl;
		random_pose = agents[0].SetRandomPose(config::town0_restricted_roads);
		for (size_t i = 1; i < number_of_agents; ++i) {
			random_pose = agents[i].SetRandomPose(random_pose, config::town0_restricted_roads, 30);
		}
		agent::MassAgent::DebugMultiAgentCloud(agents, number_of_agents, "/home/hosein/debugcloud.pcl");
		ros::shutdown();
	}
	// data collection loop -------------------------------------------------------------------------------------
	while (ros::ok()) {
		std::future<MASSDataType> promise[number_of_agents];
		if (data_count == max_data_count) {
			break;
		}
		auto batch_start = std::chrono::high_resolution_clock::now();
		// chain randoming poses
		random_pose = agents[0].SetRandomPose(config::town0_restricted_roads);
		for (size_t i = 1; i < number_of_agents; ++i) {
			random_pose = agents[i].SetRandomPose(random_pose, config::town0_restricted_roads, 30);
		}
		// mandatory delay
		std::this_thread::sleep_for(100ms);
		// gathering data (async)
		for (unsigned int i = 0; i < number_of_agents; ++i) {
			promise[i] = std::async(&agent::MassAgent::GenerateDataPoint, &agents[i], 0.1, 35, 7);
		}
		for (unsigned int i = 0; i < number_of_agents; ++i) {
			MASSDataType datapoint = promise[i].get();
			dataset->AppendElement(&datapoint);
		}
		// timing stuff
		++data_count;
		auto batch_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> batch_duration = batch_end - batch_start;
		avg_batch_time = avg_batch_time + (1.0f / static_cast<float> (data_count)) *
										  (batch_duration.count() - avg_batch_time);
		update = true;
	}
	if (!debug_mode) {
		std::cout << "\ndone, closing dataset..." << std::endl;
		dataset->Close();
		time_thread->join();
	}
	return 0;
}

/* keybord interrupt handler */
void inter(int signo) {
	(void)signo;
	std::cout << "shutting down" << std::endl;
	ros::shutdown();
}
/* prints the status of data collection */
void StatusThreadCallback(size_t* data_count, size_t max_data_count, float* avg_batch_time, bool* update) {
	uint32 remaining_s = 0;
	while (ros::ok()) {
		if (*avg_batch_time == 0) {
			std::cout << '\r'
					  << "gathering " << *data_count + 1 << "/" << max_data_count
					  << "\testimating remaining time..." << std::flush;
			std::this_thread::sleep_for(1s);
		} else {
			std::cout << '\r'
					<< "gathering " << *data_count + 1 << "/" << max_data_count
					<< std::setw(20) << std::setfill(' ') << "avg. itr: " << *avg_batch_time << "s"
					<< std::setw(30) << std::setfill(' ') << "estimated remaining time: " << remaining_s / 60 << "m "
					<< remaining_s % 60 << "s   " << std::flush;
			std::this_thread::sleep_for(1s);
			remaining_s = std::max<int>(0, remaining_s - 1);
		}
		if (*update) {
			if (*data_count == max_data_count) {
				break;
			}
			*update = false;
			remaining_s = (*avg_batch_time) * (max_data_count - (*data_count));
		}
	}
}
/* asserts that the dataset image size is equal to that of the gathered samples */
void AssertDataDimensions() {
	std::string yaml_path = ros::package::getPath("mass_data_collector")
							+ "/param/sensors.yaml";
	YAML::Node sensors_base = YAML::LoadFile(yaml_path);
	yaml_path = ros::package::getPath("mass_data_collector")
				+ "/param/sc_settings.yaml";
	YAML::Node sconfig_base = YAML::LoadFile(yaml_path);
	bool failed = false;
	if (sensors_base["camera-front"]["image_size_x"].as<size_t>() != statics::front_rgb_width) {
		failed = true;
		std::cout << "rgb image width doesn't match the dataset struct size" << std::endl;
	}
	if (sensors_base["camera-front"]["image_size_y"].as<size_t>() != statics::front_rgb_height) {
		failed = true;
		std::cout << "rgb image height doesn't match the dataset struct size" << std::endl;
	}
	if (sconfig_base["bev"]["image_rows"].as<size_t>() != statics::top_semseg_height) {
		failed = true;
		std::cout << "bev image height doesn't match the dataset struct size" << std::endl;
	}
	if (sconfig_base["bev"]["image_cols"].as<size_t>() != statics::top_semseg_width) {
		failed = true;
		std::cout << "bev image width doesn't match the dataset struct size" << std::endl;
	}
	if (failed) {
		std::exit(EXIT_FAILURE);
	}
}