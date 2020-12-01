#include <chrono>
#include <cstdlib>
#include <thread>
#include <csignal>
#include <iomanip>

#include <ros/ros.h>
#include <ros/package.h>

#include "hdf5_api/hdf5_dataset.h"
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

	// reading command line args -----------------------------------------------------------------------
	size_t number_of_agents = 0;
	size_t max_data_count = 0;
	size_t data_count = 0;
	if (argc == 1) {
		number_of_agents = 1;
		max_data_count = 100;
	} else if (argc == 2) {
		number_of_agents = std::atoi(argv[1]); // NOLINT
		max_data_count = 100;
	} else if (argc ==  3) {
		number_of_agents = std::atoi(argv[1]); // NOLINT
		max_data_count = std::atoi(argv[2]); // NOLINT
	} else {
		std::cout << "use: rosrun mass_data_collector <number of agents> <data count>" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	srand(RANDOM_SEED);
	ROS_INFO("starting data collection with %zu agent(s) for %zu iterations", number_of_agents, max_data_count);
	// dataset -----------------------------------------------------------------------------------------
	HDF5Dataset dataset("/home/hosein/dataset_1.hdf5", "dataset_1", mode::FILE_TRUNC | mode::DSET_CREAT,
						compression::ZLIB | 6, 1, 20000, 32, number_of_agents);
	// creating agents
	agent::MassAgent agents[number_of_agents];
	boost::shared_ptr<carla::client::Waypoint> random_pose;
	// timing ------------------------------------------------------------------------------------------
	auto program_start = std::chrono::high_resolution_clock::now();
	float avg_batch_time = 0.0f;
	// data collection loop ----------------------------------------------------------------------------
	while (ros::ok()) {
		if (++data_count > max_data_count) {
			break;
		}
		auto batch_start = std::chrono::high_resolution_clock::now();
		// chain randoming poses
		random_pose = agents[0].SetRandomPose();
		for (size_t i = 1; i < number_of_agents; ++i) {
			random_pose = agents[i].SetRandomPose(random_pose);
		}
		// gathering data
		for (auto& agent : agents) {
			agent.CaptureOnce(false);
			MASSDataType datapoint = agent.GenerateDataPoint();
			dataset.AppendElement(&datapoint);
		}
		// timing stuff
		auto batch_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> batch_duration = batch_end - batch_start;
		std::chrono::duration<float> elapsed = batch_end - program_start;
		avg_batch_time = avg_batch_time + (1.0f / static_cast<float> (data_count)) *
										  (batch_duration.count() - avg_batch_time);
		uint32 remaining_s = (avg_batch_time * (max_data_count - data_count)); // NOLINT
		std::cout << '\r'
				  << "gathered " << data_count << "/" << max_data_count
				  << std::setw(20) << std::setfill(' ') << "avg. itr: " << avg_batch_time << "s"
				  << std::setw(20) << std::setfill(' ') << "elapsed: " << (elapsed.count()) << "s"
				  << std::setw(30) << std::setfill(' ') << "estimated remaining time: " << remaining_s / 60 << "m "
				  << remaining_s % 60 << "s" << std::flush;
		ros::spinOnce();
	}
	std::cout << "\ndone, closing dataset..." << std::endl;
	dataset.Close();
	// agent.~MassAgent(); 
	return 0;
}
