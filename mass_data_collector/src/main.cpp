#include <chrono>
#include <cstdlib>
#include <opencv2/imgcodecs.hpp>
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

/* 
To Test:
	[5.7187e+01, 1.9257e+02] # buildings look weird
	[2.2314e+02, 2.0064e+02] # tunnel
	[9.94530000, 138.730700] # inside
	[9.97690000, 140.011500] # this one
	[147.305500, 151.092400] # buildings look weird again

	[1.6323e+02, 1.9390e+02] # end of junction leading to tunnel
	[170.402800, 193.900000] # beginning of tunnel
	[174.264400, 193.900000] # more inside the tunnel

	[-1.489e+02, 7.9915e+01] # under tram line
	[-1.4547e+02,8.0353e+01] # under tram line

	[-74.1480, -79.5321] # fine but has a weird building artifact
	[-71.1481, -79.5239] # to the right on a bike lane ?
	[-69.5806, -79.5196] # to the right on the sidewalk!

	[244.0300,  78.4943] # inside tunnel and another car
	[244.2139,  86.0548] # inside tunnel
	[244.0843,  80.7248] # inside tunnel
	[234.0437,  99.7434] # ...
	[234.2429, 107.9337] # ...
	[234.1480, 104.0332] # ...

	[ 84.2659, -68.3524] # fine
	[ 85.3140, -70.1346] # to its left on gravel
	[ 87.7725, -65.8354] # in front, on gravel

	[ -35.6166, -177.2975] # gas station?
	[ -29.9447, -174.1371] # yes?
	[ -25.4275, -172.1239] # maybe

	[248.6527, 147.9556] # tunnel and side lane
	[251.6419, 148.2102] # tunnel and side lane
	[250.8483, 154.8452] # tunnel

	[-24.7947,  -5.7120] # fine
	[-26.2067,  -7.1284] # to its left and on side lane
	[-28.7932,  -5.2003] # side lane

	[87.2907, 37.3840] # side lane
	[105.7390,  78.6139] # side lane again
 */

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
	HDF5Dataset dataset("/home/hosein/test.hdf5", "dataset_1", mode::FILE_TRUNC | mode::DSET_CREAT,
						compression::ZLIB | 6, 1, 20000, 32, number_of_agents);
	// CARLA setup -------------------------------------------------------------------------------------
	agent::MassAgent agents[number_of_agents];
	boost::shared_ptr<carla::client::Waypoint> random_pose;
	auto world = agent::MassAgent::carla_client()->GetWorld();
	// addings vehicle masks to the dataset ------------------------------------------------------------
	for (size_t i = 0; i < number_of_agents; ++i) {
		auto vmask = agents[i].CreateVehicleMask();
		cv::imwrite("/home/hosein/catkin_ws/src/mass_data_collector/guide/vmask_" + std::to_string(i) + ".png", vmask);
	}
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
		std::this_thread::sleep_for(100ms);
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
