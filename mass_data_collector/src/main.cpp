#include <thread>
#include <chrono>
#include <csignal>
#include <future>
#include <random>
#include <algorithm>

#include <ros/ros.h>
#include <ros/package.h>

#include "config/agent_config.h"
#include "hdf5_api/hdf5_dataset.h"
#include "mass_agent/mass_agent.h"
#include "data_collection.h"

using namespace std::chrono_literals;

void SIGINT_handler(int signo);
void StatusThreadCallback(size_t*, size_t, char*, unsigned int*, unsigned int*, float*, bool*);
void AssertDataDimensions();
std::string SecondsToString(uint32);


int main(int argc, char **argv)
{
	// assertions, config & stats
	AssertDataDimensions();
	auto& config = CollectionConfig::GetConfig();
	CollectionStats stats(config.minimum_cars, config.maximum_cars);
	// ros init, signal handling
	ros::init(argc, argv, "mass_data_collector");
	std::shared_ptr<ros::NodeHandle> node_handle = std::make_shared<ros::NodeHandle>();
	signal(SIGINT, SIGINT_handler);

	// reading command line args --------------------------------------------------------------------------------
	bool debug_mode = false;
	size_t number_of_agents = 0;
	size_t batch_count = 0;
	if (argc == 1) {
		number_of_agents = config.maximum_cars;
	} else if (argc == 2) {
		if (std::strcmp(argv[1], "--debug") == 0) {
			number_of_agents = 3;
			debug_mode = true;
		} else {
			std::cout << "use --debug" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	} else {
		std::cout << "use param/data_collection.yaml for conf" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	srand(config.random_seed);
	ROS_INFO("starting data collection with up to %zu agent(s) for %u iterations",
			 number_of_agents, config.max_batch_count);
	// dataset --------------------------------------------------------------------------------------------------
	HDF5Dataset* dataset = nullptr;
	if (!debug_mode) {
		dataset = new HDF5Dataset(config.dataset_path, config.dataset_name,
								  mode::FILE_TRUNC | mode::DSET_CREAT, compression::NONE, 1,
								  (config.max_batch_count + 1) * config.maximum_cars, 32);
	}
	// CARLA setup ----------------------------------------------------------------------------------------------
	agent::MassAgent agents[number_of_agents];
	boost::shared_ptr<carla::client::Waypoint> random_pose;
	// some timing stuff ----------------------------------------------------------------------------------------
	float avg_batch_time = 0.0f;
	bool update = false;
	std::thread *time_thread = nullptr;
	char state = 'r';
	unsigned int batch_size = 0;
	unsigned int agents_done = 0;
	if (!debug_mode) {
		time_thread = new std::thread(StatusThreadCallback, &batch_count,  config.max_batch_count,
									  &state, &agents_done, &batch_size, &avg_batch_time, &update);
	}
	// debugging ------------------------------------------------------------------------------------------------
	if (debug_mode) {
		std::vector<unsigned int> indices(number_of_agents);
    	std::iota(indices.begin(), indices.end(), 0);
		std::cout << "creating uniform pointcloud" << std::endl;
		random_pose = agents[0].SetRandomPose(config::town0_restricted_roads);
		for (size_t i = 1; i < number_of_agents; ++i) {
			random_pose = agents[i].SetRandomPose(random_pose, 30, agents, indices, i,
												  config::town0_restricted_roads);
		}
		agent::MassAgent::DebugMultiAgentCloud(agents, number_of_agents, config.dataset_path + ".pcl");
		ros::shutdown();
	}
	// random distribution & shuffling necessities --------------------------------------------------------------
	std::mt19937 random_gen(config.random_seed);
    std::uniform_int_distribution<> distrib(config.minimum_cars, config.maximum_cars);
	std::vector<unsigned int> shuffled(number_of_agents);
    std::iota(shuffled.begin(), shuffled.end(), 0);
	// data collection loop -------------------------------------------------------------------------------------
	while (ros::ok()) {
		if (batch_count == config.max_batch_count) {
			break;
		}
		// timing
		auto batch_start_t = std::chrono::high_resolution_clock::now();
		// randoming the batch size and shuffling the agents
		batch_size = distrib(random_gen);
		stats.AddNewBatch(batch_size);
    	std::shuffle(shuffled.begin(), shuffled.end(), random_gen);
		// chain randoming poses for the chosen ones
		state = 'p';
		agents_done = 0;
		random_pose = agents[shuffled[0]].SetRandomPose(config::town0_restricted_roads);
		agents_done = 1;
		for (size_t i = 1; i < batch_size; ++i) {
			agents_done += 1;
			random_pose = agents[shuffled[i]].SetRandomPose(random_pose, 30, agents, shuffled, i,
															config::town0_restricted_roads);
		}
		// hiding the ones that didn't make it
		state = 'h'; // hiding
		for (size_t i = batch_size; i < number_of_agents; ++i) {
			agents[shuffled[i]].HideAgent();
		}
		// mandatory delay because moving cars in carla is not a blocking call
		std::this_thread::sleep_for(std::chrono::milliseconds(config.batch_delay_ms));

		std::future<MASSDataType> promise[batch_size];
		// gathering data (async)
		agents_done = 0;
		state = 'g';
		size_t agent_batch_index = 0;
		for (unsigned int i = 0; i < batch_size; ++i) {
			agents_done += 1;
			promise[i] = std::async(&agent::MassAgent::GenerateDataPoint, &agents[shuffled[i]], agent_batch_index++);
		}
		state = 's'; // saving
		agents_done = 0;
		for (unsigned int i = 0; i < batch_size; ++i) {
			agents_done += 1;
			MASSDataType datapoint = promise[i].get();
			dataset->AppendElement(&datapoint);
		}
		// timing stuff
		state = 't';
		batch_count += 1;
		auto batch_end_t = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> batch_duration = batch_end_t - batch_start_t;
		avg_batch_time = avg_batch_time + (1.0f / static_cast<float>(batch_count)) *
										  (batch_duration.count() - avg_batch_time);
		update = true;
	}
	if (!debug_mode) {
		std::cout << "\ndone, closing dataset..." << std::endl;
		auto batch_histogram = stats.AsVector();
		dataset->AddU32Attribute(&batch_histogram[0], batch_histogram.size(), "batch_histogram");
		unsigned int data[] = {config.maximum_cars};
		dataset->AddU32Attribute(data, 1, "max_agent_count");
		data[0] = config.minimum_cars;
		dataset->AddU32Attribute(data, 1, "min_agent_count");
		dataset->Close();
		time_thread->join();
	}
	return 0;
}

/* keybord interrupt handler */
void SIGINT_handler(int signo) {
	(void)signo;
	std::cout << "\nshutting down. please wait" << std::endl;
	ros::shutdown();
}

/* prints the status of data collection */
void StatusThreadCallback(size_t* batch_count, size_t max_batch_count, char* state, unsigned int* done,
						  unsigned int* batch_size, float* avg_batch_time, bool* update) {
	uint32 remaining_s = 0;
	uint32 elapsed_s = 0;
	std::string msg;
	while (ros::ok()) {
		msg = "\rgathering " + std::to_string(*batch_count + 1) + "/" + std::to_string(max_batch_count) +
			  ", S:" + *state + ":" + std::to_string(*done) + "/" + std::to_string(*batch_size) + ", E: " +
			  SecondsToString(elapsed_s);
		if (*avg_batch_time == 0) {
			msg += ", estimating remaining time ...";
		} else {
			msg += ", avg: " + std::to_string(*avg_batch_time) + "s" +
				   ", R: " + SecondsToString(remaining_s);
			remaining_s = std::max<int>(0, remaining_s - 1);
		}
		std::cout << msg + "    " << std::flush;
		std::this_thread::sleep_for(1s);
		elapsed_s += 1;
		if (*update) {
			if (*batch_count == max_batch_count) {
				break;
			}
			*update = false;
			remaining_s = (*avg_batch_time) * (max_batch_count - (*batch_count));
		}
	}
}

/* returns human readable string for the given period in seconds */
std::string SecondsToString(uint32 period_in_secs) {
	static unsigned int SECS_IN_HOUR = 60 * 60;
	static unsigned int SECS_IN_DAY = 60 * 60 * 24;
	uint32 days = 0;
	uint32 hours = 0;
	uint32 minutes = 0;
	if (period_in_secs >= SECS_IN_DAY) {
		days = period_in_secs / SECS_IN_DAY;
		period_in_secs %= SECS_IN_DAY;
	}
	if (period_in_secs >= SECS_IN_HOUR) {
		hours = period_in_secs / SECS_IN_HOUR;
		period_in_secs %= SECS_IN_HOUR;
	}
	if (period_in_secs >= 60) {
		minutes = period_in_secs / 60;
		period_in_secs %= 60;
	}
	return std::to_string(days) + "d" + std::to_string(hours) + "h" +
		   std::to_string(minutes) + "m" + std::to_string(period_in_secs) + "s";
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