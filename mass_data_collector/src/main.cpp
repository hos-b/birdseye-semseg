#include <thread>
#include <chrono>
#include <csignal>
#include <future>
#include <algorithm>

#include <ros/ros.h>
#include <ros/package.h>

#include "config/agent_config.h"
#include "hdf5_api/hdf5_dataset.h"
#include "mass_agent/mass_agent.h"
#include "data_collection.h"

using namespace std::chrono_literals;

void SIGINT_handler(int signo);
void WatchdogThreadCallback(size_t*, size_t, char*, unsigned int*, unsigned int*, float*, bool*, bool*);
void AssertConfiguration();
bool SwitchTown(size_t, size_t, std::unordered_map<int, bool>&, std::mt19937&);
std::string SecondsToString(uint32);

using agent::MassAgent;

int main(int argc, char **argv)
{
	// assertions, config & stats
	AssertConfiguration();
	auto& config = CollectionConfig::GetConfig();
	CollectionStats stats(config.minimum_cars, config.maximum_cars);
	std::unordered_map<int, bool> restricted_roads;
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
	std::cout << "collecting data with up to " << number_of_agents << " agents for "
			  << config.cumulative_batch_counts.back() << " iterations in towns " << config.towns_string() << "\n";
	// dataset --------------------------------------------------------------------------------------------------
	HDF5Dataset* dataset = nullptr;
	if (!debug_mode) {
		unsigned int mode = mode::FILE_TRUNC | mode::DSET_CREAT;
		if (config.append) {
			mode = mode::FILE_RDWR | mode::DSET_OPEN;
			std::cout << "appending to the dataset ...\n";
		}
		dataset = new HDF5Dataset(config.dataset_path, config.dataset_name,
								  mode, compression::NONE, 1,
								 (config.max_batch_count + 1) * config.maximum_cars,
								  config.hdf5_chunk_size);
		auto [sz, max_sz] = dataset->GetCurrentSize();
		if ((max_sz - sz + 1) < config.cumulative_batch_counts.back() * config.maximum_cars) {
			std::cout << "not enough remaining space for given round batch count" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}
	// CARLA setup ----------------------------------------------------------------------------------------------
	boost::shared_ptr<carla::client::Waypoint> random_pose;
	// some timing stuff ----------------------------------------------------------------------------------------
	float avg_batch_time = 0.0f;
	bool batch_finished = false;
	std::thread *time_thread = nullptr;
	char state = 'i';
	unsigned int batch_size = 0;
	unsigned int agents_done = 0;
	bool deadlock = false;
	if (!debug_mode) {
		time_thread = new std::thread(WatchdogThreadCallback, &batch_count, config.cumulative_batch_counts.back(),
									  &state, &agents_done, &batch_size, &avg_batch_time, &batch_finished, &deadlock);
	}
	// random distribution & shuffling necessities --------------------------------------------------------------
	std::mt19937 random_gen(config.random_seed);
    std::uniform_int_distribution<> distrib(config.minimum_cars, config.maximum_cars);
	std::vector<unsigned int> shuffled(number_of_agents);
    std::iota(shuffled.begin(), shuffled.end(), 0);
	// debugging ------------------------------------------------------------------------------------------------
	auto& agents = agent::MassAgent::agents();
	if (debug_mode) {
		SwitchTown(0, number_of_agents, restricted_roads, random_gen);
		std::vector<unsigned int> indices(number_of_agents);
    	std::iota(indices.begin(), indices.end(), 0);
		std::cout << "creating uniform pointcloud" << std::endl;
		random_pose = agents[0]->SetRandomPose(restricted_roads);
		for (size_t i = 1; i < number_of_agents; ++i) {
			random_pose = agents[i]->SetRandomPose(random_pose, 30, &deadlock,
												   indices, i, restricted_roads);
		}
		agent::MassAgent::DebugMultiAgentCloud(config.dataset_path + ".pcl");
		ros::shutdown();
	}
	// data collection loop -------------------------------------------------------------------------------------
	while (ros::ok()) {
		if (batch_count == config.cumulative_batch_counts.back()) {
			break;
		}
		state = 'i';
		if (!SwitchTown(batch_count, number_of_agents, restricted_roads, random_gen)) {
			break;
		}
		// timing
		auto batch_start_t = std::chrono::high_resolution_clock::now();
		// randoming the batch size and shuffling the agents
		state = 'r';
		batch_size = distrib(random_gen);
		stats.AddNewBatch(batch_size);
    	std::shuffle(shuffled.begin(), shuffled.end(), random_gen);
		// enabling callbacks for chosen ones
		for (size_t i = 0; i < batch_size; ++i) {
			agents[shuffled[i]]->ResumeSensorCallbacks();
		}
		state = 'p';
		// chain randoming poses for the chosen ones, reset on deadlock
		do {
			deadlock = false;
			agents_done = 0;
			random_pose = agents[shuffled[0]]->SetRandomPose(restricted_roads);
			agents_done = 1;
			for (size_t i = 1; i < batch_size; ++i) {
				agents_done += 1;
				random_pose = agents[shuffled[i]]->SetRandomPose(random_pose, 32, &deadlock,
																 shuffled, i, restricted_roads);
			}
		} while (deadlock);
		// hiding & disabling callbacks for the ones that didn't make it
		state = 'h';
		for (size_t i = batch_size; i < number_of_agents; ++i) {
			agents[shuffled[i]]->PauseSensorCallbacks();
			agents[shuffled[i]]->HideAgent();
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
			promise[i] = std::async(&agent::MassAgent::GenerateDataPoint
									<geom::CloudBackend::KD_TREE>,
									agents[shuffled[i]], agent_batch_index++);
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
		batch_finished = true;
	}
	if (!debug_mode) {
		std::cout << "\ndone, closing dataset..." << std::endl;
		auto batch_histogram = stats.AsVector();
		if (config.append) {
			// updating histogram from last run
			std::vector<uint32_t> old_stats(batch_histogram.size(), 0);
			dataset->ReadU32Attribute("batch_histogram", &old_stats[0]);
			for (size_t i = 0; i < batch_histogram.size(); ++i) {
				batch_histogram[i] += old_stats[i];
			}
			dataset->UpdateU32Attribute("batch_histogram", &batch_histogram[0]);
			// updating max agent count if it's larger in this round
			uint32_t data[] = {0};
			dataset->ReadU32Attribute("max_agent_count", data);
			if (config.maximum_cars > data[0]) {
				data[0] = config.maximum_cars;
				dataset->UpdateU32Attribute("max_agent_count", data);
			}
		} else {
			dataset->AddU32Attribute(&batch_histogram[0], batch_histogram.size(), "batch_histogram");
			uint32_t data[] = {config.maximum_cars};
			dataset->AddU32Attribute(data, 1, "max_agent_count");
			data[0] = config.minimum_cars;
			dataset->AddU32Attribute(data, 1, "min_agent_count");
		}
		dataset->Close();
		time_thread->join();
	}
	for (size_t i = 0; i < agents.size(); ++i) {
		delete agents[i];
	}
	return 0;
}

/* keybord interrupt handler */
void SIGINT_handler(int signo) {
	(void)signo;
	std::cout << "\nshutting down. please wait" << std::endl;
	ros::shutdown();
}

/* prints the status of data collection, detects deadlocks */
void WatchdogThreadCallback(size_t* batch_count, size_t max_batch_count, char* state,
							unsigned int* done, unsigned int* batch_size, float* avg_batch_time,
							bool* batch_finished, bool* deadlock) {
	uint32 remaining_s = 0;
	uint32 elapsed_s = 0;
	uint32 batch_s = 0;
	auto deadlock_multiplier = CollectionConfig::GetConfig().deadlock_multiplier;
	std::string msg;
	while (ros::ok()) {
		// do nothing while in initialization phase
		while (*state == 'i') {
			std::this_thread::sleep_for(1s);
		}
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
		batch_s += 1;
		if (*batch_finished) {
			if (*batch_count == max_batch_count) {
				break;
			}
			*batch_finished = false;
			remaining_s = (*avg_batch_time) * (max_batch_count - (*batch_count));
			batch_s = 0;
		// deadlock detection
		} else if (*batch_count > 0 && batch_s >= (*avg_batch_time) * deadlock_multiplier) {
			*deadlock = true;
			std::cout << "\ndeadlock detected at batch #" << *batch_count + 1
					  << ", retrying ..." << std::endl;
			batch_s = 0;
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
void AssertConfiguration() {
	auto& col_conf = CollectionConfig::GetConfig();
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
	for (auto& town_batch_count : col_conf.town_batch_counts) {
		if ((town_batch_count + 1) * col_conf.maximum_cars < col_conf.hdf5_chunk_size) {
			failed = true;
			std::cout << "hdf5 chunk size must be smaller than the collected data" << std::endl;
		}
		if (town_batch_count == 0) {
			failed = true;
			std::cout << "town batch size cannot be 0" << std::endl;
		}
	}
	for (auto& town_no : col_conf.towns) {
		if (town_no < 1 || town_no > 10 || town_no == 8 || town_no == 9) {
			failed = true;
			std::cout << "invalid town number " << town_no << ", expected 1-7 or 10" << std::endl;
		}
	}
	if (col_conf.towns.size() != col_conf.town_batch_counts.size()) {
		failed = true;
		std::cout << "number of towns " << col_conf.towns.size()
				  << " doesn't match batch counts " << col_conf.town_batch_counts.size() << std::endl;
	}
	if (failed) {
		std::exit(EXIT_FAILURE);
	}
}

/* switch town based on the batch number */
bool SwitchTown(size_t batch, size_t number_of_agents, std::unordered_map<int, bool>& restricted_roads,
				std::mt19937& random_gen) {
	auto& col_conf = CollectionConfig::GetConfig();
	auto& agents = MassAgent::agents();
	if (batch == col_conf.max_batch_count) {
		throw std::runtime_error("should not be reaching this line");
	}
	if (batch == 0) {
		std::string new_town = config::town_map_strings.at(col_conf.towns[0]);
		auto current_town = agent::MassAgent::carla_client()->GetWorld().GetMap()->GetName();
		if (current_town != new_town) {
			std::cout << "switching to first town: " << new_town << std::endl;
			agent::MassAgent::carla_client()->LoadWorld(new_town);
			for (size_t i = 0; i < number_of_agents; ++i) {
				new agent::MassAgent(random_gen);
			}
			restricted_roads = *config::restricted_roads[col_conf.towns[0]];
		}
		return true;
	}
	for (size_t i = 1; i < col_conf.cumulative_batch_counts.size(); ++i) {
		if (batch == col_conf.cumulative_batch_counts[i - 1]) {
			std::string new_town = config::town_map_strings.at(col_conf.towns[i]);
			std::cout << "\nswitching to " << new_town << std::endl;
			for (size_t i = 0; i < number_of_agents; ++i) {
				delete agents[i];
			}
			agents.clear();
			auto current_town = agent::MassAgent::carla_client()->GetWorld().GetMap()->GetName();
			// in case same town is used consecutively to force agent changing/recoloring
			if (current_town != new_town) {
				try {
					agent::MassAgent::carla_client()->LoadWorld(new_town);
				} catch(carla::client::TimeoutException& e) {
					std::cout << "connection to simulator timed out..." << std::endl;
					return false;
				}
				std::this_thread::sleep_for(10s);
			}
			for (size_t i = 0; i < number_of_agents; ++i) {
				new agent::MassAgent(random_gen);
			}
			restricted_roads = *config::restricted_roads[col_conf.towns[i]];
			std::cout << "switch complete, gathering data..." << std::endl;
			return true;
		}
	}
	return true;
}