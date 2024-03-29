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

void SIGINT_handler(int signo);
void WatchdogThreadCallback(size_t*, char*, unsigned int*, unsigned int*, float*, bool*);
void AssertConfiguration();
bool SwitchTown(size_t, const std::string& weather, std::mt19937&);
std::string SecondsToString(uint32);
bool EnableSyncMode();

int main(int argc, char **argv)
{
	// ros init, signal handling --------------------------------------------------------------------------------
	ros::init(argc, argv, "mass_data_collector");
	std::shared_ptr<ros::NodeHandle> node_handle = std::make_shared<ros::NodeHandle>();
	signal(SIGINT, SIGINT_handler);
	// reading configs ------------------------------------------------------------------------------------------
	AssertConfiguration();
	auto& config = CollectionConfig::GetConfig();
	CollectionStats stats(config.maximum_cars);
	size_t number_of_agents = config.maximum_cars;
	srand(config.random_seed);
	size_t batch_count = 0;
	std::cout << "collecting data with up to " << number_of_agents
			  << " agents for " << config.cumulative_batch_counts.back()
			  << " iterations in towns " << config.towns_string() << "\n";
	// dataset --------------------------------------------------------------------------------------------------
	unsigned int mode = mode::FILE_TRUNC | mode::DSET_CREAT;
	if (config.append) {
		mode = mode::FILE_RDWR | mode::DSET_OPEN;
		std::cout << "appending to the dataset ...\n";
	}
	HDF5Dataset dataset(config.dataset_path, config.dataset_name,
						mode, compression::NONE, 1,
					   (config.max_batch_count + 1) * config.maximum_cars,
						config.hdf5_chunk_size);
	auto [sz, max_sz] = dataset.GetCurrentSize();
	if ((max_sz - sz + 1) < config.cumulative_batch_counts.back() * config.maximum_cars) {
		std::cout << "not enough remaining space for given round batch count" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	// some timing stuff ----------------------------------------------------------------------------------------
	float avg_batch_time = 0.0f;
	bool batch_finished = false;
	char state = 'i';
	unsigned int batch_size = 0;
	unsigned int agents_done = 0;
	std::thread time_thread(WatchdogThreadCallback, &batch_count, &state, &agents_done,
							&batch_size, &avg_batch_time, &batch_finished);
	// random distribution & shuffling necessities --------------------------------------------------------------
	std::mt19937 random_gen(config.random_seed);
	std::vector<unsigned int> shuffled(number_of_agents);
    std::iota(shuffled.begin(), shuffled.end(), 0);
	// CARLA stuff ----------------------------------------------------------------------------------------------
	auto& agents = MassAgent::agents();
	boost::shared_ptr<carla::client::Waypoint> random_pose;
	// data collection loop -------------------------------------------------------------------------------------
	while (ros::ok()) {
		if (batch_count == config.cumulative_batch_counts.back()) {
			break;
		}
		state = 'i';
		auto [switch_flag, town_index, town_weather] = config.GetNewTown(batch_count);
		// exit if town switch was unsuccessful
		if (switch_flag && !SwitchTown(town_index, town_weather, random_gen)) {
			break;
		}
		// timing
		auto batch_start_t = std::chrono::high_resolution_clock::now();
		// randoming the batch size and shuffling the agents
		state = 'r';
		batch_size = config.GetBatchSize(batch_count);
		stats.AddNewBatch(batch_size);
    	std::shuffle(shuffled.begin(), shuffled.end(), random_gen);
		state = 'r';
		// hiding peasants
		for (size_t i = batch_size; i < number_of_agents; ++i) {
			agents[shuffled[i]]->HideAgent();
		}
		// chain randoming poses for the chosen ones, reset if placement fails
		do {
			random_pose = agents[shuffled[0]]->SetRandomPose();
			agents_done = 1;
			for (size_t i = 1; i < batch_size; ++i) {
				agents_done += 1;
				random_pose = agents[shuffled[i]]->SetRandomPose(random_pose, 32, shuffled, i);
			}
		} while (random_pose == nullptr);
		// Ticking() the simulator
		try {
			MassAgent::carla_client()->GetWorld().Tick(5s);
		} catch (carla::client::TimeoutException& ex) {
			std::cout << "connection to simulator timed out while Tick()ing" << std::endl;
			break;
		}
		// capturing frames
		state = 'c';
		agents_done = 0;
		for (size_t i = 0; i < batch_size; ++i) {
			agents[shuffled[i]]->CaptureOnce();
		}
		state = 't';
		// Ticking() the simulator. for some reason once doesn't suffice
		try {
			MassAgent::carla_client()->GetWorld().Tick(5s);
			MassAgent::carla_client()->GetWorld().Tick(5s);
		} catch (carla::client::TimeoutException& ex) {
			std::cout << "connection to simulator timed out while Tick()ing" << std::endl;
			break;
		}
		std::future<MASSDataType> promise[batch_size];
		// gathering data (async)
		state = 'g';
		agents_done = 0;
		size_t agent_batch_index = 0;
		for (unsigned int i = 0; i < batch_size; ++i) {
			agents_done += 1;
			promise[i] = std::async(&MassAgent::GenerateDataPoint,
									agents[shuffled[i]], agent_batch_index++);
		}
		state = 's'; // saving
		agents_done = 0;
		for (unsigned int i = 0; i < batch_size; ++i) {
			agents_done += 1;
			MASSDataType datapoint = promise[i].get();
			dataset.AppendElement(&datapoint);
		}
		// timing stuff
		batch_count += 1;
		auto batch_end_t = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> batch_duration = batch_end_t - batch_start_t;
		avg_batch_time = avg_batch_time + (1.0f / static_cast<float>(batch_count)) *
										  (batch_duration.count() - avg_batch_time);
		batch_finished = true;
	}
	auto batch_histogram = stats.AsVector();
	std::cout << "\nbatch size histogram:\n";
	stats.Print();
	if (config.append) {
		// updating histogram from last run
		std::vector<uint32_t> old_stats(batch_histogram.size(), 0);
		dataset.ReadU32Attribute("batch_histogram", &old_stats[0]);
		for (size_t i = 0; i < batch_histogram.size(); ++i) {
			batch_histogram[i] += old_stats[i];
		}
		dataset.UpdateU32Attribute("batch_histogram", &batch_histogram[0]);
		// updating max agent count if it's larger in this round
		uint32_t data[] = {0};
		dataset.ReadU32Attribute("max_agent_count", data);
		if (config.maximum_cars > data[0]) {
			data[0] = config.maximum_cars;
			dataset.UpdateU32Attribute("max_agent_count", data);
		}
	} else {
		dataset.AddU32Attribute(&batch_histogram[0], batch_histogram.size(), "batch_histogram");
		uint32_t data[] = {config.maximum_cars};
		dataset.AddU32Attribute(data, 1, "max_agent_count");
		data[0] = 1;
		dataset.AddU32Attribute(data, 1, "min_agent_count");
	}
	dataset.Close();
	std::cout << "data collection finished, dataset closed" << std::endl;
	ros::shutdown();
	time_thread.join();
	std::cout << "joined timer thread" << std::endl;
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
void WatchdogThreadCallback(size_t* batch_count, char* state, unsigned int* done,
							unsigned int* batch_size, float* avg_batch_time, bool* batch_finished) {
	uint32 remaining_s = 0;
	uint32 elapsed_s = 0;
	std::string msg;
	size_t max_batch_count = CollectionConfig::GetConfig().cumulative_batch_counts.back();
	while (ros::ok()) {
		// do nothing while in initialization phase (very beginning and during town switches)
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
		if (*batch_finished) {
			if (*batch_count == max_batch_count) {
				break;
			}
			*batch_finished = false;
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
	if (failed || !EnableSyncMode()) {
		std::exit(EXIT_FAILURE);
	}
}

/* switch town based on the batch number */
bool SwitchTown(size_t town_index, const std::string& weather, std::mt19937& random_gen) {
	static const std::map<std::string, carla::rpc::WeatherParameters> weather_map = {
		{"Default", carla::rpc::WeatherParameters::Default},
		{"ClearNoon", carla::rpc::WeatherParameters::ClearNoon},
		{"CloudyNoon", carla::rpc::WeatherParameters::CloudyNoon},
		{"WetNoon", carla::rpc::WeatherParameters::WetNoon},
		{"WetCloudyNoon", carla::rpc::WeatherParameters::WetCloudyNoon},
		{"MidRainyNoon", carla::rpc::WeatherParameters::MidRainyNoon},
		{"HardRainNoon", carla::rpc::WeatherParameters::HardRainNoon},
		{"SoftRainNoon", carla::rpc::WeatherParameters::SoftRainNoon},
		{"ClearSunset", carla::rpc::WeatherParameters::ClearSunset},
		{"CloudySunset", carla::rpc::WeatherParameters::CloudySunset},
		{"WetSunset", carla::rpc::WeatherParameters::WetSunset},
		{"WetCloudySunset", carla::rpc::WeatherParameters::WetCloudySunset},
		{"MidRainSunset", carla::rpc::WeatherParameters::MidRainSunset},
		{"HardRainSunset", carla::rpc::WeatherParameters::HardRainSunset},
		{"SoftRainSunset", carla::rpc::WeatherParameters::SoftRainSunset}
	};
	auto& col_conf = CollectionConfig::GetConfig();
	auto& agents = MassAgent::agents();
	auto number_of_agents = col_conf.maximum_cars;
	std::cout << std::endl;
	for (size_t j = 0; j < agents.size(); ++j) {
		delete agents[j];
	}
	agents.clear();
	// switching the town if necessary
	std::string new_town = config::town_map_short_names.at(town_index);
	if (MassAgent::carla_client()->GetWorld().GetMap()->GetName() != new_town) {
		std::cout << "switching to " << new_town << std::endl;
		size_t timeout_count = 0;
		try {
			MassAgent::carla_client()->LoadWorld(config::town_map_full_names.at(town_index));
			MassAgent::carla_client()->GetWorld().Tick(5s);
		} catch(carla::client::TimeoutException& e) {
			timeout_count = 1;
			std::cout << "connection to simulator timed out. reconnecting ..." << std::endl;
		}
		// retry up to 10 times
		while (timeout_count > 0 && timeout_count < 10) {
			try {
				MassAgent::carla_client().reset(new cc::Client("127.0.0.1", 2000));
				MassAgent::carla_client()->GetWorld().Tick(5s);
				break;
			} catch(carla::client::TimeoutException& e) {
				std::cout << "retry failed (" << timeout_count++ << ")" << std::endl;
			}
		}
		if (timeout_count == 10) {
			std::cout << "maximum number of retries faield. exting ..." << std::endl;
			return false;
		}
	}
	for (size_t i = 0; i < number_of_agents; ++i) {
		new MassAgent(random_gen, config::GetRestrictedRoads(town_index));
	}
	// setting weather & retrying if disconnected
	std::cout << "setting weather to " << weather << std::endl;
	size_t timeout_count = 0;
	try {
			MassAgent::carla_client()->GetWorld().SetWeather(weather_map.at(weather));
			MassAgent::carla_client()->GetWorld().Tick(5s);
		} catch(carla::client::TimeoutException& e) {
			timeout_count = 1;
			std::cout << "connection to simulator timed out. reconnecting ..." << std::endl;
		}
	while (timeout_count > 0 && timeout_count < 10) {
		try {
			MassAgent::carla_client().reset(new cc::Client("127.0.0.1", 2000));
			MassAgent::carla_client()->GetWorld().Tick(5s);
			break;
		} catch(carla::client::TimeoutException& e) {
			std::cout << "retry failed (" << timeout_count++ << ")" << std::endl;
		}
	}
	std::cout << "town switch complete" << std::endl;
	return true;
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