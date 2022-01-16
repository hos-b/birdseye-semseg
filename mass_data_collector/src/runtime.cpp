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
bool SwitchTown(size_t, const std::string&, std::mt19937&);
std::string SecondsToString(uint32);
bool EnableSyncMode();

int main(int argc, char **argv)
{
	// ros init, signal handling --------------------------------------------------------------------------------
	ros::init(argc, argv, "mass_data_collector");
	std::shared_ptr<ros::NodeHandle> node_handle = std::make_shared<ros::NodeHandle>();
	signal(SIGINT, SIGINT_handler);
	// reading configs ------------------------------------------------------------------------------------------
	auto& config = RuntimeConfig::GetConfig();
	AssertConfiguration();
	CollectionStats stats(config.agent_count);
	srand(config.random_seed);
	std::cout << "simulating with " << config.agent_count << " for "
              << config.batch_count << " iterations in town " << std::to_string(config.town) << "\n";
	// dataset --------------------------------------------------------------------------------------------------
	unsigned int mode = mode::FILE_TRUNC | mode::DSET_CREAT;
	HDF5Dataset dataset(config.dataset_path, config.dataset_name,
						mode, compression::NONE, 1,
					   (config.max_batch_count + 1) * config.agent_count,
						config.hdf5_chunk_size);
	// timing & collection stuff --------------------------------------------------------------------------------
	float avg_batch_time = 0.0f;
	bool batch_finished = false;
	char state = 'i';
	size_t batch_count = 0;
	unsigned int batch_size = config.agent_count;
	unsigned int agents_done = 0;
	std::thread time_thread(WatchdogThreadCallback, &batch_count, &state, &agents_done,
							&batch_size, &avg_batch_time, &batch_finished);
	// random distribution & shuffling necessities --------------------------------------------------------------
	std::mt19937 random_gen(config.random_seed);
	// CARLA stuff ----------------------------------------------------------------------------------------------
	auto& agents = MassAgent::agents();
    // go to designated town
    bool success = SwitchTown(config.town, config.weather, random_gen);
	// data collection loop -------------------------------------------------------------------------------------
	while (ros::ok()) {
		if (batch_count == config.batch_count) {
			break;
		}
		state = 'i';
		// timing
		stats.AddNewBatch(batch_size);
		auto batch_start_t = std::chrono::high_resolution_clock::now();
		// // tick to move the agents
		state = 'm';
        agents_done = 0;
		for (size_t i = 0; i < batch_size; ++i) {
			agents[i]->MoveForward(config.meters_per_capture);
			agents_done += 1;
		}
		// Tick() the simulator for positioning
		try {
			MassAgent::carla_client()->GetWorld().Tick(5s);
		} catch (carla::client::TimeoutException& ex) {
			std::cout << "connection to simulator timed out while Tick()ing" << std::endl;
			break;
		}
		// capture frames
		state = 'c';
		agents_done = 0;
		for (size_t i = 0; i < batch_size; ++i) {
			agents[i]->CaptureOnce();
			agents_done += 1;
		}
		state = 't';
		// Tick() the simulator for data. for some reason once doesn't suffice
		try {
			MassAgent::carla_client()->GetWorld().Tick(5s);
			MassAgent::carla_client()->GetWorld().Tick(5s);
		} catch (carla::client::TimeoutException& ex) {
			std::cout << "connection to simulator timed out while Tick()ing" << std::endl;
			break;
		}
		// gather data (async)
		state = 'g';
		std::future<MASSDataType> promise[batch_size];
		agents_done = 0;
		size_t agent_batch_index = 0;
		for (unsigned int i = 0; i < batch_size; ++i) {
			promise[i] = std::async(&MassAgent::GenerateDataPoint,
									agents[i], agent_batch_index++);
			agents_done += 1;
		}
		state = 's'; // saving
		agents_done = 0;
		for (unsigned int i = 0; i < batch_size; ++i) {
			MASSDataType datapoint = promise[i].get();
			dataset.AppendElement(&datapoint);
			agents_done += 1;
		}
		// timing stuff
		batch_count += 1;
		auto batch_end_t = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> batch_duration = batch_end_t - batch_start_t;
		avg_batch_time = avg_batch_time + (1.0f / static_cast<float>(batch_count)) *
										  (batch_duration.count() - avg_batch_time);
		batch_finished = true;
	}
    // add metadata to dataset file
	auto batch_histogram = stats.AsVector();
	std::cout << "\nbatch size histogram:\n";
	stats.Print();
    dataset.AddU32Attribute(&batch_histogram[0], batch_histogram.size(), "batch_histogram");
    uint32_t data[] = {config.agent_count};
    dataset.AddU32Attribute(data, 1, "max_agent_count");
    data[0] = 1;
    dataset.AddU32Attribute(data, 1, "min_agent_count");
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
	size_t max_batch_count = RuntimeConfig::GetConfig().batch_count;
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
    if (!failed && !EnableSyncMode()) {
        failed = true;
        std::cout << "could not enable synchronized mode" << std::endl;
    }
    if (failed) {
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
	auto& col_conf = RuntimeConfig::GetConfig();
	auto& agents = MassAgent::agents();
	auto number_of_agents = col_conf.agent_count;
	std::cout << std::endl;
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
        auto [x, y, z] = col_conf.starting_points[i];
		new MassAgent(random_gen, x, y, z);
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