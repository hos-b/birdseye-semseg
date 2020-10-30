#include <cstdio>
#include <memory>
#include <ros/ros.h>
#include <unistd.h>
#include <csignal>
#include "freicar_carla_proxy/freicar_carla_proxy.h"

std::unique_ptr<freicar::agent::FreiCarCarlaProxy> carla_proxy;
void sig_handler(int signo) {
	(void)signo;
	carla_proxy->~FreiCarCarlaProxy();
	ros::shutdown();
}

freicar::agent::FreiCarCarlaProxy::Settings GetAgentParameters() {
    freicar::agent::FreiCarCarlaProxy::Settings settings;
    std::string type_string;

    if (!ros::param::get("~spawn/x", settings.spawn_point.x)) {
        ROS_ERROR("ERROR: could not find parameter: spawn/x");
        std::exit(EXIT_FAILURE);
    }

    if (!ros::param::get("~spawn/y", settings.spawn_point.y)) {
        ROS_ERROR("ERROR: could not find parameter: spawn/y");
        std::exit(EXIT_FAILURE);
    } else {
		// carla y is flipped
		settings.spawn_point.y = -settings.spawn_point.y;
	}

    if (!ros::param::get("~spawn/z", settings.spawn_point.z)) {
        ROS_ERROR("ERROR: could not find parameter: spawn/z");
        std::exit(EXIT_FAILURE);
    }

	if (!ros::param::get("~name", settings.name)) {
		ROS_ERROR("ERROR: could not find parameter: name");
		std::exit(EXIT_FAILURE);
	}
	if (!ros::param::get("~tf_name", settings.tf_name)) {
		ROS_ERROR("ERROR: could not find parameter: tf_name");
		std::exit(EXIT_FAILURE);
	}

	if (!ros::param::get("~type", type_string)) {
		ROS_ERROR("ERROR: could not find parameter: type");
		std::exit(EXIT_FAILURE);
	}
    if (!ros::param::get("~spawn_sensors", settings.spawn_sensors)) {
        ROS_ERROR("ERROR: could not find parameter: spawn_sensors");
        std::exit(EXIT_FAILURE);
    }
	if (!ros::param::get("~thread_sleep_ms", settings.thread_sleep_ms)) {
		ROS_ERROR("ERROR: could not find parameter: thread_sleep_ms");
		std::exit(EXIT_FAILURE);
	}
	if (!ros::param::get("~brake_multiplier", settings.brake_multiplier)) {
		ROS_ERROR("ERROR: could not find parameter: brake_multiplier");
		std::exit(EXIT_FAILURE);
	}
	if (settings.thread_sleep_ms <= 0) {
		ROS_ERROR("ERROR: thread_sleep_ms should be a positive integer, is %d", settings.thread_sleep_ms);
		std::exit(EXIT_FAILURE);		
	}
	if (type_string == "sim") {
        settings.type_code = freicar::agent::type::SIMULATED;
		int steps_per_second;
		if (!ros::param::get("/sim_sps", steps_per_second) || !ros::param::has("/sim_sync_mode") ||
															  !ros::param::has("/sim_headless")) {
			ROS_ERROR("ERROR: could not find simulation parameters");
			std::exit(EXIT_FAILURE);
		}
		// in sim_only, the delay is determined by the sim config
        settings.thread_sleep_ms = (steps_per_second != 0) ? ((1.0 / steps_per_second) * 1000) : settings.thread_sleep_ms;
	}
	else if (type_string == "real")
        settings.type_code = freicar::agent::type::REAL;
	else if (type_string == "mixed") {
        settings.type_code = freicar::agent::type::SIMULATED | freicar::agent::type::REAL;

        if (!ros::param::get("~sync_topic", settings.sync_topic)) {
            std::cout << "Topic-sync inactive..." << std::endl;
        }

        if (settings.sync_topic == "!"){
            settings.sync_topic = "";
        }

        if (!ros::param::get("~height_offset", settings.height_offset)) {
            ROS_ERROR("ERROR: could not find parameter: height_offset");
            std::exit(EXIT_FAILURE);
        }

        if (!ros::param::get("~sim_delay", settings.sim_delay)) {
            ROS_INFO("INFO: could not find parameter: sim_delay, setting to 0.0...");
            settings.sim_delay = 0.0;
        }

    }
	else {
		ROS_ERROR("ERROR: unknown agent type %s. use real, sim or mixed", type_string.c_str());
		std::exit(EXIT_FAILURE);
	}
	// getting host & user name
	char hostname[30], username[30];
	gethostname(hostname, 30);
	getlogin_r(username, 30);
    settings.owner = std::string(username) + "@" + std::string(hostname);
	return settings;
}
int main(int argc, char **argv)
{
	ros::init(argc, argv, "freicar_carla_proxy");
	std::shared_ptr<ros::NodeHandle> node_handle = std::make_shared<ros::NodeHandle>();
	ROS_INFO("starting carla proxy...");
	srand(135);
	// signal handler
	signal(SIGINT, sig_handler);
	signal(SIGKILL, sig_handler);
	// parsing parameters
	freicar::agent::FreiCarCarlaProxy::Settings settings = GetAgentParameters();

	switch (settings.type_code) {
	case freicar::agent::type::SIMULATED:
		carla_proxy = std::make_unique<freicar::agent::FreiCarCarlaProxy>(settings, node_handle);
		carla_proxy->ActivateCarlaAgent("127.0.0.1", 2000, false);
		carla_proxy->SetupCallbacks();
		break;
	case freicar::agent::type::REAL | freicar::agent::type::SIMULATED:
		carla_proxy = std::make_unique<freicar::agent::FreiCarCarlaProxy>(settings, node_handle);
		carla_proxy->ActivateCarlaAgent("127.0.0.1", 2000, false);
		carla_proxy->SetupCallbacks();
		break;
	default:
		std::cout << "invalide agent type code" << settings.type_code << std::endl;
		std::exit(EXIT_SUCCESS);
		break;
	}
	ros::spin();
	return 0;
}