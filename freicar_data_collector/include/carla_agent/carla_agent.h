#ifndef __AISCAR_RL_AGENT__
#define __AISCAR_RL_AGENT__

#include <string>
#include <memory>
#include <thread>

#include <ros/ros.h>
// #include <sensor_msgs/Image.h>
#include "carla_agent/sensor_structs.h"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

////CARLA STUFF
#include <carla/client/Map.h>
#include <carla/client/World.h>
#include <carla/client/Client.h>
#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/TimeoutException.h>
#include <carla/geom/Transform.h>

namespace cc = carla::client;

class CarlaAgent
{
public:
	~CarlaAgent();
	CarlaAgent(const CarlaAgent&) = delete;
	const CarlaAgent& operator=(const CarlaAgent&) = delete;
	CarlaAgent(unsigned long thread_sleep_ms);
	void ActivateCarlaAgent(const std::string &address, unsigned int port);
	void StopAgentThread();
	void SetupCallbacks();
private:
	void Step(unsigned int thread_sleep_ms);
	void SetupSensors();
	void DestroyAgent();
	
	// agent's state
	std::thread *agent_thread_;
	bool running_;
	unsigned long thread_sleep_ms_;
	// carla stuff
	std::unique_ptr<cc::Client> carla_client_;
	boost::shared_ptr<carla::client::Vehicle> vehicle_;
	CarlaCamera front_cam_;
    CarlaSemanticCamera semseg_cam_;
    CarlaDepthCamera depth_cam_;
    bool sync_trigger_rgb;
    bool sync_trigger_depth;
    bool sync_trigger_sem;
};


#endif
