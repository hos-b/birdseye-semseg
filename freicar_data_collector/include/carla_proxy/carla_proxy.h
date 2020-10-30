#ifndef __AISCAR_RL_AGENT__
#define __AISCAR_RL_AGENT__

#include <string>
#include <memory>
#include <thread>

#include <ros/ros.h>
#include <std_msgs/String.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/Image.h>

#include "freicar_common/FreiCarHalt.h"

#include "freicar_common/shared/halt_type.h"
#include "freicar_carla_proxy/sensor_structs.h"
#include "raiscar_msgs/ControlCommand.h"
#include <tf2_ros/transform_listener.h>
#include <tf2/transform_datatypes.h>
#include <tf2/transform_storage.h>
#include <tf2/buffer_core.h>
#include <tf2/convert.h>
#include <tf2/utils.h>
#include <geometry_msgs/TransformStamped.h>
//#include "freicar_carla_proxy/json.h"


#include <Eigen/Dense>
////CARLA STUFF
#include <carla/client/Map.h>
#include <carla/client/World.h>
#include <carla/client/Client.h>
#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/TimeoutException.h>
#include <carla/geom/Transform.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//using json = nlohmann::json;

namespace freicar
{
namespace agent
{
namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;
namespace type 
{
	static const unsigned int REAL = 0b01;
	static const unsigned int SIMULATED = 0b10;
}



class FreiCarCarlaProxy
{
public:
	struct Settings {
		unsigned int type_code;
		int thread_sleep_ms;
		std::string name;
		std::string owner;
		std::string tf_name;
		std::string sync_topic;
		float height_offset;
		bool spawn_sensors;
		float sim_delay;
		cg::Vector3D spawn_point;
		float brake_multiplier;
	};
	~FreiCarCarlaProxy();
	FreiCarCarlaProxy(const FreiCarCarlaProxy&) = delete;
	const FreiCarCarlaProxy& operator=(const FreiCarCarlaProxy&) = delete;
	FreiCarCarlaProxy(FreiCarCarlaProxy::Settings settings, std::shared_ptr<ros::NodeHandle> nh);
	void ActivateCarlaAgent(const std::string &address, unsigned int port, bool spectate);
	void StopAgentThread();
	void SetupCallbacks();
private:
	void ControlCallback(const raiscar_msgs::ControlCommand::ConstPtr& control_command);
	void Step(unsigned int thread_sleep_ms);
	void HaltCallback(const freicar_common::FreiCarHalt::ConstPtr& msg);
	void ResumeCallback(const std_msgs::String::ConstPtr& msg);
    void SyncCallback(const sensor_msgs::CameraInfo::ConstPtr &msg);
	void SetupSensors();
	void UpdateSimSpectator(cg::Transform target_transform);
	void DestroyCarlaAgent();
	
	// agent's identity
    ros::Time last_status_publish_;
	// agent's state
	bool spectate_;
	bool suspended_;
    Settings settings_;
	// publishers & subscribers
	std::shared_ptr<ros::NodeHandle> node_handle_;
	std::unique_ptr<image_transport::ImageTransport> image_transport_;
	ros::Subscriber control_sub_;
	ros::Subscriber halt_sub_;
	ros::Subscriber resume_sub_;
	ros::Subscriber sync_sub_;
	ros::Publisher odometry_pub_;
	// thread
	std::thread *agent_thread_;
	bool running_;
	// carla stuff
	std::unique_ptr<cc::Client> carla_client_;
	boost::shared_ptr<carla::client::Vehicle> vehicle_;
	CarlaCamera front_cam_;
    CarlaSemanticCamera semseg_cam_;
    CarlaDepthCamera depth_cam_;
	CarlaLidar lidar_;
    tf2_ros::Buffer tfBuffer_;
    std::unique_ptr<tf2_ros::TransformListener> tfListener_;
    ros::Time send_sync_stamp_;
    bool sync_trigger_rgb;
    bool sync_trigger_depth;
    bool sync_trigger_sem;
    Eigen::Matrix<float, 3, 3> readIntrinsics(const std::string calibration_path);
    cg::Transform LookupCameraExtrinsics(std::string from, std::string to);
    cg::Transform LookupLidarExtrinsics(std::string from, std::string to);
};

namespace{
	/* getting rid of ccls warnings */
	inline void TypePlaceHolder() {
		(void)type::REAL;
		(void)type::SIMULATED;
	}
}

} // namespace agent
} // namespace freicar

#endif
