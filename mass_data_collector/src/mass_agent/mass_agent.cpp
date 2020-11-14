#include "mass_agent/mass_agent.h"
#include "config/agent_config.h"
#include "config/geom_config.h"
#include "mass_agent/sensors.h"

#include <Eigen/src/Core/Matrix.h>
#include <cstdlib>
#include <thread>
#include <tuple>
#include <cmath>
#include <ros/package.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/imgcodecs.hpp>
#include <pcl/io/pcd_io.h>

using namespace std::chrono_literals;
namespace cg = carla::geom;

namespace agent {
std::vector<const MassAgent*>& MassAgent::agents() {
	static std::vector<const MassAgent*> activte_agents;
	return activte_agents;
}

MassAgent::MassAgent() : map_instance_(freicar::map::Map::GetInstance()) {
	// initialize some stuff & things
	carla_client_ = nullptr;
	transform_.setZero();
	id_ = agents().size();
	agents().emplace_back(this);
	std::cout << "created mass-agent-" << id_ << std::endl;
}

/* destructor */
MassAgent::~MassAgent() {
	// releasing resources
	DestroyAgent();
}
/* connects to CARLA */
void MassAgent::ActivateCarlaAgent(const std::string &address, unsigned int port) {
	try {
		carla_client_ = std::make_unique<cc::Client>(address, port);
		carla_client_->SetTimeout(10s);
		std::cout << "client version: " << carla_client_->GetClientVersion() << '\t'
				  << "server version: " << carla_client_->GetServerVersion() << std::endl;
		auto world = carla_client_->GetWorld();
		auto blueprint_library = world.GetBlueprintLibrary();
		auto vehicles = blueprint_library->Filter("freicar_1");
		if (vehicles->empty()) {
			ROS_ERROR("ERROR: did not find car model with name: %s... using default model: freicar_10 \n", "freicar_1");
			vehicles = blueprint_library->Filter("freicar_10");
		}
		carla::client::ActorBlueprint blueprint = (*vehicles)[0];
		transform_.setIdentity();
		auto actor = world.TrySpawnActor(blueprint, cg::Transform(cg::Location(cg::Vector3D(0, 0, 0))));
		if (!actor) {
			std::cout << "failed to spawn " << "freicar_1" << ". exiting..." << std::endl;
			std::exit(EXIT_FAILURE);
		}
		vehicle_ = boost::static_pointer_cast<cc::Vehicle>(actor);
		std::cout << "spawned " << vehicle_->GetDisplayId() << std::endl;
		// turn off physics
		vehicle_->SetSimulatePhysics(false);
	} catch (const cc::TimeoutException &e) {
		ROS_ERROR("ERROR: connection to the CARLA server timed out\n%s\n", e.what());
		std::exit(EXIT_FAILURE);
	} catch (const std::exception &e) {
		ROS_ERROR("ERROR: %s", e.what());
		DestroyAgent();
		std::exit(EXIT_FAILURE);
	}
	SetupSensors();
}
/* places the agent on a random point in the map at a random pose, returns true if successful. [[blocking]] */
bool MassAgent::SetRandomPose() {
	bool admissable = false;
	float yaw = 0;
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	while (!admissable) {
		admissable = true;
		auto lane_point = map_instance_.GetRandomLanePoint();
		std::tie(x, y, z) = lane_point.GetCoords();
		yaw = lane_point.GetHeading();
		for (const auto *agent : agents()) {
			if (agent->id_ == id_) {
				continue;
			}
			auto distance = std::sqrt((agent->x() - x) * (agent->x() - x) +
									  (agent->y() - y) * (agent->y() - y) +
									  (agent->z() - z) * (agent->z() - z));
			if (distance < config::kMinimumAgentDistance) {
				admissable = false;
				break;
			}
		}
	}
	Eigen::Matrix3d rot;
	rot = Eigen::AngleAxisd(0.0 * config::kToRadians * M_PI, Eigen::Vector3d::UnitX()) *
		  Eigen::AngleAxisd(0.0 * config::kToRadians * M_PI, Eigen::Vector3d::UnitY()) *
		  Eigen::AngleAxisd(yaw * config::kToRadians * M_PI, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d trans(x, y, z);
    transform_.block<3, 3>(0, 0) = rot;
    transform_.block<3, 1>(0, 3) = trans;
	// CARLA/Unreal' C.S. is left-handed
	vehicle_->SetTransform(cg::Transform(cg::Location(x, -y, z),
										 cg::Rotation(0, -yaw * config::kToDegrees, 0)));
	do {
		auto transform = vehicle_->GetTransform();
		// if this is not enough, it just means I have a scientifically proven shit luck
		if (std::abs(transform.location.x - x) < 1e-2) { // NOLINT
			break;
		}
	} while(true);
	return true;
}
/* caputres one frame for all sensors [[blocking]] */
void MassAgent::CaptureOnce() {
	front_cam_->CaputreOnce();
	for (auto& semantic_cam : semantic_pc_cams_) {
		semantic_cam->CaputreOnce();
	}
	datapoint_transforms_.emplace_back(transform_);
	std::cout << "captured once" << std::endl;
}
/* destroys the agent in the simulation & its sensors. */
void MassAgent::DestroyAgent() {
	std::cout << "destroying mass-agent-" << id_ << std::endl;
	front_cam_->Destroy();
	for (auto& semantic_depth_cam : semantic_pc_cams_) {
		if (semantic_depth_cam) {
			semantic_depth_cam->Destroy();
		}
	}
	if (vehicle_) {
		vehicle_->Destroy();
		vehicle_ = nullptr;
	}
	if (carla_client_) {
		carla_client_.reset();
	}
}
/* checks the buffers and creates a single data point from all the sensors */
void MassAgent::GenerateDataPoint() {
	// TODO(hosein): find a better way of checking for an empty stack
	if (datapoint_transforms_.empty()) {
		return;
	}
	auto car_transform = datapoint_transforms_.front();
	std::cout << "decoding with car transform \n" << car_transform << std::endl;
	for (auto& semantic_depth_cam : semantic_pc_cams_) {
		auto[success, semantic, depth] = semantic_depth_cam->pop();
		if (success) {
			semantic_cloud_.AddSemanticDepthImage(semantic_depth_cam->geometry(), car_transform, semantic, depth);
			std::cout << semantic_depth_cam->name() << " added to pointcloud" << std::endl;
		} else {
			std::cout << semantic_depth_cam->name() << " unresponsive" << std::endl;
		}
	}
	semantic_cloud_.SaveCloud("/home/hosein/s2cloud.pcl");
	std::cout << "point cloud saved" << std::endl;
	pcl::PointCloud<pcl::PointXYZRGB> filtered_cloud = semantic_cloud_.MaskOutlierPoints(front_cam_->geometry(), car_transform);
	if (filtered_cloud.points.empty()) {
		std::cout << "filtered point cloud is empty!" << std::endl;
	} else {
		pcl::io::savePCDFile("/home/hosein/f2cloud.pcl", filtered_cloud);
		std::cout << "filtered point cloud saved" << std::endl;
	}
	datapoint_transforms_.erase(datapoint_transforms_.begin());
}
/* returns x coordinate */
double MassAgent::x() const {
	return transform_(0, 3);
}
/* returns y coordinate */
double MassAgent::y() const {
	return transform_(1, 3);
}
/* returns z coordinate */
double MassAgent::z() const {
	return transform_(2, 3);
}
/* initializes the structures for the camera, lidar & depth measurements */
void MassAgent::SetupSensors() {
	std::string yaml_path = ros::package::getPath("mass_data_collector");
	yaml_path += "/param/sensors.yaml";
	std::cout << "reading sensor description file: " << yaml_path << std::endl;
	auto bp_library = vehicle_->GetWorld().GetBlueprintLibrary();
	YAML::Node base = YAML::LoadFile(yaml_path);
	YAML::Node front_cam_node = base["camera-front"];
	// front rgb cam
	if (front_cam_node.IsDefined()) {
		front_cam_ = std::make_unique<data::RGBCamera>(front_cam_node, bp_library, vehicle_, true);
	}
	// depth camera
	YAML::Node mass_cam_node = base["camera-mass"];
	if (mass_cam_node.IsDefined()) {
		// static_cast<size_t>(data::CameraPosition::BACK)
		for (size_t i = 0; i <= 1; ++i) {
			semantic_pc_cams_.emplace_back(std::make_unique<data::SemanticPointCloudCamera>(mass_cam_node,
															bp_library,
															vehicle_,
															static_cast<data::CameraPosition>(i),
															true));
		}
	}
}
} // namespace agent