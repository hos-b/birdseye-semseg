#include "mass_agent/mass_agent.h"
#include "config/agent_config.h"
#include "config/geom_config.h"

#include <cstdlib>
#include <thread>
#include <tuple>
#include <cmath>
#include <ros/package.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/imgcodecs.hpp>

#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/io/FileFormatIO.h>
#include <open3d/Open3D.h>

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
	x_ = y_ = z_ = 0;
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
			ROS_ERROR("ERROR: Did not find car model with name: %s , ... using default model: freicar_10 \n", "freicar_1");
			vehicles = blueprint_library->Filter("freicar_10");
		}
		carla::client::ActorBlueprint blueprint = (*vehicles)[0];
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
/* places the agent on a random point in the map at a random pose */
void MassAgent::SetRandomPose() {
	bool admissable = false;
	float yaw = 0;
	while (!admissable) {
		admissable = true;
		auto lane_point = map_instance_.GetRandomLanePoint();
		std::tie(x_, y_, z_) = lane_point.GetCoords();
		yaw = lane_point.GetHeading();
		for (const auto *agent : agents()) {
			if (agent->id_ == id_) {
				continue;
			}
			auto distance = std::sqrt((agent->x_ - x_) * (agent->x_ - x_) +
									  (agent->y_ - y_) * (agent->y_ - y_) +
									  (agent->z_ - z_) * (agent->z_ - z_));
			if (distance < config::kMinimumAgentDistance) {
				admissable = false;
				break;
			}
		}
	}
	// CARLA/Unreal' C.S. is left-handed
	vehicle_->SetTransform(cg::Transform(cg::Location(x_, -y_, z_),
										 cg::Rotation(0, -yaw * config::kToDegrees, 0)));
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
		front_cam_->CaputreOnce();
	}
	// depth camera
	YAML::Node depth_cam_node = base["camera-depth"];
	YAML::Node semseg_cam_node = base["camera-semseg"];
	if (depth_cam_node.IsDefined() && semseg_cam_node.IsDefined()) {
		semantic_pc_cam_center_ = std::make_unique<data::SemanticPointCloudCamera>(depth_cam_node,
																 semseg_cam_node,
																 bp_library,
																 vehicle_, true);
		semantic_pc_cam_center_->CaputreOnce();
	}
}
/* destroys the agent in the simulation & its sensors. */
void MassAgent::DestroyAgent() {
	std::cout << "destroying mass-agent-" << id_ << std::endl;
	// TODO(hosein): fix
	front_cam_->Destroy();
	semantic_pc_cam_center_->Destroy();
	if (vehicle_) {
		vehicle_->Destroy();
		vehicle_ = nullptr;
	}
	if (carla_client_) {
		carla_client_.reset();
	}
}

/* returns the current position of the agent */
std::tuple<float, float, float> MassAgent::GetPostion() const {
	return std::make_tuple(x_, y_, z_);
}

} // namespace agent