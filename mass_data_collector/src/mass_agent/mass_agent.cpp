#include "mass_agent/mass_agent.h"

#include <carla/Memory.h>
#include <carla/client/Map.h>
#include <carla/client/Waypoint.h>
#include <carla/geom/Transform.h>
#include <carla/client/Junction.h>

#include "carla/road/Lane.h"
#include "hdf5_api/hdf5_dataset.h"
#include "mass_agent/sensors.h"
#include "geometry/semantic_cloud.h"
#include "config/agent_config.h"
#include "config/geom_config.h"

#include <chrono>
#include <mutex>
#include <opencv2/core.hpp>
#include <string>
#include <thread>
#include <tuple>
#include <cmath>
#include <cstdlib>

#include <Eigen/Dense>
#include <ros/package.h>
#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <ros/ros.h>

using namespace std::chrono_literals;

namespace agent {
/* constructor */
MassAgent::MassAgent() {
	// initialize some stuff & things
	transform_.setZero();
	id_ = agents().size();
	agents().emplace_back(this);
	try {
		auto world = carla_client()->GetWorld();
		auto spawn_points = world.GetMap()->GetRecommendedSpawnPoints();
		// initializing kd_tree (if empty)
		InitializeKDTree();

		auto blueprint_library = world.GetBlueprintLibrary();
		blueprint_name_ = GetBlueprintNames()[id_];
		auto vehicles = blueprint_library->Filter(blueprint_name_);
		if (vehicles->empty()) {
			ROS_ERROR("ERROR: did not find car model with name: %s... using vehicle.seat.leon", blueprint_name_.c_str());
			vehicles = blueprint_library->Filter("vehicle.seat.leon");
		}
		carla::client::ActorBlueprint blueprint = (*vehicles)[0];
		auto initial_pos = spawn_points[id_]; //kd_points()[0]->GetTransform();
		transform_.setIdentity();
		Eigen::Matrix3d rot;
		rot = Eigen::AngleAxisd(initial_pos.rotation.roll	*  config::kToRadians * M_PI, Eigen::Vector3d::UnitX()) *
			  Eigen::AngleAxisd(initial_pos.rotation.pitch	* -config::kToRadians * M_PI, Eigen::Vector3d::UnitY()) *
			  Eigen::AngleAxisd(initial_pos.rotation.yaw	*  config::kToRadians * M_PI, Eigen::Vector3d::UnitZ());
    	Eigen::Vector3d trans(initial_pos.location.x, -initial_pos.location.y, initial_pos.location.z);
    	transform_.block<3, 3>(0, 0) = rot;
    	transform_.block<3, 1>(0, 3) = trans;
		auto actor = world.TrySpawnActor(blueprint, initial_pos);
		if (!actor) {
			std::cout << "failed to spawn " << blueprint_name_ << ". exiting..." << std::endl;
			std::exit(EXIT_FAILURE);
		}
		vehicle_ = boost::static_pointer_cast<cc::Vehicle>(actor);
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
	std::cout << "created mass-agent-" << id_ << ": " + blueprint_name_ << std::endl;
	SetupSensors();
	// reading the dimensions of the vehicle
	std::string yaml_path = ros::package::getPath("mass_data_collector");
	yaml_path += "/param/dimensions.yaml";
	YAML::Node base = YAML::LoadFile(yaml_path);
	YAML::Node boundaries = base[blueprint_name_];
	width_ = std::max(boundaries["left"].as<double>(), boundaries["right"].as<double>()) * 2;
	length_ = std::max(boundaries["front"].as<double>(), boundaries["back"].as<double>()) * 2;
	// extra "padding"
	width_ += 0.1;
	length_ += 0.1;
}
/* destructor */
MassAgent::~MassAgent() {
	// releasing resources
	DestroyAgent();
}
/* places the agent on a random point in the map and makes sure it doesn't collide with others [[blocking]] */
boost::shared_ptr<carla::client::Waypoint> MassAgent::SetRandomPose() {
	boost::shared_ptr<carla::client::Waypoint> initial_wp;
	carla::geom::Transform tf;
	bool admissable = false;
	while (!admissable) {
		admissable = true;
		initial_wp = kd_points()[std::rand() % kd_points().size()]; // NOLINT
		tf = initial_wp->GetTransform();
		for (const auto *agent : agents()) {
			if (agent->id_ == id_) {
				continue;
			}
			auto distance = std::sqrt((agent->carla_x() - tf.location.x) *
									  (agent->carla_x() - tf.location.x) +
									  (agent->carla_y() - tf.location.y) *
									  (agent->carla_y() - tf.location.y) +
									  (agent->carla_z() - tf.location.z) *
									  (agent->carla_z() - tf.location.z));
			if (distance < config::kMinimumAgentDistance) {
				admissable = false;
				break;
			}
		}
	}
	Eigen::Matrix3d rot;
	rot = Eigen::AngleAxisd(tf.rotation.roll * config::kToRadians * M_PI, Eigen::Vector3d::UnitX()) *
		  Eigen::AngleAxisd(tf.rotation.pitch * -config::kToRadians * M_PI, Eigen::Vector3d::UnitY()) *
		  Eigen::AngleAxisd(tf.rotation.yaw * config::kToRadians * M_PI, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d trans(tf.location.x, -tf.location.y, tf.location.z);
    transform_.block<3, 3>(0, 0) = rot;
    transform_.block<3, 1>(0, 3) = trans;
	vehicle_->SetTransform(tf);
	// blocking until moved
	do {
		auto transform = vehicle_->GetTransform();
		// if this is not enough, it just means I have a scientifically proven shit luck
		if (std::abs(transform.location.x - tf.location.x) + std::abs(transform.location.y - tf.location.y) < 1e-2) {
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(config::kPollInterval));
	} while(true);
	return initial_wp;
}
/* places the agent around the given waypoint and also makes sure it doesn't collide with others [[blocking]] */
boost::shared_ptr<carla::client::Waypoint> MassAgent::SetRandomPose(boost::shared_ptr<carla::client::Waypoint> initial_wp) {
	// getting candidates around the given waypoints
	std::vector<boost::shared_ptr<carla::client::Waypoint>> candidates;
	auto front_wps = initial_wp->GetNext(length_ + 0.15 + 
						(static_cast<double>(std::rand() % config::kMaxAgentDistance) / 100.0)); // NOLINT
	auto rear_wps = initial_wp->GetPrevious(length_ + 0.15 + 
						(static_cast<double>(std::rand() % config::kMaxAgentDistance) / 100.0)); // NOLINT
	candidates.insert(candidates.end(), front_wps.begin(), front_wps.end());
	candidates.insert(candidates.end(), rear_wps.begin(), rear_wps.end());
	candidates.emplace_back(initial_wp->GetLeft());
	candidates.emplace_back(initial_wp->GetRight());
	boost::shared_ptr<carla::client::Waypoint> next_wp;
	carla::geom::Transform tf;
	bool admissable = false;
	while (!admissable) {
		admissable = true;
		next_wp = candidates[std::rand() % candidates.size()]; // NOLINT
		if (next_wp == nullptr) {
			admissable = false;
			continue;
		}
		tf = next_wp->GetTransform();
		for (const auto *agent : agents()) {
			if (agent->id_ == id_) {
				continue;
			}
			auto distance = std::sqrt((agent->carla_x() - tf.location.x) *
									  (agent->carla_x() - tf.location.x) +
									  (agent->carla_y() - tf.location.y) *
									  (agent->carla_y() - tf.location.y) +
									  (agent->carla_z() - tf.location.z) *
									  (agent->carla_z() - tf.location.z));
			if (distance < config::kMinimumAgentDistance) {
				admissable = false;
				break;
			}
		}
	}
	Eigen::Matrix3d rot;
	rot = Eigen::AngleAxisd(tf.rotation.roll * config::kToRadians * M_PI, Eigen::Vector3d::UnitX()) *
		  Eigen::AngleAxisd(tf.rotation.pitch * -config::kToRadians * M_PI, Eigen::Vector3d::UnitY()) *
		  Eigen::AngleAxisd(tf.rotation.yaw * config::kToRadians * M_PI, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d trans(tf.location.x, -tf.location.y, tf.location.z);
    transform_.block<3, 3>(0, 0) = rot;
    transform_.block<3, 1>(0, 3) = trans;
	vehicle_->SetTransform(tf);
	// blocking until moved
	do {
		auto transform = vehicle_->GetTransform();
		// if this is not enough, it just means I have a scientifically proven shit luck
		if (std::abs(transform.location.x - tf.location.x) + std::abs(transform.location.y - tf.location.y) < 1e-2) {
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(config::kPollInterval));
	} while(true);
	return next_wp;
}
/* caputres one frame for all sensors [[blocking]] */
void MassAgent::CaptureOnce(bool log) {
	front_rgb_->CaputreOnce();
	front_semantic_pc_->CaputreOnce();
	for (auto& semantic_cam : semantic_pc_cams_) {
		semantic_cam->CaputreOnce();
	}
	datapoint_transforms_.emplace_back(transform_);
	if (log) {
		std::cout << "captured once" << std::endl;
	}
}
/* destroys the agent in the simulation & its sensors. */
void MassAgent::DestroyAgent() {
	std::cout << "destroying mass-agent-" << id_ << std::endl;
	front_rgb_->Destroy();
	front_semantic_pc_->Destroy();
	for (auto& semantic_depth_cam : semantic_pc_cams_) {
		if (semantic_depth_cam) {
			semantic_depth_cam->Destroy();
		}
	}
	if (vehicle_) {
		vehicle_->Destroy();
		vehicle_ = nullptr;
	}
}
/* checks the buffers and creates a single data point from all the sensors */
MASSDataType MassAgent::GenerateDataPoint() {
	MASSDataType datapoint{};
	if (datapoint_transforms_.empty()) {
		std::cout << "GenerateDataPoint() called on agent " << id_
				  << " with an empty queue" << std::endl;
		return datapoint;
	}
	// -------------------------- car transform --------------------------
	auto car_transform = datapoint_transforms_.front();
	// ---------------------- creating target cloud ----------------------
	geom::SemanticCloud target_cloud(config::kPointCloudMaxLocalX,
									 config::kPointCloudMaxLocalY,
									 config::kSemanticBEVRows,
									 config::kSemanticBEVCols);
	for (auto& semantic_depth_cam : semantic_pc_cams_) {
		auto[success, semantic, depth] = semantic_depth_cam->pop();
		if (success) {
			target_cloud.AddSemanticDepthImage(semantic_depth_cam->geometry(), semantic, depth);
		} else {
			std::cout << "ERROR: agent " + std::to_string(id_)
					  + "'s " << semantic_depth_cam->name() << " is unresponsive" << std::endl;
			return datapoint;
		}
	}
	target_cloud.BuildKDTree();
	auto semantic_bev = target_cloud.GetSemanticBEV(32);
	// ----------------------- creating mask cloud -----------------------
	geom::SemanticCloud mask_cloud(config::kPointCloudMaxLocalX,
								   config::kPointCloudMaxLocalY,
								   config::kSemanticBEVRows,
								   config::kSemanticBEVCols);
	auto[succ, semantic, depth] = front_semantic_pc_->pop();
	if (!succ) {
		std::cout << "ERROR: agent " + std::to_string(id_) + "'s front pc cam is unresponsive" << std::endl;
		return datapoint;
	}
	mask_cloud.AddSemanticDepthImage(front_semantic_pc_->geometry(), semantic, depth);
	mask_cloud.BuildKDTree();
	cv::Mat bev_mask = mask_cloud.GetBEVMask(0.1);
	// ------------------------ getting rgb image ------------------------
	auto[success, rgb_image] = front_rgb_->pop();
	if (!success) {
		std::cout << "ERROR: agent " + std::to_string(id_) + "'s rgb cam is unresponsive" << std::endl;
		return datapoint;
	}
	if (!rgb_image.isContinuous()) {
		std::cout << "ERROR: rgb image is not a continuous byte array" << std::endl;
		return datapoint;
	}
	if (!semantic_bev.isContinuous()) {
		std::cout << "ERROR: BEV image is not a continuous byte array" << std::endl;
		return datapoint;
	}
	if (!bev_mask.isContinuous()) {
		std::cout << "ERROR: BEV mask is not a continuous byte array" << std::endl;
		return datapoint;
	}
	// filling the datapoint
	datapoint.agent_id = id_;
	for (size_t i = 0; i < statics::front_rgb_byte_count; ++i) {
		datapoint.front_rgb[i] = rgb_image.data[i]; // NOLINT
	}
	for (size_t i = 0; i < statics::top_semseg_byte_count; ++i) {
		datapoint.top_semseg[i] = semantic_bev.data[i]; // NOLINT
	}
	for (size_t i = 0; i < statics::top_semseg_byte_count; ++i) {
		datapoint.top_mask[i] = bev_mask.data[i]; // NOLINT
	}
	for (size_t i = 0; i < statics::transform_length; ++i) {
		datapoint.transform[i] = car_transform.data()[i]; // NOLINT
	}
	// popping car transform
	datapoint_transforms_.erase(datapoint_transforms_.begin());
	return datapoint;
}
/* captures one datapoint and creates a BEV mask containing the vehicle */
cv::Mat MassAgent::CreateVehicleMask() {
	CaptureOnce(false);
	if (datapoint_transforms_.empty()) {
		std::cout << "CreateVehicleMask() called on agent " << id_
				  << " with an empty queue" << std::endl;
	}
	// ---------------------- creating target cloud ----------------------
	geom::SemanticCloud target_cloud(config::kPointCloudMaxLocalX,
									 config::kPointCloudMaxLocalY,
									 config::kSemanticBEVRows,
									 config::kSemanticBEVCols);
	for (auto& semantic_depth_cam : semantic_pc_cams_) {
		auto[success, semantic, depth] = semantic_depth_cam->pop();
		if (success) {
			target_cloud.AddSemanticDepthImage(semantic_depth_cam->geometry(), semantic, depth,
											   {config::kCARLAVehiclesSemanticID});
		} else {
			std::cout << "ERROR: agent " + std::to_string(id_)
					  + "'s " << semantic_depth_cam->name() << " is unresponsive" << std::endl;
		}
	}
	target_cloud.BuildKDTree();
	cv::Mat vehicle_mask = target_cloud.CalculateVehicleMask(width_, length_, 3, 0.05);
	// ----------------------- creating mask cloud -----------------------
	auto[succ, semantic, depth] = front_semantic_pc_->pop();
	if (!succ) {
		std::cout << "ERROR: agent " + std::to_string(id_) + "'s front pc cam is unresponsive" << std::endl;
	}
	// ------------------------ getting rgb image ------------------------
	auto[success, rgb_image] = front_rgb_->pop();
	if (!success) {
		std::cout << "ERROR: agent " + std::to_string(id_) + "'s rgb cam is unresponsive" << std::endl;
	}
	// popping car transform
	datapoint_transforms_.erase(datapoint_transforms_.begin());
	return vehicle_mask;
}
/* writes an image of the map to file */
void MassAgent::WriteMapToFile(const std::string& path) {
	float minx = 0.0f, maxx = 0.0f;
	float miny = 0.0f, maxy = 0.0f;
	// initialize waypoints if not done already
	InitializeKDTree();
	std::cout << "visaulizing " << kd_points().size() << " waypoints" << std::endl;
	for (auto& waypoint : kd_points()) {
		auto transform = waypoint->GetTransform();
		if (transform.location.x < minx) {
			minx = transform.location.x;
		}
		if (transform.location.y < miny) {
			miny = transform.location.y;
		}
		if (transform.location.x > maxx) {
			maxx = transform.location.x;
		}
		if (transform.location.y > maxy) {
			maxy = transform.location.y;
		}
		// junction id = -1 if not leading to junction
		// id is nonesense
		// road id is the same as street name
		// lane id: opendrive
		// section id, mostly zero, rarely 1, barely 2
	}
	size_t rows = static_cast<size_t>(maxy - miny) * 10 + 10;
	size_t cols = static_cast<size_t>(maxx - minx) * 10 + 10;
	cv::Mat map_img = cv::Mat::zeros(rows, cols, CV_8UC3);
	map_img = cv::Scalar(127, 127, 127);
	minx = std::abs(minx);
	miny = std::abs(miny);
	for (auto& waypoint : kd_points()) {
		auto transform = waypoint->GetTransform();
		unsigned int x = static_cast<int>(transform.location.x + minx + 1) * 10;
		unsigned int y = static_cast<int>(transform.location.y + miny + 1) * 10;
		if (waypoint->GetJunctionId() == -1) {
			cv::circle(map_img, cv::Point(x, y), 10, cv::Scalar(255, 0, 0), -1, cv::LineTypes::FILLED, 0);
		} else {
			cv::circle(map_img, cv::Point(x, y), 10, cv::Scalar(0, 255, 0), -1, cv::LineTypes::FILLED, 0);
			carla::SharedPtr<carla::client::Junction> junction = waypoint->GetJunction();
			// drawing 2 waypoints per lane in junctions
			auto wp_pairs = junction->GetWaypoints();
			for (auto& wp_pair : wp_pairs) {
				x = static_cast<int>(wp_pair.first->GetTransform().location.x + minx + 1) * 10;
				y = static_cast<int>(wp_pair.first->GetTransform().location.y + miny + 1) * 10;
				cv::circle(map_img, cv::Point(x, y), 10, cv::Scalar(0, 255, 255), -1, cv::LineTypes::FILLED, 0);
				x = static_cast<int>(wp_pair.second->GetTransform().location.x + minx + 1) * 10;
				y = static_cast<int>(wp_pair.second->GetTransform().location.y + miny + 1) * 10;
				cv::circle(map_img, cv::Point(x, y), 10, cv::Scalar(0, 255, 255), -1, cv::LineTypes::FILLED, 0);
			}
		}
	}
	cv::imwrite(path, map_img);
	std::cout << "wrote image to " << path << std::endl;
}
/* returns vehicle blueprint names */
std::vector<std::string> MassAgent::GetBlueprintNames() {
	static const std::vector<std::string> vehicle_names = {
		"vehicle.audi.a2",
		"vehicle.audi.tt",
		"vehicle.nissan.micra",
		"vehicle.citroen.c3",
		"vehicle.mercedes-benz.coupe",
		"vehicle.mini.cooperst",
		"vehicle.nissan.patrol",
		"vehicle.mustang.mustang",
		"vehicle.lincoln.mkz2017",
		"vehicle.toyota.prius",
		"vehicle.bmw.grandtourer",
		"vehicle.tesla.model3",
		"vehicle.dodge_charger.police",
		"vehicle.jeep.wrangler_rubicon",
		"vehicle.chevrolet.impala",
		"vehicle.audi.etron",
		"vehicle.seat.leon",
	};
	return vehicle_names; // NOLINT
}
/* initializes waypoints and builds a kd index on top of them */
void MassAgent::InitializeKDTree() {
	if (kd_points().empty()) {
		auto waypoints = carla_client()->GetWorld().GetMap()->GenerateWaypoints(3);
		kd_points().insert(kd_points().begin(), waypoints.begin(), waypoints.end());
	}
	if (kd_tree_ == nullptr) {
		kd_tree_ = std::make_unique<WaypointKDTree>(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10));
		kd_tree_->buildIndex();
	}
}
/* returns carla's x coordinate */
double MassAgent::carla_x() const {
	return transform_(0, 3);
}
/* returns carla's y coordinate */
double MassAgent::carla_y() const {
	return -transform_(1, 3);
}
/* returns carla's z coordinate */
double MassAgent::carla_z() const {
	return transform_(2, 3);
}
/* initializes the structures for the camera, lidar & depth measurements */
void MassAgent::SetupSensors() {
	std::string yaml_path = ros::package::getPath("mass_data_collector");
	yaml_path += "/param/sensors.yaml";
	auto bp_library = vehicle_->GetWorld().GetBlueprintLibrary();
	YAML::Node base = YAML::LoadFile(yaml_path);
	YAML::Node front_rgb_node = base["camera-front"];
	// front rgb & semantic pc cam
	if (front_rgb_node.IsDefined()) {
		front_rgb_ = std::make_unique<data::RGBCamera>(front_rgb_node, bp_library, vehicle_, false);
		front_semantic_pc_ = std::make_unique<data::SemanticPointCloudCamera>(front_rgb_node,
															bp_library,
															vehicle_,
															data::CameraPosition::CENTER,
															false);
	} else {
		std::cout << "ERROR: front camera node is not defined in sensor definition file" << std::endl;
	}
	// depth camera
	YAML::Node mass_cam_node = base["camera-mass"];
	if (mass_cam_node.IsDefined()) {
		for (size_t i = 0; i <= static_cast<size_t>(data::CameraPosition::REARRIGHT); ++i) {
			semantic_pc_cams_.emplace_back(std::make_unique<data::SemanticPointCloudCamera>(mass_cam_node,
															bp_library,
															vehicle_,
															static_cast<data::CameraPosition>(i),
															false));
		}
	} else {
		std::cout << "ERROR: mass camera node is not defined in sensor definition file" << std::endl;
	}
}
std::vector<const MassAgent*>& MassAgent::agents() {
	static std::vector<const MassAgent*> activte_agents;
	return activte_agents;
}
std::vector<boost::shared_ptr<carla::client::Waypoint>>& MassAgent::kd_points() {
	static std::vector<boost::shared_ptr<carla::client::Waypoint>> kd_points;
	return kd_points;
}
std::unique_ptr<cc::Client>& MassAgent::carla_client() {
	static std::unique_ptr<cc::Client> carla_client = std::make_unique<cc::Client>("127.0.0.1", 2000);
	static std::once_flag flag;
	std::call_once(flag, [&]() {
		carla_client->SetTimeout(2s);
		std::cout << "client version: " << carla_client->GetClientVersion() << "\t"
				  << "server version: " << carla_client->GetServerVersion() << std::endl;
	});
	return carla_client;
}
} // namespace agent