#include "mass_agent/mass_agent.h"

#include <carla/client/Junction.h>
#include <carla/road/Lane.h>

#include "config/agent_config.h"
#include "config/geom_config.h"

#include <cassert>
#include <cstdlib>

#include <ros/ros.h>
#include <Eigen/Dense>
#include <ros/package.h>
#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>

using namespace std::chrono_literals;

#define __RELEASE

namespace agent {

/* constructor */
MassAgent::MassAgent() {
	// initialize some stuff & things
	transform_.setIdentity();
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
		auto initial_pos = spawn_points[id_];
		Eigen::Matrix3d rot;
		rot = Eigen::AngleAxisd(-initial_pos.rotation.roll  * config::kToRadians, Eigen::Vector3d::UnitX()) *
			  Eigen::AngleAxisd(-initial_pos.rotation.pitch * config::kToRadians, Eigen::Vector3d::UnitY()) *
			  Eigen::AngleAxisd(-initial_pos.rotation.yaw   * config::kToRadians, Eigen::Vector3d::UnitZ());
    	Eigen::Vector3d trans(initial_pos.location.x, initial_pos.location.y, initial_pos.location.z);
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
	// reading the dimensions of the vehicle
	std::string yaml_path = ros::package::getPath("mass_data_collector");
	yaml_path += "/param/vehicle_dimensions.yaml";
	YAML::Node base = YAML::LoadFile(yaml_path);
	YAML::Node boundaries = base[blueprint_name_];
	vehicle_width_ = std::max(boundaries["left"].as<double>(), boundaries["right"].as<double>()) * 2;
	vehicle_length_ = std::max(boundaries["front"].as<double>(), boundaries["back"].as<double>()) * 2;
	SetupSensors(boundaries["camera-shift"].as<float>());
	std::cout << "created mass-agent-" << id_ << ": " + blueprint_name_ << std::endl;
}

/* destructor */
MassAgent::~MassAgent() {
	// releasing resources
	DestroyAgent();
}

/* places the agent on a random point in the map and makes sure it doesn't collide with others [[blocking]] 
   this function should only be called for the first car of the batch, because it does not check for collision
   with other cars.
*/
boost::shared_ptr<carla::client::Waypoint> MassAgent::SetRandomPose(const std::unordered_map<int, bool>& restricted_roads) {
	boost::shared_ptr<carla::client::Waypoint> initial_wp;
	carla::geom::Transform tf;
	while (true) {
		initial_wp = kd_points_[std::rand() % kd_points_.size()];
		// if not in a restricted area
		if (restricted_roads.find(initial_wp->GetRoadId()) == restricted_roads.end()) {
			tf = initial_wp->GetTransform();
			break;
		}
	}
	Eigen::Matrix3d rot;
	rot = Eigen::AngleAxisd(-tf.rotation.roll  * config::kToRadians, Eigen::Vector3d::UnitX()) *
		  Eigen::AngleAxisd(-tf.rotation.pitch * config::kToRadians, Eigen::Vector3d::UnitY()) *
		  Eigen::AngleAxisd(-tf.rotation.yaw   * config::kToRadians, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d trans(tf.location.x, -tf.location.y, tf.location.z);
    transform_.block<3, 3>(0, 0) = rot;
    transform_.block<3, 1>(0, 3) = trans;
	vehicle_->SetTransform(tf);
	// blocking until moved
	do {
		auto current_tf = vehicle_->GetTransform();
		// if this is not enough, it just means I have a scientifically proven shit luck
		if (std::abs(current_tf.location.x - tf.location.x) +
			std::abs(current_tf.location.y - tf.location.y) < 1e-2) {
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(config::kPollInterval));
	} while(true);
	return initial_wp;
}

/* expands the given waypoint by finding the points in its vicinity */
static std::vector<boost::shared_ptr<carla::client::Waypoint>>
ExpandWayoint(boost::shared_ptr<carla::client::Waypoint> wp, double min_dist) {
	std::vector<boost::shared_ptr<carla::client::Waypoint>> candidates;
	auto front_wps = wp->GetNext(min_dist +
						(static_cast<double>(std::rand() % config::kMaxAgentDistance) / 100.0));
	auto rear_wps = wp->GetPrevious(min_dist +
						(static_cast<double>(std::rand() % config::kMaxAgentDistance) / 100.0));
	candidates.insert(candidates.end(), front_wps.begin(), front_wps.end());
	candidates.insert(candidates.end(), rear_wps.begin(), rear_wps.end());
	candidates.emplace_back(wp->GetLeft());
	candidates.emplace_back(wp->GetRight());
	return candidates;
}

/* places the agent around the given waypoint and also makes sure it doesn't collide with others [[blocking]] 
   the indices of the other agents must be given and only up until `max_index will be considered */
boost::shared_ptr<carla::client::Waypoint>
MassAgent::SetRandomPose(boost::shared_ptr<carla::client::Waypoint> initial_wp,
						 size_t knn_pts, const MassAgent* agents, const bool* deadlock,
				  		 std::vector<unsigned int> indices, unsigned int max_index,
						 const std::unordered_map<int, bool>& restricted_roads) {
	// last agent in queue has hit a deadlock, forwarding nullptr
	if (initial_wp == nullptr) {
		return nullptr;
	}
	// getting candidates around the given waypoints
	std::vector<boost::shared_ptr<carla::client::Waypoint>> candidates;
	auto initial_expansion = ExpandWayoint(initial_wp, vehicle_length_ * config::kMinDistCoeff);
	candidates.insert(candidates.end(), initial_expansion.begin(), initial_expansion.end());
	if (knn_pts > 0) {
		auto query_tfm = initial_wp->GetTransform();
		double query[3] = {query_tfm.location.x, query_tfm.location.y, query_tfm.location.y};
		std::vector<size_t> knn_ret_indices(knn_pts);
		std::vector<double> knn_sqrd_dist(knn_pts);
		knn_pts = kd_tree_->knnSearch(&query[0], knn_pts, &knn_ret_indices[0], &knn_sqrd_dist[0]);
		knn_ret_indices.resize(knn_pts);
		for (auto close_wp_idx : knn_ret_indices) {
			auto expansion = ExpandWayoint(kd_points_[close_wp_idx], vehicle_length_ * config::kMinDistCoeff);
			candidates.insert(candidates.end(), expansion.begin(), expansion.end());
		}
	}
	boost::shared_ptr<carla::client::Waypoint> next_wp;
	carla::geom::Transform target_tf;
	bool admissable = false;
	while (!admissable) {
		// it is possible to hit a deadlock when the batch size is high
		// and the randomed waypoints don't suffice to fit them all in.
		if (*deadlock) {
			return nullptr;
		}
		admissable = true;
		next_wp = candidates[std::rand() % candidates.size()];
		// nullptr happens if unavailable
		if (next_wp == nullptr) {
			admissable = false;
			continue;
		}
		// if the randomed wp is not drivable (parking spaces, shoulder lanes, etc)
		if ((static_cast<uint32_t>(next_wp->GetType()) &
			 static_cast<uint32_t>(carla::road::Lane::LaneType::Driving)) == 0u) {
			admissable = false;
			continue;
		}
		// if it's in the restricted area
		if (restricted_roads.find(next_wp->GetRoadId()) != restricted_roads.end()) {
			admissable = false;
			continue;
		}
		target_tf = next_wp->GetTransform();
		for (unsigned int i = 0; i < max_index; ++i) {
			const MassAgent* agent = &agents[indices[i]];
			// technically impossible to hit this if
			if (agent->id_ == id_) {
				continue;
			}
			auto distance = std::sqrt((agent->carla_x() - target_tf.location.x) *
									  (agent->carla_x() - target_tf.location.x) +
									  (agent->carla_y() - target_tf.location.y) *
									  (agent->carla_y() - target_tf.location.y) +
									  (agent->carla_z() - target_tf.location.z) *
									  (agent->carla_z() - target_tf.location.z));
			if (distance < vehicle_length_ * config::kMinDistCoeff) {
				admissable = false;
				continue;
			}
		}
	}
	Eigen::Matrix3d rot;
	rot = Eigen::AngleAxisd(-target_tf.rotation.roll  * config::kToRadians, Eigen::Vector3d::UnitX()) *
		  Eigen::AngleAxisd(-target_tf.rotation.pitch * config::kToRadians, Eigen::Vector3d::UnitY()) *
		  Eigen::AngleAxisd(-target_tf.rotation.yaw   * config::kToRadians, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d trans(target_tf.location.x, -target_tf.location.y, target_tf.location.z);
    transform_.block<3, 3>(0, 0) = rot;
    transform_.block<3, 1>(0, 3) = trans;
	vehicle_->SetTransform(target_tf);
	// blocking until moved
	do {
		auto current_tf = vehicle_->GetTransform();
		// if this is not enough, it just means I have a scientifically proven shit luck
		if (std::abs(current_tf.location.x - target_tf.location.x) +
			std::abs(current_tf.location.y - target_tf.location.y) < 1e-2) {
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(config::kPollInterval));
	} while(true);
	return next_wp;
}

/* caputres one frame for all sensors [[blocking]] */
void MassAgent::CaptureOnce() {
	front_rgb_->CaputreOnce();
	front_mask_pc_->CaputreOnce();
	for (auto& semantic_cam : semantic_pc_cams_) {
		semantic_cam->CaputreOnce();
	}
}

/* destroys the agent in the simulation & its sensors. */
void MassAgent::DestroyAgent() {
	std::cout << "destroying mass-agent-" << id_ << std::endl;
	front_rgb_->Destroy();
	front_mask_pc_->Destroy();
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
template <>
MASSDataType MassAgent::GenerateDataPoint
		<geom::CloudBackend::KD_TREE>(unsigned int agent_batch_index) {
	CaptureOnce();
	MASSDataType datapoint{};
	// ----------------------------------------- creating mask cloud -----------------------------------------
	geom::SemanticCloud<geom::CloudBackend::KD_TREE> mask_cloud(sc_settings());
	auto[succ, front_semantic, front_depth] = front_mask_pc_->pop();
#ifndef __RELEASE
	if (!succ) {
		std::cout << "ERROR: agent " + std::to_string(id_) + "'s front pc cam is unresponsive" << std::endl;
		return datapoint;
	}
#endif
	mask_cloud.AddSemanticDepthImage(front_mask_pc_->geometry(), front_semantic, front_depth);
	mask_cloud.BuildKDTree();
	cv::Mat fov_mask = mask_cloud.GetFOVMask();
	// ---------------------------------------- creating target cloud ----------------------------------------
	geom::SemanticCloud<geom::CloudBackend::KD_TREE> target_cloud(sc_settings());
	for (auto& semantic_depth_cam : semantic_pc_cams_) {
		auto[success, semantic, depth] = semantic_depth_cam->pop();
#ifndef __RELEASE
		if (!success) {
			std::cout << "ERROR: agent " + std::to_string(id_)
					  + "'s " << semantic_depth_cam->name() << " is unresponsive" << std::endl;
			return datapoint;
		}
#endif
		target_cloud.AddSemanticDepthImage(semantic_depth_cam->geometry(), semantic, depth);
	}
	target_cloud.BuildKDTree();
	auto[semantic_bev, vehicle_mask] = target_cloud.GetBEVData(vehicle_width_, vehicle_length_);
	// ------------------------------------------ getting rgb image ------------------------------------------
	auto[success, rgb_image] = front_rgb_->pop();
#ifndef __RELEASE
	if (!success) {
		std::cout << "ERROR: agent " + std::to_string(id_) + "'s rgb cam is unresponsive" << std::endl;
		return datapoint;
	}
#endif
	// filling the datapoint
	datapoint.agent_id = agent_batch_index;
	// omp for does not improve performance, may even degrade
	for (size_t i = 0; i < statics::front_rgb_byte_count; ++i) {
		datapoint.front_rgb[i] = rgb_image.data[i];
	}
	for (size_t i = 0; i < statics::top_semseg_byte_count; ++i) {
		datapoint.top_semseg[i] = semantic_bev.data[i];
	}
	for (size_t i = 0; i < statics::top_semseg_byte_count; ++i) {
		datapoint.top_mask[i] = fov_mask.data[i] | vehicle_mask.data[i];
	}
	for (size_t i = 0; i < statics::transform_length; ++i) {
		datapoint.transform[i] = transform_.data()[i];
	}
	return datapoint;
}

/* checks the buffers and creates a single data point from all the sensors */
template<>
MASSDataType MassAgent::GenerateDataPoint
		<geom::CloudBackend::SURFACE_MAP>(unsigned int agent_batch_index) {
	CaptureOnce();
	MASSDataType datapoint{};
	// ----------------------------------------- creating mask cloud -----------------------------------------
	geom::SemanticCloud<geom::CloudBackend::SURFACE_MAP> mask_cloud(sc_settings());
	auto[succ, front_semantic, front_depth] = front_mask_pc_->pop();
#ifndef __RELEASE
	if (!succ) {
		std::cout << "ERROR: agent " + std::to_string(id_) + "'s front pc cam is unresponsive" << std::endl;
		return datapoint;
	}
#endif
	mask_cloud.AddSemanticDepthImage(front_mask_pc_->geometry(), front_semantic, front_depth);
	cv::Mat fov_mask = mask_cloud.GetFOVMask(1);
	// std::cout << "max mask size: " << mask_cloud.find_max_container_size() << std::endl;
	// ---------------------------------------- creating target cloud ----------------------------------------
	geom::SemanticCloud<geom::CloudBackend::SURFACE_MAP> target_cloud(sc_settings());
	for (auto& semantic_depth_cam : semantic_pc_cams_) {
		auto[success, semantic, depth] = semantic_depth_cam->pop();
#ifndef __RELEASE
		if (!success) {
			std::cout << "ERROR: agent " + std::to_string(id_)
					  + "'s " << semantic_depth_cam->name() << " is unresponsive" << std::endl;
			return datapoint;
		}
#endif
		target_cloud.AddSemanticDepthImage(semantic_depth_cam->geometry(), semantic, depth);
	}
	// std::cout << "max semantic size: " << target_cloud.find_max_container_size() << std::endl;
	auto[semantic_bev, vehicle_mask] =
		target_cloud.GetBEVData<geom::AggregationStrategy::HIGHEST_Z>(vehicle_width_, vehicle_length_);
	// ------------------------------------------ getting rgb image ------------------------------------------
	auto[success, rgb_image] = front_rgb_->pop();
#ifndef __RELEASE
	if (!success) {
		std::cout << "ERROR: agent " + std::to_string(id_) + "'s rgb cam is unresponsive" << std::endl;
		return datapoint;
	}
#endif
	// filling the datapoint
	datapoint.agent_id = agent_batch_index;
	// omp for does not improve performance, may even degrade
	for (size_t i = 0; i < statics::front_rgb_byte_count; ++i) {
		datapoint.front_rgb[i] = rgb_image.data[i];
	}
	for (size_t i = 0; i < statics::top_semseg_byte_count; ++i) {
		datapoint.top_semseg[i] = semantic_bev.data[i];
	}
	for (size_t i = 0; i < statics::top_semseg_byte_count; ++i) {
		datapoint.top_mask[i] = fov_mask.data[i] | vehicle_mask.data[i];
	}
	for (size_t i = 0; i < statics::transform_length; ++i) {
		datapoint.transform[i] = transform_.data()[i];
	}
	return datapoint;
}

/* checks the buffers and creates a single data point from all the sensors */
template<>
MASSDataType MassAgent::GenerateDataPoint
		<geom::CloudBackend::MIXED>(unsigned int agent_batch_index) {
	CaptureOnce();
	MASSDataType datapoint{};
	// ----------------------------------------- creating mask cloud -----------------------------------------
	geom::SemanticCloud<geom::CloudBackend::MIXED> mixed_cloud(sc_settings());
	auto[succ, front_semantic, front_depth] = front_mask_pc_->pop();
#ifndef __RELEASE
	if (!succ) {
		std::cout << "ERROR: agent " + std::to_string(id_) + "'s front pc cam is unresponsive" << std::endl;
		return datapoint;
	}
#endif
	mixed_cloud.AddSemanticDepthImage<geom::DestinationMap::MASK>
		(front_mask_pc_->geometry(), front_semantic, front_depth);
	// ---------------------------------------- creating target cloud ----------------------------------------
	for (auto& semantic_depth_cam : semantic_pc_cams_) {
		auto[success, semantic, depth] = semantic_depth_cam->pop();
#ifndef __RELEASE
		if (!success) {
			std::cout << "ERROR: agent " + std::to_string(id_)
					  + "'s " << semantic_depth_cam->name() << " is unresponsive" << std::endl;
			return datapoint;
		}
#endif
		mixed_cloud.AddSemanticDepthImage<geom::DestinationMap::SEMANTIC>
			(semantic_depth_cam->geometry(), semantic, depth);
	}
	auto[semantic_bev, mask] =
		mixed_cloud.GetBEVData<geom::AggregationStrategy::WEIGHTED_MAJORITY>(vehicle_width_, vehicle_length_);
	// ------------------------------------------ getting rgb image ------------------------------------------
	auto[success, rgb_image] = front_rgb_->pop();
#ifndef __RELEASE
	if (!success) {
		std::cout << "ERROR: agent " + std::to_string(id_) + "'s rgb cam is unresponsive" << std::endl;
		return datapoint;
	}
#endif
	// filling the datapoint
	datapoint.agent_id = agent_batch_index;
	// omp for does not improve performance, may even degrade
	for (size_t i = 0; i < statics::front_rgb_byte_count; ++i) {
		datapoint.front_rgb[i] = rgb_image.data[i];
	}
	for (size_t i = 0; i < statics::top_semseg_byte_count; ++i) {
		datapoint.top_semseg[i] = semantic_bev.data[i];
	}
	for (size_t i = 0; i < statics::top_semseg_byte_count; ++i) {
		datapoint.top_mask[i] = mask.data[i];
	}
	for (size_t i = 0; i < statics::transform_length; ++i) {
		datapoint.transform[i] = transform_.data()[i];
	}
	return datapoint;
}

/* moves the agent underground next to the ninja turtles */
void MassAgent::HideAgent() {
	carla::geom::Transform tf = vehicle_->GetTransform();
	tf.location.z = -200;
    transform_(2, 3) = -200;
	vehicle_->SetTransform(tf);
	// blocking until moved
	do {
		auto current_tf = vehicle_->GetTransform();
		// if this is not enough, it just means I have a scientifically proven shit luck
		if (std::abs(current_tf.location.z - tf.location.z) < 1e-2) {
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(config::kPollInterval));
	} while(true);
}

/* stops the callback on all sensors */
void MassAgent::PauseSensorCallbacks() {
	front_rgb_->PauseCallback();
	front_mask_pc_->PauseCallback();
	for (auto& semantic_cam : semantic_pc_cams_) {
		semantic_cam->PauseCallback();
	}
}

/* resumes the callback on all sensors */
void MassAgent::ResumeSensorCallbacks() {
	front_rgb_->ResumeCallback();
	front_mask_pc_->ResumeCallback();
	for (auto& semantic_cam : semantic_pc_cams_) {
		semantic_cam->ResumeCallback();
	}
}

/* returns an image of the map */
cv::Mat MassAgent::GetMap() {
	float minx = 0.0f, maxx = 0.0f;
	float miny = 0.0f, maxy = 0.0f;
	// initialize waypoints if not done already
	InitializeKDTree();
	std::cout << "visaulizing " << kd_points_.size() << " waypoints" << std::endl;
	for (auto& waypoint : kd_points_) {
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
		// id is useless
		// road id is the same as street name
		// lane id: opendrive standard
		// section id, mostly zero, rarely 1, barely 2
	}
	size_t rows = static_cast<size_t>(maxy - miny) * 10 + 10;
	size_t cols = static_cast<size_t>(maxx - minx) * 10 + 10;
	cv::Mat map_img = cv::Mat::zeros(rows, cols, CV_8UC3);
	map_img = cv::Scalar(127, 127, 127);
	minx = std::abs(minx);
	miny = std::abs(miny);
	for (auto& waypoint : kd_points_) {
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
	return map_img;
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
		"vehicle.mustang.mustang",
		"vehicle.lincoln.mkz2017",
		"vehicle.toyota.prius",
		"vehicle.bmw.grandtourer",
		"vehicle.tesla.model3",
		"vehicle.nissan.patrol",
		"vehicle.jeep.wrangler_rubicon",
		"vehicle.dodge_charger.police",
		"vehicle.chevrolet.impala",
		"vehicle.audi.etron",
		"vehicle.seat.leon",
	};
	return vehicle_names;
}

/* initializes waypoints and builds a kd index on top of them */
void MassAgent::InitializeKDTree() {
	auto waypoints = carla_client()->GetWorld().GetMap()->GenerateWaypoints(config::kWaypointGenerationDistance);
	kd_points_.insert(kd_points_.begin(), waypoints.begin(), waypoints.end());
	kd_tree_ = std::make_unique<WaypointKDTree>(3, *this,
		nanoflann::KDTreeSingleIndexAdaptorParams(config::kWaypointKDTreeBins));
	kd_tree_->buildIndex();
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
void MassAgent::SetupSensors(float rgb_cam_shift) {
	std::string yaml_path = ros::package::getPath("mass_data_collector");
	yaml_path += "/param/sensors.yaml";
	auto bp_library = vehicle_->GetWorld().GetBlueprintLibrary();
	YAML::Node base = YAML::LoadFile(yaml_path);
	YAML::Node front_rgb_node = base["camera-front"];
	YAML::Node front_msk_node = base["camera-mask"];
	// front rgb cam
	if (front_rgb_node.IsDefined()) {
		front_rgb_ = std::make_unique<data::RGBCamera>(front_rgb_node, bp_library,
													   vehicle_, rgb_cam_shift, false);
	} else {
		std::cout << "ERROR: front camera node is not defined in sensor definition file" << std::endl;
	}
	// front semantic cam for mask generation. i'm a fucking genius
	if (front_msk_node.IsDefined()) {
		front_mask_pc_ = std::make_unique<data::SemanticPointCloudCamera>(front_msk_node,
															bp_library,
															vehicle_,
															0, // no x hover
															0, // no y hover
															false);
	} else {
		std::cout << "ERROR: front mask camera node is not defined in sensor definition file" << std::endl;
	}
	// depth camera
	YAML::Node mass_cam_node = base["camera-mass"];
	if (mass_cam_node.IsDefined()) {
		// reading camera grid file
		std::string yaml_path = ros::package::getPath("mass_data_collector")
							+ "/param/camera_grid.yaml";
		YAML::Node camera_grid = YAML::LoadFile(yaml_path);
		size_t rows = camera_grid["rows"].as<size_t>();
		size_t cols = camera_grid["cols"].as<size_t>();
		for (size_t i = 0; i < rows; ++i) {
			auto row_str = "row-" + std::to_string(i);
			for (size_t j = 0; j < cols; ++j) {
				auto col_str = "col-" + std::to_string(j);
				if (!camera_grid[row_str][col_str]["enabled"].as<bool>()) {
					continue;
				}
				auto delta_x = camera_grid[row_str][col_str]["x"].as<float>();
				auto delta_y = camera_grid[row_str][col_str]["y"].as<float>();
				semantic_pc_cams_.emplace_back(std::make_unique<data::SemanticPointCloudCamera>(mass_cam_node,
																bp_library,
																vehicle_,
																delta_x,
																delta_y,
																false));
			}
		}
	} else {
		std::cout << "ERROR: mass camera node is not defined in sensor definition file" << std::endl;
	}
}

/* returns the static carla client. only one is needed for all agents */
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

/* asserts that all the sensor data containers have the same size */
void MassAgent::AssertSize(size_t size) {
	for (auto& spc_cam : semantic_pc_cams_) {
		bool assertion = (spc_cam->depth_image_count() == spc_cam->semantic_image_count() &&
						  spc_cam->depth_image_count() == size);
		if (!assertion) {
			std::cout << "\nassertion failed: " << spc_cam->name() << ": " << spc_cam->depth_image_count()
					  << ", " << spc_cam->semantic_image_count() << " != " << size;
		}
	}
	bool assertion = (front_mask_pc_->depth_image_count() == front_mask_pc_->semantic_image_count() &&
					  front_mask_pc_->depth_image_count() == size);
	if (!assertion) {
		std::cout << "\nassertion failed: front pc cam: " << front_mask_pc_->depth_image_count()
					<< ", " << front_mask_pc_->semantic_image_count() << " != " << size;
	}
	if (front_rgb_->count() != size) {
		std::cout << "\nassertion failed: front rgb: " << front_rgb_->count() << " != " << size;
	}
}

/* create a single cloud using all cameras of all agents */
void MassAgent::DebugMultiAgentCloud(MassAgent* agents, size_t size, const std::string& path) {
	geom::base_members::Settings semantic_conf{1000, -1000, 1000, -1000, 0.1, 0, 0, 7, 32, 128};
	geom::SemanticCloud<geom::CloudBackend::KD_TREE> target_cloud(semantic_conf);
	for (size_t i = 0; i < size; ++i) {
		agents[i].CaptureOnce();
		// ---------------------- creating target cloud ----------------------
		for (auto& semantic_depth_cam : agents[i].semantic_pc_cams_) {
			auto[success, semantic, depth] = semantic_depth_cam->pop();
			if (success) {
				Eigen::Matrix4d tf = agents[0].transform_.inverse() * agents[i].transform_;
				target_cloud.AddSemanticDepthImage(semantic_depth_cam->geometry(), semantic, depth, tf);
			} else {
				std::cout << "ERROR: agent " + std::to_string(agents[i].id_)
						+ "'s " << semantic_depth_cam->name() << " is unresponsive" << std::endl;
				return;
			}
		}
	}
	auto [minx, maxx, miny, maxy] = target_cloud.GetBoundaries();
	std::cout << "cloud boundaries: x in (" << minx << ", " << maxx << "), y in ("
								 			<< miny << ", " << maxy << ")" << std::endl;
	target_cloud.SaveCloud(path);
}

/* returns the cloud geometry settings */
geom::base_members::Settings& MassAgent::sc_settings() {
	static geom::base_members::Settings semantic_cloud_settings;
	static std::once_flag flag;
	std::call_once(flag, [&]() {
		std::string yaml_path = ros::package::getPath("mass_data_collector") +
								"/param/sc_settings.yaml";
		YAML::Node base = YAML::LoadFile(yaml_path);
		YAML::Node cloud = base["cloud"];
		YAML::Node bev = base["bev"];
		YAML::Node mask = base["mask"];
		semantic_cloud_settings.max_point_x = cloud["max_point_x"].as<float>();
		semantic_cloud_settings.min_point_x = cloud["min_point_x"].as<float>();
		semantic_cloud_settings.max_point_y = cloud["max_point_y"].as<float>();
		semantic_cloud_settings.min_point_y = cloud["min_point_y"].as<float>();
		semantic_cloud_settings.stitching_threshold = mask["stitching_threshold"].as<float>();
		semantic_cloud_settings.image_rows = bev["image_rows"].as<unsigned int>();
		semantic_cloud_settings.image_cols = bev["image_cols"].as<unsigned int>();
		semantic_cloud_settings.vehicle_mask_padding = mask["vehicle_mask_padding"].as<unsigned int>();
		semantic_cloud_settings.knn_count = mask["knn_count"].as<unsigned int>();
		semantic_cloud_settings.kd_max_leaf = base["kd_max_leaf"].as<unsigned int>();
	});
	return semantic_cloud_settings;
}

/* static function that returns a vector of all active agents */
std::vector<const MassAgent*>& MassAgent::agents() {
	static std::vector<const MassAgent*> agents;
	return agents;
}

/* used to explore the map to find the road ID of corrupt samples */
[[deprecated("debug function")]] void MassAgent::ExploreMap() {
	/* 
	enum class LaneType : uint32_t {
		None          = 0x1,
		Driving       = 0x1 << 1,
		Stop          = 0x1 << 2,
		Shoulder      = 0x1 << 3,
		Biking        = 0x1 << 4,
		Sidewalk      = 0x1 << 5,
		Border        = 0x1 << 6,
		Restricted    = 0x1 << 7,
		Parking       = 0x1 << 8,
		Bidirectional = 0x1 << 9,
		Median        = 0x1 << 10,
		Special1      = 0x1 << 11,
		Special2      = 0x1 << 12,
		Special3      = 0x1 << 13,
		RoadWorks     = 0x1 << 14,
		Tram          = 0x1 << 15,
		Rail          = 0x1 << 16,
		Entry         = 0x1 << 17,
		Exit          = 0x1 << 18,
		OffRamp       = 0x1 << 19,
		OnRamp        = 0x1 << 20,
		Any           = 0xFFFFFFFE
		};
	*/
	float poses[16][2] = {
	// {5.7187e+01, 1.9257e+02}, // buildings look weird
	// {147.305500, 151.092400}, // buildings look weird again
	// {-74.1480, -79.5321}, // fine but has a weird building artifact
	{2.2314e+02, 2.0064e+02}, // tunnel
	{1.6323e+02, 1.9390e+02}, // end of junction leading to tunnel
	{170.402800, 193.900000}, // beginning of tunnel
	{174.264400, 193.900000}, // more inside the tunnel
	{244.0300,  78.4943}, // inside tunnel and another car
	{244.2139,  86.0548}, // inside tunnel
	{244.0843,  80.7248}, // inside tunnel
	{234.0437,  99.7434}, // tunnel
	{234.2429, 107.9337}, // tunnel
	{234.1480, 104.0332}, // tunnel
	{ -35.6166, -177.2975}, // gas station?
	{ -29.9447, -174.1371}, // yes?
	{ -25.4275, -172.1239}, // maybe
	{248.6527, 147.9556}, // tunnel and side lane
	{251.6419, 148.2102}, // tunnel and side lane
	{250.8483, 154.8452}, // tunnel
	};
	std::string strings[] = {
		"tunnel",
		"end of junction leading to tunnel",
		"beginning of tunnel",
		"more inside the tunnel",
		"inside tunnel and another car",
		"inside tunnel",
		"inside tunnel",
		"tunnel",
		"tunnel",
		"tunnel",
		"gas station?",
		"yes?",
		"maybe",
		"tunnel and side lane",
		"tunnel and side lane",
		"tunnel"
	};
	auto map = carla_client()->GetWorld().GetMap();
	for (int i = 0; i < 33; ++i) {
		auto wp = map->GetWaypoint(carla::geom::Location(poses[i][0], -poses[i][1], 0.0));
		std::cout << i << " " + strings[i] << ": " << wp->GetRoadId() << " X " << wp->GetLaneId() << std::endl;
	}
}
} // namespace agent