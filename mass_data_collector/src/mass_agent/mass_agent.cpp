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

/* templated random choice code from CARLA example */
#define EXPECT_TRUE(pred) if (!(pred)) { throw std::runtime_error(#pred); }

template <typename RangeT, typename RNG>
static auto &RandomChoice(const RangeT &range, RNG &&generator) {
  EXPECT_TRUE(range.size() > 0u);
  std::uniform_int_distribution<size_t> dist{0u, range.size() - 1u};
  return range[dist(std::forward<RNG>(generator))];
}

/* constructor for dataset collection */
MassAgent::MassAgent(std::mt19937& random_generator, const std::unordered_map<int, bool>& restricted_roads)
	: random_generator_(random_generator) {
	// initialize noise distribution
	noise_setting_ = CollectionConfig::GetConfig().noise_setting;
	agent_yaw_noise_dist_ = std::normal_distribution<float>(noise_setting_.agent_yaw_mean,
															noise_setting_.agent_yaw_std);
	// initialize some stuff & things
	restricted_roads_ = restricted_roads;
	waypoint_lambda_ = [this] (auto wp) {
		return wp != nullptr &&
			   (static_cast<uint32_t>(wp->GetType()) &
			    static_cast<uint32_t>(carla::road::Lane::LaneType::Driving)) != 0u &&
			   restricted_roads_.find(wp->GetRoadId()) == restricted_roads_.end();
	};
	transform_.setIdentity();
	id_ = agents().size();
	agents().emplace_back(this);
	try {
		auto world = carla_client()->GetWorld();
		auto spawn_points = world.GetMap()->GetRecommendedSpawnPoints();
		// initialize kd_tree (if empty)
		InitializeKDTree();
		// random the car model
		auto car_models = GetBlueprintNames();
		bool repetitive = false;
		auto& all_agents = agents();
		do {
			repetitive = false;
			blueprint_name_	= RandomChoice(car_models, random_generator_);
			for (size_t i = 0; i < all_agents.size(); ++i) {
				if (i != id_ && blueprint_name_ == all_agents[i]->blueprint_name_) {
					repetitive = true;
					break;
				}
			}
		} while(repetitive);
		auto blueprint_library = world.GetBlueprintLibrary();
		auto vehicles = blueprint_library->Filter(blueprint_name_);
		if (vehicles->empty()) {
			ROS_ERROR("ERROR: did not find car model with name: %s... using vehicle.seat.leon", blueprint_name_.c_str());
			vehicles = blueprint_library->Filter("vehicle.seat.leon");
		}
		carla::client::ActorBlueprint blueprint = (*vehicles)[0];
		// randomize the blueprint color
		if (blueprint.ContainsAttribute("color")) {
			auto &attribute = blueprint.GetAttribute("color");
			blueprint.SetAttribute(
				"color",
				RandomChoice(attribute.GetRecommendedValues(), random_generator_));
		}
		auto initial_pos = spawn_points[id_];
		Eigen::Matrix3d rot;
		rot = Eigen::AngleAxisd(-initial_pos.rotation.roll  * config::kToRadians, Eigen::Vector3d::UnitX()) *
			  Eigen::AngleAxisd(-initial_pos.rotation.pitch * config::kToRadians, Eigen::Vector3d::UnitY()) *
			  Eigen::AngleAxisd(-initial_pos.rotation.yaw   * config::kToRadians, Eigen::Vector3d::UnitZ());
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

/* constructor for runtime */
MassAgent::MassAgent(std::mt19937& random_generator, float x, float y, float z)
	: random_generator_(random_generator) {
	transform_.setIdentity();
	id_ = agents().size();
	agents().emplace_back(this);
	try {
		auto world = carla_client()->GetWorld();
		auto spawn_points = world.GetMap()->GetRecommendedSpawnPoints();
		// initialize kd_tree (if empty)
		InitializeKDTree();
		// random the car model
		auto car_models = GetBlueprintNames();
		bool repetitive = false;
		auto& all_agents = agents();
		do {
			repetitive = false;
			blueprint_name_	= RandomChoice(car_models, random_generator_);
			for (size_t i = 0; i < all_agents.size(); ++i) {
				if (i != id_ && blueprint_name_ == all_agents[i]->blueprint_name_) {
					repetitive = true;
					break;
				}
			}
		} while(repetitive);
		auto blueprint_library = world.GetBlueprintLibrary();
		auto vehicles = blueprint_library->Filter(blueprint_name_);
		if (vehicles->empty()) {
			ROS_ERROR("ERROR: did not find car model with name: %s... using vehicle.seat.leon", blueprint_name_.c_str());
			vehicles = blueprint_library->Filter("vehicle.seat.leon");
		}
		carla::client::ActorBlueprint blueprint = (*vehicles)[0];
		// randomize the blueprint color
		if (blueprint.ContainsAttribute("color")) {
			auto &attribute = blueprint.GetAttribute("color");
			blueprint.SetAttribute(
				"color",
				RandomChoice(attribute.GetRecommendedValues(), random_generator_));
		}
		auto initial_pos = spawn_points[id_];
		auto actor = world.TrySpawnActor(blueprint, initial_pos);
		if (!actor) {
			std::cout << "failed to spawn " << blueprint_name_ << ". exiting..." << std::endl;
			std::exit(EXIT_FAILURE);
		}
		vehicle_ = boost::static_pointer_cast<cc::Vehicle>(actor);
		// set location to the given point
		double query[3] = {x, y, z};
		std::vector<size_t> knn_ret_indices(1);
		std::vector<double> knn_sqrd_dist(1);
		kd_tree_->knnSearch(&query[0], 1, &knn_ret_indices[0], &knn_sqrd_dist[0]);
		auto new_pos = kd_points_[knn_ret_indices[0]]->GetTransform();
		vehicle_->SetTransform(new_pos);
		Eigen::Matrix3d rot;
		rot = Eigen::AngleAxisd(-new_pos.rotation.roll  * config::kToRadians, Eigen::Vector3d::UnitX()) *
			  Eigen::AngleAxisd(-new_pos.rotation.pitch * config::kToRadians, Eigen::Vector3d::UnitY()) *
			  Eigen::AngleAxisd(-new_pos.rotation.yaw   * config::kToRadians, Eigen::Vector3d::UnitZ());
    	Eigen::Vector3d trans(new_pos.location.x, -new_pos.location.y, new_pos.location.z);
		transform_.block<3, 3>(0, 0) = rot;
    	transform_.block<3, 1>(0, 3) = trans;
		// turn on physics & autopilot
		vehicle_->SetSimulatePhysics(false);
		vehicle_->SetAutopilot(false);
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
	std::cout << "created autonomous mass-agent-" << id_ << ": " + blueprint_name_ << std::endl;
}

/* destructor */
MassAgent::~MassAgent() {
	// releasing resources
	DestroyAgent();
}

std::tuple<carla::geom::Rotation, Eigen::Matrix3d> MassAgent::AddSE3Noise(const Eigen::Matrix3d& initial_rotation) {
	float yaw_noise = agent_yaw_noise_dist_(random_generator_);
	Eigen::Matrix3d rot_noise;
	rot_noise = Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX()) *
		  		Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
		  		Eigen::AngleAxisd(yaw_noise * config::kToRadians, Eigen::Vector3d::UnitZ());
	Eigen::Matrix3d new_rotation_matrix = initial_rotation * rot_noise;
	Eigen::Vector3d euler = new_rotation_matrix.eulerAngles(0, 1, 2);
	// carla is left handed => negative
	carla::geom::Rotation new_rotation_vector{-euler[1] * config::kToDegrees,
											  -euler[2] * config::kToDegrees,
											  -euler[0] * config::kToDegrees};
	return std::make_tuple(new_rotation_vector, new_rotation_matrix);
}

/* places the agent on a random point in the map and makes sure it doesn't collide with others [[blocking]] 
   this function should only be called for the first car of the batch, because it does not check for collision
   with other cars. */
boost::shared_ptr<carla::client::Waypoint> MassAgent::SetRandomPose() {
	boost::shared_ptr<carla::client::Waypoint> initial_wp;
	carla::geom::Transform tf;
	while (true) {
		initial_wp = kd_points_[random_generator_() % kd_points_.size()];
		// if not in a restricted area
		if (restricted_roads_.find(initial_wp->GetRoadId()) == restricted_roads_.end()) {
			tf = initial_wp->GetTransform();
			break;
		}
	}
	Eigen::Matrix3d rot;
	rot = Eigen::AngleAxisd(-tf.rotation.roll  * config::kToRadians, Eigen::Vector3d::UnitX()) *
		  Eigen::AngleAxisd(-tf.rotation.pitch * config::kToRadians, Eigen::Vector3d::UnitY()) *
		  Eigen::AngleAxisd(-tf.rotation.yaw   * config::kToRadians, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d trans(tf.location.x, -tf.location.y, tf.location.z);
	// creating noise if enabled and "coin toss" is successful
	if (noise_setting_.agent_yaw_enable &&
		static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) <= noise_setting_.agent_yaw_chance) {
		carla::geom::Rotation noisy_vec;
		std::tie(noisy_vec, rot) = AddSE3Noise(rot);
		tf.rotation = noisy_vec;
	}
	transform_.block<3, 1>(0, 3) = trans;
	transform_.block<3, 3>(0, 0) = rot;
	vehicle_->SetTransform(tf);
	return initial_wp;
}

/* expands the given waypoint by finding the points in its vicinity */
std::vector<boost::shared_ptr<carla::client::Waypoint>>
MassAgent::ExpandWayoint(boost::shared_ptr<carla::client::Waypoint> wp, double min_dist) {
	std::vector<boost::shared_ptr<carla::client::Waypoint>> candidates;
	candidates.reserve(64);
	auto front_wps = wp->GetNext(min_dist +
						(static_cast<double>(std::rand() % config::kMaxAgentDistance) / 100.0));
	auto rear_wps = wp->GetPrevious(min_dist +
						(static_cast<double>(std::rand() % config::kMaxAgentDistance) / 100.0));
	std::copy_if(front_wps.begin(), front_wps.end(), std::back_inserter(candidates), waypoint_lambda_);
	std::copy_if(rear_wps.begin(), rear_wps.end(), std::back_inserter(candidates), waypoint_lambda_);
	auto left_wp = wp->GetLeft();
	auto right_wp = wp->GetRight();
	if (waypoint_lambda_(left_wp)) {
		candidates.emplace_back(left_wp);
	}
	if (waypoint_lambda_(right_wp)) {
		candidates.emplace_back(right_wp);
	}
	return candidates;
}

/* places the agent around the given waypoint and also makes sure it doesn't collide with others [[blocking]] 
   the indices of the other agents must be given and only up until `max_index will be considered */
boost::shared_ptr<carla::client::Waypoint>
MassAgent::SetRandomPose(boost::shared_ptr<carla::client::Waypoint> initial_wp,
						 size_t knn_pts, std::vector<unsigned int> agent_indices,
						 unsigned int max_index) {
	// last agent in queue failed placement, forwarding nullptr to others
	if (initial_wp == nullptr) {
		return nullptr;
	}
	// getting candidates around the given waypoints
	std::vector<boost::shared_ptr<carla::client::Waypoint>> candidates =
		ExpandWayoint(initial_wp, vehicle_length_ * config::kMinDistCoeff);
	// if there are not candidates, signal placement failure
	if (candidates.size() == 0) {
		return nullptr;
	}
	if (knn_pts > 0) {
		auto query_tfm = initial_wp->GetTransform();
		double query[3] = {query_tfm.location.x, query_tfm.location.y, query_tfm.location.z};
		std::vector<size_t> knn_ret_indices(knn_pts);
		std::vector<double> knn_sqrd_dist(knn_pts);
		knn_pts = kd_tree_->knnSearch(&query[0], knn_pts, &knn_ret_indices[0], &knn_sqrd_dist[0]);
		knn_ret_indices.resize(knn_pts);
		for (auto close_wp_idx : knn_ret_indices) {
			auto expansion = ExpandWayoint(kd_points_[close_wp_idx], vehicle_length_ * config::kMinDistCoeff);
			candidates.insert(candidates.end(), expansion.begin(), expansion.end());
		}
	}
	// shuffling and iterating candidates, choosing first good match
	boost::shared_ptr<carla::client::Waypoint> random_candidate_wp;
	carla::geom::Transform target_tf;
	std::vector<unsigned int> shuffled_candidates(candidates.size());
    std::iota(shuffled_candidates.begin(), shuffled_candidates.end(), 0);
	std::shuffle(shuffled_candidates.begin(), shuffled_candidates.end(), random_generator_);
	bool admissable;
	// we assume the placement is admissable until proven otherwise
	for (unsigned int candidate_idx : shuffled_candidates) {
		random_candidate_wp = candidates[candidate_idx];
		target_tf = random_candidate_wp->GetTransform();
		admissable = true;
		for (unsigned int i = 0; i < max_index; ++i) {
			const MassAgent* agent = agents()[agent_indices[i]];
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
				break;
			}
		}
		if (admissable) {
			break;
		}
	}
	// if no good matches are found, return nullptr to signal failure
	if (!admissable) {
		return nullptr;
	}
	Eigen::Matrix3d rot;
	rot = Eigen::AngleAxisd(-target_tf.rotation.roll  * config::kToRadians, Eigen::Vector3d::UnitX()) *
		  Eigen::AngleAxisd(-target_tf.rotation.pitch * config::kToRadians, Eigen::Vector3d::UnitY()) *
		  Eigen::AngleAxisd(-target_tf.rotation.yaw   * config::kToRadians, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d trans(target_tf.location.x, -target_tf.location.y, target_tf.location.z);
	// creating noise if enabled and "coin toss" is successful
	if (noise_setting_.agent_yaw_enable &&
		static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) <= noise_setting_.agent_yaw_chance) {
		carla::geom::Rotation noisy_vec;
		std::tie(noisy_vec, rot) = AddSE3Noise(rot);
		target_tf.rotation = noisy_vec;
	}
    transform_.block<3, 3>(0, 0) = rot;
    transform_.block<3, 1>(0, 3) = trans;
	vehicle_->SetTransform(target_tf);
	return random_candidate_wp;
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
MASSDataType MassAgent::GenerateDataPoint(unsigned int agent_batch_index) const {
	MASSDataType datapoint{};
	// ----------------------------------------- creating mask cloud -----------------------------------------
	geom::SemanticCloud mask_cloud(sc_settings());
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
	geom::SemanticCloud target_cloud(sc_settings());
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

/* moves the agent underground next to the ninja turtles */
void MassAgent::HideAgent() {
	carla::geom::Transform tf = vehicle_->GetTransform();
	tf.location.z = -200;
    transform_(2, 3) = -200;
	vehicle_->SetTransform(tf);
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
		// "vehicle.nissan.patrol",
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

/* move the vehicle forward in the simulator while avoiding collisions & obeying traffic rules */
void MassAgent::MoveForward(double distance) {
	auto curr_tfm = vehicle_->GetTransform();
	// get current and next waypoint
	double query[3] = {curr_tfm.location.x, curr_tfm.location.y, curr_tfm.location.z};
	std::vector<size_t> knn_ret_indices(1);
	std::vector<double> knn_sqrd_dist(1);
	kd_tree_->knnSearch(&query[0], 1, &knn_ret_indices[0], &knn_sqrd_dist[0]);
	auto curr_waypoint = kd_points_[knn_ret_indices[0]];
	auto next_waypoint = curr_waypoint->GetNext(distance)[0];
	auto curr_location = vehicle_->GetLocation();
	auto next_location = next_waypoint->GetTransform().location;
	// check if move is collision safe
	// for (auto* agent : agents()) {
	// 	if (agent->id_ == id_) {
	// 		continue;
	// 	}
	// 	auto distance = std::sqrt((curr_location.x - next_location.x) *
	// 							  (curr_location.x - next_location.x) +
	// 							  (curr_location.y - next_location.y) *
	// 							  (curr_location.y - next_location.y) +
	// 							  (curr_location.z - next_location.z) *
	// 							  (curr_location.z - next_location.z));
	// 	if (distance < vehicle_length_ * config::kMinDistCoeff) {
	// 		std::cout << "\n agent " << id_ << " not moved bcs of " << agent->id_ << std::endl;
	// 		return;
	// 	} 
	// }
	// check if at red light
	// if (next_waypoint->GetJunctionId() != -1) {
	// 	auto tlstate = vehicle_->GetTrafficLightState();
	// 	if (tlstate != carla::rpc::TrafficLightState::Yellow ||
	// 		tlstate != carla::rpc::TrafficLightState::Red) {
	// 		return;
	// 	}
	// }
	auto next_tnsfm = next_waypoint->GetTransform();
	vehicle_->SetTransform(next_tnsfm);
	Eigen::Matrix3d rot;
	rot = Eigen::AngleAxisd(-next_tnsfm.rotation.roll  * config::kToRadians, Eigen::Vector3d::UnitX()) *
		  Eigen::AngleAxisd(-next_tnsfm.rotation.pitch * config::kToRadians, Eigen::Vector3d::UnitY()) *
		  Eigen::AngleAxisd(-next_tnsfm.rotation.yaw   * config::kToRadians, Eigen::Vector3d::UnitZ());
	Eigen::Vector3d trans(next_tnsfm.location.x, -next_tnsfm.location.y, next_tnsfm.location.z);
	transform_.block<3, 3>(0, 0) = rot;
	transform_.block<3, 1>(0, 3) = trans;
}

/* initializes the structures for the camera, lidar & depth measurements */
void MassAgent::SetupSensors(float front_cam_shift) {
	std::string yaml_path = ros::package::getPath("mass_data_collector");
	yaml_path += "/param/sensors.yaml";
	auto bp_library = vehicle_->GetWorld().GetBlueprintLibrary();
	YAML::Node base = YAML::LoadFile(yaml_path);
	YAML::Node front_rgb_node = base["camera-front"];
	YAML::Node front_msk_node = base["camera-mask"];
	// front rgb cam
	if (front_rgb_node.IsDefined()) {
		front_rgb_ = std::make_unique<data::RGBCamera>(front_rgb_node, bp_library,
													   vehicle_, front_cam_shift, false);
	} else {
		std::cout << "ERROR: front camera node is not defined in sensor definition file" << std::endl;
	}
	// front semantic cam for mask generation. i'm a fucking genius
	if (front_msk_node.IsDefined()) {
		front_mask_pc_ = std::make_unique<data::SemanticPointCloudCamera>(front_msk_node,
															bp_library,
															vehicle_,
															front_cam_shift, // same as rgb
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
	static std::unique_ptr<cc::Client> carla_client;
	static std::once_flag flag;
	std::call_once(flag, [&]() {
		try {
			carla_client = std::make_unique<cc::Client>("127.0.0.1", 2000);
			carla_client->SetTimeout(10s);
			std::cout << "client version: " << carla_client->GetClientVersion() << "\t"
					  << "server version: " << carla_client->GetServerVersion() << std::endl;
		} catch (carla::client::TimeoutException& e) {
			std::cout << "connection to the simulator timed out" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	});
	return carla_client;
}

/* returns the RGB image, semantic mask and FoV mask */
std::tuple<cv::Mat, cv::Mat, cv::Mat> MassAgent::GetBEVSample() {
	geom::SemanticCloud mask_cloud(sc_settings());
	auto[succ, front_semantic, front_depth] = front_mask_pc_->pop();
	mask_cloud.AddSemanticDepthImage(front_mask_pc_->geometry(), front_semantic, front_depth);
	mask_cloud.BuildKDTree();
	cv::Mat fov_mask = mask_cloud.GetFOVMask();
	// ---------------------------------------- creating target cloud ----------------------------------------
	geom::SemanticCloud target_cloud(sc_settings());
	for (auto& semantic_depth_cam : semantic_pc_cams_) {
		auto[success, semantic, depth] = semantic_depth_cam->pop();
		target_cloud.AddSemanticDepthImage(semantic_depth_cam->geometry(), semantic, depth);
	}
	target_cloud.BuildKDTree();
	auto[semantic_bev, vehicle_mask] = target_cloud.GetBEVData(vehicle_width_, vehicle_length_);
	semantic_bev = geom::SemanticCloud::ConvertToCityScapesPallete(semantic_bev);
	// ------------------------------------------ getting rgb image ------------------------------------------
	fov_mask += vehicle_mask;
	auto[success, rgb_image] = front_rgb_->pop();
	return std::make_tuple(rgb_image, semantic_bev, fov_mask);
}

/* saves the cloud generated from front view cam and the full cloud that is masked using camera geometry*/
cv::Mat MassAgent::SaveMaskedClouds(const std::string& front_view, const std::string& full_view) {
	geom::SemanticCloud mask_cloud(sc_settings());
	auto[succ, front_semantic, front_depth] = front_mask_pc_->pop();
	mask_cloud.AddSemanticDepthImage(front_mask_pc_->geometry(), front_semantic, front_depth);
	mask_cloud.BuildKDTree();
	mask_cloud.SaveCloud(front_view);
	// ---------------------------------------- creating target cloud ----------------------------------------
	geom::SemanticCloud target_cloud(sc_settings());
	for (auto& semantic_depth_cam : semantic_pc_cams_) {
		auto[success, semantic, depth] = semantic_depth_cam->pop();
		target_cloud.AddSemanticDepthImage(semantic_depth_cam->geometry(), semantic, depth);
	}
	target_cloud.BuildKDTree();
	target_cloud.SaveMaskedCloud(front_rgb_->geometry(), full_view, 0.5);
	return geom::SemanticCloud::ConvertToCityScapesPallete(front_semantic);
}

void MassAgent::SaveFullCloud(const std::string& full_view) {
	geom::SemanticCloud target_cloud(sc_settings());
	for (auto& semantic_depth_cam : semantic_pc_cams_) {
		auto[success, semantic, depth] = semantic_depth_cam->pop();
		target_cloud.AddSemanticDepthImage(semantic_depth_cam->geometry(), semantic, depth);
	}
	target_cloud.SaveCloud(full_view);
}

/* asserts that all the sensor data containers have the same size */
void MassAgent::AssertSize(size_t size) const {
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

/* create a single cloud using all top-down cameras of all agents */
void MassAgent::SaveMultiAgentCloud(const std::string& path, size_t agent_index) {
	geom::SemanticCloud::Settings semantic_conf{1000, -1000, 1000, -1000, 0.1, 0, 0, 7, 32, 128};
	geom::SemanticCloud target_cloud(semantic_conf);
	for (size_t i = 0; i < agents().size(); ++i) {
		// ---------------------- creating target cloud ----------------------
		for (auto& semantic_depth_cam : agents()[i]->semantic_pc_cams_) {
			auto[success, semantic, depth] = semantic_depth_cam->pop();
			if (success) {
				Eigen::Matrix4d tf = agents()[agent_index]->transform_.inverse() * agents()[i]->transform_;
				target_cloud.AddSemanticDepthImage(semantic_depth_cam->geometry(), semantic, depth, tf);
			} else {
				std::cout << "ERROR: agent " + std::to_string(agents()[i]->id_)
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
geom::SemanticCloud::Settings& MassAgent::sc_settings() {
	static geom::SemanticCloud::Settings semantic_cloud_settings;
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
std::vector<MassAgent*>& MassAgent::agents() {
	static std::vector<MassAgent*> agents;
	return agents;
}

} // namespace agent