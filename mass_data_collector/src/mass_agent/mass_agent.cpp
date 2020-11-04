#include <carla/geom/Vector3D.h>
#include "mass_agent/mass_agent.h"
#include "mass_agent/agent_config.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <opencv2/highgui.hpp>
#include <ros/package.h>
#include <tuple>
#include <yaml-cpp/yaml.h>
#include <cmath>

#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/io/FileFormatIO.h>
#include <open3d/Open3D.h>

using namespace std::chrono_literals;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

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
		std::cout << "setting up front rgb camera" << std::endl;
		auto cam_blueprint = *bp_library->Find(front_cam_node["type"].as<std::string>());
		// setting camera attributes from the yaml
		for (YAML::const_iterator it = front_cam_node.begin(); it != front_cam_node.end(); ++it) {
			auto key = it->first.as<std::string>();
			if (cam_blueprint.ContainsAttribute(key)) {
				cam_blueprint.SetAttribute(key, it->second.as<std::string>());
			}
		}
		// spawn the camera attached to the vehicle
		auto camera_transform = cg::Transform{
			cg::Location{front_cam_node["x"].as<float>(),
						 front_cam_node["y"].as<float>(),
						 front_cam_node["z"].as<float>()},
			cg::Rotation{front_cam_node["pitch"].as<float>(),
						 front_cam_node["yaw"].as<float>(),
						 front_cam_node["roll"].as<float>()}};
		auto generic_actor = vehicle_->GetWorld().SpawnActor(cam_blueprint, camera_transform, vehicle_.get());
		front_cam_.sensor = boost::static_pointer_cast<cc::Sensor>(generic_actor);
		front_cam_.count = 0;
		// register a callback to publish images
		front_cam_.sensor->Listen([this](const boost::shared_ptr<carla::sensor::SensorData>& data) {
			auto image = boost::static_pointer_cast<csd::Image>(data);
			auto image_view = carla::image::ImageView::MakeView(*image);
			auto rgb_view = boost::gil::color_converted_view<boost::gil::rgb8_pixel_t>(image_view);
			using pixel = decltype(rgb_view)::value_type;
			static_assert(sizeof(pixel) == 3, "R, G & B");
			pixel raw_data[rgb_view.width() * rgb_view.height()];
			boost::gil::copy_pixels(rgb_view, boost::gil::interleaved_view(rgb_view.width(),
																		   rgb_view.height(),
																		   raw_data,
																		   rgb_view.width() * sizeof(pixel)));
			front_cam_.carla_image = cv::Mat(rgb_view.height(), rgb_view.width(), CV_8UC3, raw_data);
			if (++front_cam_.count == 1) {
				cv::imshow("front_rgb", front_cam_.carla_image);
				cv::waitKey(1); // NOLINT
			}
		});
	}
	// depth camera
	YAML::Node depth_cam_node = base["camera-depth"];
	if (depth_cam_node.IsDefined()) {
		std::cout << "setting up front depth camera" << std::endl;
		// usual camera info stuff
		auto cam_blueprint = *bp_library->Find(depth_cam_node["type"].as<std::string>());
		// setting camera attributes from the yaml
		for (YAML::const_iterator it = depth_cam_node.begin(); it != depth_cam_node.end(); ++it) {
			auto key = it->first.as<std::string>();
			if (cam_blueprint.ContainsAttribute(key)) {
				cam_blueprint.SetAttribute(key, it->second.as<std::string>());
			}
		}
		// spawn the depth camera attached to the vehicle
		auto camera_transform = cg::Transform{
			cg::Location{depth_cam_node["x"].as<float>(),
						 depth_cam_node["y"].as<float>(),
						 depth_cam_node["z"].as<float>()},
			cg::Rotation{depth_cam_node["pitch"].as<float>(),
						 depth_cam_node["yaw"].as<float>(),
						 depth_cam_node["roll"].as<float>()}};
		auto generic_actor = vehicle_->GetWorld().SpawnActor(cam_blueprint, camera_transform, vehicle_.get());
		depth_cam_.sensor = boost::static_pointer_cast<cc::Sensor>(generic_actor);
		depth_cam_.count = 0;
		// register a callback to publish images
		depth_cam_.sensor->Listen([this](const boost::shared_ptr<carla::sensor::SensorData>& data) {
			auto image = boost::static_pointer_cast<csd::Image>(data);
			auto image_view = carla::image::ImageView::MakeColorConvertedView(carla::image::ImageView::MakeView(*image),
																			  carla::image::ColorConverter::LogarithmicDepth());
			// auto image_view = carla::image::ImageView::MakeView(*image);
			auto grayscale_view = boost::gil::color_converted_view<boost::gil::gray8_pixel_t>(image_view);
			using pixel = decltype(grayscale_view)::value_type;
			static_assert(sizeof(pixel) == 1, "single channel");
			pixel raw_data[grayscale_view.width() * grayscale_view.height()];
			boost::gil::copy_pixels(grayscale_view, boost::gil::interleaved_view(grayscale_view.width(),
																				 grayscale_view.height(),
																				 raw_data,
																				 grayscale_view.width() * sizeof(pixel)));
			depth_cam_.carla_image = cv::Mat(grayscale_view.height(), grayscale_view.width(), CV_8UC1, raw_data);
			if (++depth_cam_.count == 1) {
				cv::imshow("front_depth", depth_cam_.carla_image);
				cv::waitKey(1); // NOLINT
			}
		});
	}
	// semantic segmentation camera
	YAML::Node semseg_cam_node = base["camera-semseg"];
	if (semseg_cam_node.IsDefined()) {
		std::cout << "setting up front semantic camera" << std::endl;
		auto cam_blueprint = *bp_library->Find(semseg_cam_node["type"].as<std::string>());
		// setting camera attributes from the yaml
		for (YAML::const_iterator it = semseg_cam_node.begin(); it != semseg_cam_node.end(); ++it) {
			auto key = it->first.as<std::string>();
			if (cam_blueprint.ContainsAttribute(key)) {
				cam_blueprint.SetAttribute(key, it->second.as<std::string>());
			}
		}
		// spawn the camera attached to the vehicle.	
		auto camera_transform = cg::Transform{
			cg::Location{semseg_cam_node["x"].as<float>(),
						 semseg_cam_node["y"].as<float>(),
						 semseg_cam_node["z"].as<float>()},
			cg::Rotation{semseg_cam_node["pitch"].as<float>(),
						 semseg_cam_node["yaw"].as<float>(),
						 semseg_cam_node["roll"].as<float>()}};
		auto generic_actor = vehicle_->GetWorld().SpawnActor(cam_blueprint, camera_transform, vehicle_.get());
		semseg_cam_.sensor = boost::static_pointer_cast<cc::Sensor>(generic_actor);
		semseg_cam_.count = 0;
		// callback
		semseg_cam_.sensor->Listen([this](const boost::shared_ptr<carla::sensor::SensorData>& data) {
			auto image = boost::static_pointer_cast<csd::Image>(data);
			auto image_view = carla::image::ImageView::MakeColorConvertedView(carla::image::ImageView::MakeView(*image), 
																			  carla::image::ColorConverter::CityScapesPalette());
			auto rgb_view = boost::gil::color_converted_view<boost::gil::rgb8_pixel_t>(image_view);
			using pixel = decltype(rgb_view)::value_type;
			static_assert(sizeof(pixel) == 3, "R, G & B");
			pixel raw_data[rgb_view.width() * rgb_view.height()];
			boost::gil::copy_pixels(rgb_view, boost::gil::interleaved_view(rgb_view.width(),
																		   rgb_view.height(),
																		   raw_data,
																		   rgb_view.width() * sizeof(pixel)));
			semseg_cam_.carla_image = cv::Mat(rgb_view.height(), rgb_view.width(), CV_8UC3, raw_data);
			if (++semseg_cam_.count == 1) {
				cv::imshow("front_semseg", semseg_cam_.carla_image);
				cv::waitKey(1); // NOLINT
			}
		});
	}
}
/* destroys the agent in the simulation & its sensors. */
void MassAgent::DestroyAgent() {
	if (front_cam_.sensor) {
		front_cam_.sensor->Destroy();
		front_cam_.sensor = nullptr;
	}
	if (depth_cam_.sensor) {
		depth_cam_.sensor->Destroy();
		depth_cam_.sensor = nullptr;
	}
	if (semseg_cam_.sensor) {
		semseg_cam_.sensor->Destroy();
		semseg_cam_.sensor = nullptr;
	}
	if (vehicle_) {
		vehicle_->Destroy();
		vehicle_ = nullptr;
	}
	if (carla_client_) {
		carla_client_.reset();
	}
}
/* places the agent on a random point in the map at a random pose */
void MassAgent::SetRandomPose() {
	bool admissable = true;
	float yaw = 0;
	while (true) {
		admissable = true;
		auto lane_point = map_instance_.GetRandomLanePoint();
		std::tie(x_, y_, z_) = lane_point.GetCoords();
		yaw = lane_point.GetHeading();
		for (const auto *agent : agents()) {
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
/* returns the current position of the agent */
std::tuple<float, float, float> MassAgent::GetPostion() const {
	return std::make_tuple(x_, y_, z_);
}