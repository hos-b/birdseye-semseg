#include "carla_agent/carla_agent.h"
#include "carla/geom/Vector3D.h"
#include "carla_agent/agent_config.h"

#include <visualization_msgs/Marker.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
// #include <tf2_ros/transform_listener.h>

#include <ros/package.h>
#include <yaml-cpp/yaml.h>
#include <cmath>

#define TO_RAD M_PI / 180
#define TO_DEG 180 / M_PI
#define FIXED_FRAME "map"
#define FRONT_CAMERA_FRAME "/zed_camera"
using namespace std::chrono_literals;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

CarlaAgent::CarlaAgent(unsigned long thread_sleep_ms) {
    // initialize some stuff & things
    carla_client_ = nullptr;
    agent_thread_ = nullptr;
    thread_sleep_ms_ = thread_sleep_ms;
}

/* destructor */
CarlaAgent::~CarlaAgent() {
    // releasing thread resources
    StopAgentThread();
    DestroyAgent();
}

/* connects to airsim */
void CarlaAgent::ActivateCarlaAgent(const std::string &address, unsigned int port) {
    try {
        carla_client_ = std::make_unique<cc::Client>(address, port);
        carla_client_->SetTimeout(10s);
        std::cout << "connected to CARLA sucessfully" << std::endl;
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
        // TODO spawn at random pose
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
        ROS_ERROR("ERROR: connection to the CARLA server timed out\n%s", e.what());
        std::exit(EXIT_FAILURE);
    } catch (const std::exception &e) {
        ROS_ERROR("ERROR: %s", e.what());
        DestroyAgent();
        std::exit(EXIT_FAILURE);
    }
    SetupSensors();
}
/* sets up subscribers, publishers & runs thread */
void CarlaAgent::SetupCallbacks() {
    // sim_only handles these topics here
    agent_thread_ = new std::thread(&CarlaAgent::Step, this, thread_sleep_ms_);
    std::cout << ": carla proxy thread started" << std::endl;
}

/* thread function for CarlaAgent */
void CarlaAgent::Step(unsigned int thread_sleep_ms) {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(thread_sleep_ms));
    }
}



int roundUp(int numToRound, int multiple)
{
    if (multiple == 0)
        return numToRound;

    int remainder = numToRound % multiple;
    if (remainder == 0)
        return numToRound;

    return numToRound + multiple - remainder;
}

cv::Mat translateImg(cv::Mat &img, int offsetx, int offsety){
    cv::Mat trans_mat = (cv::Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
    warpAffine(img, img, trans_mat, img.size());
    return img;
}

/* initializes the structures for the camera, lidar & depth measurements */
void CarlaAgent::SetupSensors() {
    std::string yaml_path;
    if (!ros::param::get("~sensor_description_file", yaml_path)) {
        return;
    }
    std::cout << yaml_path << std::endl;
    auto bp_library = vehicle_->GetWorld().GetBlueprintLibrary();
    YAML::Node base = YAML::LoadFile(yaml_path);
    YAML::Node front_cam_node = base["camera-front"];
    if (front_cam_node.IsDefined()) {
        std::cout << "setting up front camera" << std::endl;
        auto cam_blueprint = *bp_library->Find(front_cam_node["type"].as<std::string>());
        // setting camera attributes from the yaml
        const float cx = front_cam_node["cx"].as<float>();
        const float cy = front_cam_node["cy"].as<float>();
        const int sx = front_cam_node["image_size_x"].as<int>();
        const int sy = front_cam_node["image_size_y"].as<int>();
        const float shift_x = cx - (sx/2.);
        const float shift_y = cy - (sy/2.);
        const int margin_x = roundUp(std::abs(int(shift_x)), 16);
        const int margin_y = roundUp(std::abs(int(shift_y)), 16);

        for (YAML::const_iterator it = front_cam_node.begin(); it != front_cam_node.end(); ++it) {
            auto key = it->first.as<std::string>();
            if (cam_blueprint.ContainsAttribute(key))
                cam_blueprint.SetAttribute(key, it->second.as<std::string>());
        }
        // spawn the camera attached to the vehicle.
        cam_blueprint.SetAttribute("image_size_x", std::to_string(sx + margin_x*2));
        cam_blueprint.SetAttribute("image_size_y", std::to_string(sy + margin_y*2));
        front_cam_.margin_x = margin_x;
        front_cam_.margin_y = margin_y;
        front_cam_.shift_x = shift_x;
        front_cam_.shift_y = shift_y;
        // Calculate margin for principal point cropping
        cg::Transform camera_transform; // = LookupCameraExtrinsics("/base_link", FRONT_CAMERA_FRAME);
        auto generic_actor = vehicle_->GetWorld().SpawnActor(cam_blueprint, camera_transform, vehicle_.get());
        front_cam_.sensor = boost::static_pointer_cast<cc::Sensor>(generic_actor);
        // register a callback to publish images
        front_cam_.sensor->Listen([this](boost::shared_ptr<carla::sensor::SensorData> data) {
            auto image = boost::static_pointer_cast<csd::Image>(data);
            auto image_view = carla::image::ImageView::MakeView(*image);
            auto rgb_view = boost::gil::color_converted_view<boost::gil::rgb8_pixel_t>(image_view);
            typedef decltype(rgb_view)::value_type pixel;
            static_assert(sizeof(pixel) == 3, "R, G & B");
            pixel raw_data[rgb_view.width() * rgb_view.height()];
            boost::gil::copy_pixels(rgb_view, boost::gil::interleaved_view(rgb_view.width(),
                                                                            rgb_view.height(),
                                                                            raw_data,
                                                                            rgb_view.width() * sizeof(pixel)));
            front_cam_.carla_image = cv::Mat(rgb_view.height(), rgb_view.width(), CV_8UC3, raw_data);
            front_cam_.carla_image = translateImg(front_cam_.carla_image, front_cam_.shift_x, front_cam_.shift_y);
            const cv::Rect roi(front_cam_.margin_x, front_cam_.margin_y,
                               int(front_cam_.carla_image.cols - (front_cam_.margin_x*2)),
                               int(front_cam_.carla_image.rows - (front_cam_.margin_y*2)));
            front_cam_.carla_image = front_cam_.carla_image(roi);
        });
    }
    YAML::Node depth_cam_node = base["camera-depth"];
    if (depth_cam_node.IsDefined()) {
        std::cout << "setting up front camera" << std::endl;
        auto cam_blueprint = *bp_library->Find(depth_cam_node["type"].as<std::string>());
        // setting camera attributes from the yaml
        const float cx = depth_cam_node["cx"].as<float>();
        const float cy = depth_cam_node["cy"].as<float>();
        const int sx = depth_cam_node["image_size_x"].as<int>();
        const int sy = depth_cam_node["image_size_y"].as<int>();
        const float shift_x = cx - (sx/2.);
        const float shift_y = cy - (sy/2.);
        const int margin_x = roundUp(std::abs(int(shift_x)), 16);
        const int margin_y = roundUp(std::abs(int(shift_y)), 16);
        for (YAML::const_iterator it = depth_cam_node.begin(); it != depth_cam_node.end(); ++it) {
            auto key = it->first.as<std::string>();
            if (cam_blueprint.ContainsAttribute(key))
                cam_blueprint.SetAttribute(key, it->second.as<std::string>());
        }
        cam_blueprint.SetAttribute("image_size_x", std::to_string(sx + margin_x*2));
        cam_blueprint.SetAttribute("image_size_y", std::to_string(sy + margin_y*2));
        depth_cam_.margin_x = margin_x;
        depth_cam_.margin_y = margin_y;
        depth_cam_.shift_x = shift_x;
        depth_cam_.shift_y = shift_y;

        // spawn the depth camera attached to the vehicle.
        cg::Transform camera_transform; //= LookupCameraExtrinsics("/base_link", FRONT_CAMERA_FRAME);
        auto generic_actor = vehicle_->GetWorld().SpawnActor(cam_blueprint, camera_transform, vehicle_.get());
        depth_cam_.sensor = boost::static_pointer_cast<cc::Sensor>(generic_actor);
        // register a callback to publish images
        depth_cam_.sensor->Listen([this](boost::shared_ptr<carla::sensor::SensorData> data) {
            auto image = boost::static_pointer_cast<csd::Image>(data);
            cv::Mat cvm(image->GetHeight(), image->GetWidth(), CV_8UC4, image->data());
            // Simulate principal point
            cvm = translateImg(cvm, depth_cam_.shift_x, depth_cam_.shift_y);
            const cv::Rect roi(depth_cam_.margin_x, depth_cam_.margin_y,
                               int(cvm.cols - (depth_cam_.margin_x*2)),
                               int(cvm.rows - (depth_cam_.margin_y*2)));
            cvm = cvm(roi);
            std::vector<cv::Mat> splits;
            cv::split(cvm, splits);
            cv::Mat cvm_f(image->GetHeight(), image->GetWidth(), CV_32FC1);
            cv::Mat R, G, B;
            splits[2].convertTo(R, CV_32FC1);
            splits[1].convertTo(G, CV_32FC1);
            splits[0].convertTo(B, CV_32FC1);
            cvm_f = ((R + G * 256.0 + B * (256.0 * 256.0)) / (256.0 * 256.0 * 256.0 - 1.0)) * 1000.0;
        });
    }
    YAML::Node semseg_cam_node = base["camera-semseg"];
    if (semseg_cam_node.IsDefined()) {
        std::cout << "setting up front camera" << std::endl;
        auto cam_blueprint = *bp_library->Find(semseg_cam_node["type"].as<std::string>());
        // setting camera attributes from the yaml
        const float cx = semseg_cam_node["cx"].as<float>();
        const float cy = semseg_cam_node["cy"].as<float>();
        const int sx = semseg_cam_node["image_size_x"].as<int>();
        const int sy = semseg_cam_node["image_size_y"].as<int>();
        const float shift_x = cx - (sx/2.);
        const float shift_y = cy - (sy/2.);
        const int margin_x = roundUp(std::abs(int(shift_x)), 16);
        const int margin_y = roundUp(std::abs(int(shift_y)), 16);
        for (YAML::const_iterator it = semseg_cam_node.begin(); it != semseg_cam_node.end(); ++it) {
            auto key = it->first.as<std::string>();
            if (cam_blueprint.ContainsAttribute(key))
                cam_blueprint.SetAttribute(key, it->second.as<std::string>());
        }
        // spawn the camera attached to the vehicle.
        cam_blueprint.SetAttribute("image_size_x", std::to_string(sx + margin_x*2));
        cam_blueprint.SetAttribute("image_size_y", std::to_string(sy + margin_y*2));
        semseg_cam_.margin_x = margin_x;
        semseg_cam_.margin_y = margin_y;
        semseg_cam_.shift_x = shift_x;
        semseg_cam_.shift_y = shift_y;
        cg::Transform camera_transform; // = LookupCameraExtrinsics("/base_link", FRONT_CAMERA_FRAME);
        auto generic_actor = vehicle_->GetWorld().SpawnActor(cam_blueprint, camera_transform, vehicle_.get());
        semseg_cam_.sensor = boost::static_pointer_cast<cc::Sensor>(generic_actor);
        // register a callback to save images to disk.
        semseg_cam_.sensor->Listen([this](boost::shared_ptr<carla::sensor::SensorData> data) {
            auto image = boost::static_pointer_cast<csd::Image>(data);
            cv::Mat cvm(image->GetHeight(), image->GetWidth(), CV_8UC4, image->data());
            cvm = translateImg(cvm, semseg_cam_.shift_x, semseg_cam_.shift_y);
            const cv::Rect roi(semseg_cam_.margin_x, semseg_cam_.margin_y, int(cvm.cols - (semseg_cam_.margin_x*2)), int(cvm.rows - (semseg_cam_.margin_y*2)));
            cvm = cvm(roi);
            std::vector<cv::Mat> splits;
            cv::split(cvm, splits);
        });
    }
}

/* stops the agent thread */
void CarlaAgent::StopAgentThread() {
    running_ = false;
    if (agent_thread_ && agent_thread_->joinable()) {
        agent_thread_->join();
        delete agent_thread_;
    }
}

/* destroys the agent in the simulation & its sensors. */
void CarlaAgent::DestroyAgent() {
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
