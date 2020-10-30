#include "freicar_carla_proxy/freicar_carla_proxy.h"
#include "freicar_carla_proxy/agent_config.h"
#include "freicar_common/Track.h"

#include <visualization_msgs/Marker.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/transform_listener.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/String.h>
#include <ros/package.h>
#include <yaml-cpp/yaml.h>
#include <cmath>

#define TO_RAD M_PI / 180
#define TO_DEG 180 / M_PI
#define FIXED_FRAME "map"
#define FRONT_CAMERA_FRAME "/zed_camera"
using namespace std::chrono_literals;

namespace freicar {
namespace agent {
FreiCarCarlaProxy::FreiCarCarlaProxy(FreiCarCarlaProxy::Settings settings,
                                     std::shared_ptr<ros::NodeHandle> nh) {
    // initialize some stuff & things
    node_handle_ = nh;
    spectate_ = false;
    suspended_ = false;

    sync_trigger_rgb = false;
    sync_trigger_depth = false;
    sync_trigger_sem = false;

    carla_client_ = nullptr;
    agent_thread_ = nullptr;
    image_transport_ = nullptr;
    send_sync_stamp_ = ros::Time::now();
    settings_ = settings;

    if (settings_.sync_topic != "") {
        bool sync_mode = false;
        if (!ros::param::get("/sim_sync_mode", sync_mode)) {
            ROS_ERROR("ERROR: If any agent has the sync topic turned on, the simulator has to run in SYCHRONOUS MODE!!");
            std::exit(EXIT_FAILURE);

        }
        if (!sync_mode) {
            ROS_ERROR("ERROR: If any agent has the sync topic turned on, the simulator has to run in SYCHRONOUS MODE!!");
            std::exit(EXIT_FAILURE);
        }
    }
}

/* destructor */
FreiCarCarlaProxy::~FreiCarCarlaProxy() {
    // releasing thread resources
    StopAgentThread();
    DestroyCarlaAgent();
}

/* connects to airsim */
void FreiCarCarlaProxy::ActivateCarlaAgent(const std::string &address, unsigned int port, bool spectate) {
    try {
        carla_client_ = std::make_unique<cc::Client>(address, port);
        carla_client_->SetTimeout(10s);
        std::cout << "connected to CARLA sucessfully" << std::endl;
        std::cout << "client version: " << carla_client_->GetClientVersion() << '\t'
                  << "server version: " << carla_client_->GetServerVersion() << std::endl;
        auto world = carla_client_->GetWorld();
        auto blueprint_library = world.GetBlueprintLibrary();
        auto vehicles = blueprint_library->Filter(settings_.name);
        if (vehicles->empty()) {
            ROS_ERROR("ERROR: Did not find car model with name: %s , ... using default model: freicar_10 \n", settings_.name.c_str());
            vehicles = blueprint_library->Filter("freicar_10");
        }
        carla::client::ActorBlueprint blueprint = (*vehicles)[0];
        auto actor = world.TrySpawnActor(blueprint, cg::Transform(cg::Location(settings_.spawn_point)));
        if (!actor) {
            std::cout << "failed to spawn " << settings_.name << ". exiting..." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        vehicle_ = boost::static_pointer_cast<cc::Vehicle>(actor);
        std::cout << "spawned " << vehicle_->GetDisplayId() << std::endl;
        // turn off physics in mixed case
        if (settings_.type_code == type::SIMULATED)
            vehicle_->SetSimulatePhysics(true);
        else
            vehicle_->SetSimulatePhysics(false);
    } catch (const cc::TimeoutException &e) {
        ROS_ERROR("ERROR: connection to the CARLA server timed out\n%s", e.what());
        std::exit(EXIT_FAILURE);
    } catch (const std::exception &e) {
        ROS_ERROR("ERROR: %s", e.what());
        DestroyCarlaAgent();
        std::exit(EXIT_FAILURE);
    }
    tfListener_ = std::make_unique<tf2_ros::TransformListener>(tfBuffer_);
    if (settings_.spawn_sensors) {
        SetupSensors();
    }
    spectate_ = spectate;
}
/* sets up subscribers, publishers & runs thread */
void FreiCarCarlaProxy::SetupCallbacks() {
    // sim_only handles these topics here
    if (settings_.type_code == type::SIMULATED) {
        odometry_pub_ = node_handle_->advertise<nav_msgs::Odometry>("odometry", 10);
        halt_sub_ = node_handle_->subscribe<freicar_common::FreiCarHalt>("halt", 10,
                                                                &FreiCarCarlaProxy::HaltCallback, this);
        resume_sub_ = node_handle_->subscribe<std_msgs::String>("resume", 10,
                                                                &FreiCarCarlaProxy::ResumeCallback, this);
        control_sub_ = node_handle_->subscribe<raiscar_msgs::ControlCommand>("control", 10,
                                                                &FreiCarCarlaProxy::ControlCallback, this);
    }
    // start agent thread
    running_ = true;
    if (settings_.sync_topic != "") {
        std::cout << "Syncing to: " << settings_.sync_topic << std::endl;
        sync_sub_ = node_handle_->subscribe<sensor_msgs::CameraInfo>(settings_.sync_topic, 10,
                                                                    &FreiCarCarlaProxy::SyncCallback, this);
    } else {
        std::cout << "Syncing to topic inactive ... " << std::endl;
        agent_thread_ = new std::thread(&FreiCarCarlaProxy::Step, this, settings_.thread_sleep_ms);
    }
    std::cout << settings_.name << ": carla proxy thread started" << std::endl;
}

/* thread function for FreiCarCarlaProxy */
void FreiCarCarlaProxy::Step(unsigned int thread_sleep_ms) {
    geometry_msgs::TransformStamped current_pose;
    nav_msgs::Odometry odom;
    odom.header.frame_id = FIXED_FRAME;
    odom.child_frame_id = settings_.tf_name;
    // sim only
    if (settings_.type_code == type::SIMULATED) {
        current_pose.header.frame_id = FIXED_FRAME;
        current_pose.child_frame_id = settings_.tf_name;
        cg::Transform carla_transform;
        // TF & odom
        tf2::Quaternion q_carla_rot;
        tf2::Vector3 v_carla_vel;
        tf2::Transform t_carla_rot;
        tf2_ros::TransformBroadcaster tf_broadcaster;
        while (running_) {
            carla_transform = vehicle_->GetTransform();
            // updating/publishing current tf2 transform
            current_pose.transform.translation.x = odom.pose.pose.position.x = carla_transform.location.x;
            current_pose.transform.translation.y = odom.pose.pose.position.y = -carla_transform.location.y;
            current_pose.transform.translation.z = odom.pose.pose.position.z = carla_transform.location.z;
            q_carla_rot.setRPY(carla_transform.rotation.roll * TO_RAD,
                              -carla_transform.rotation.pitch * TO_RAD,
                              -carla_transform.rotation.yaw * TO_RAD);
            current_pose.transform.rotation.w = odom.pose.pose.orientation.w = q_carla_rot.w();
            current_pose.transform.rotation.x = odom.pose.pose.orientation.x = q_carla_rot.x();
            current_pose.transform.rotation.y = odom.pose.pose.orientation.y = q_carla_rot.y();
            current_pose.transform.rotation.z = odom.pose.pose.orientation.z = q_carla_rot.z();
            odom.header.stamp = current_pose.header.stamp = ros::Time::now();
            tf_broadcaster.sendTransform(current_pose);
            // converting carla's global odometry to local frame && publishing
            auto angular = vehicle_->GetAngularVelocity();
            auto velocity = vehicle_->GetVelocity();
            t_carla_rot.setIdentity();
            t_carla_rot.setRotation(q_carla_rot);
            v_carla_vel.setValue(velocity.x, velocity.y, velocity.z);
            v_carla_vel = t_carla_rot * v_carla_vel;
            odom.twist.twist.linear.x = v_carla_vel.x();
            odom.twist.twist.linear.y = v_carla_vel.y();
            odom.twist.twist.linear.z = v_carla_vel.z();
            odom.twist.twist.angular.x = angular.x * TO_RAD;
            odom.twist.twist.angular.y = angular.y * TO_RAD;
            odom.twist.twist.angular.z = -angular.z * TO_RAD;
            odometry_pub_.publish(odom);

            std::this_thread::sleep_for(std::chrono::milliseconds(thread_sleep_ms));
        }
        // mixed case
    } else {
        tf2::Quaternion temp_qtr;
        tf2::Matrix3x3 mat;
        double roll, pitch, yaw;
        while (running_) {
            // updating current transform
            try {
                current_pose = tfBuffer_.lookupTransform(FIXED_FRAME, settings_.tf_name + "/base_link", ros::Time(0));
            } catch (tf2::TransformStorage &ex) {
                ROS_WARN("WARNING: got some buffer error while looking up %s w.r.t. map",
                            (settings_.tf_name + "/base_link").c_str());
                ros::Duration(0.2).sleep();
            } catch (tf2::LookupException &ex) {
                ROS_WARN("WARNING: could not find transform %s w.r.t. map",
                            (settings_.tf_name + "/base_link").c_str());
                ros::Duration(0.2).sleep();
            } catch (tf2::ExtrapolationException &ex) {
                ROS_WARN("WARNING: got some extrapolation error while looking up %s w.r.t. map",
                            (settings_.tf_name + "/base_link").c_str());
                ros::Duration(0.2).sleep();
            }

            temp_qtr.setW(current_pose.transform.rotation.w);
            temp_qtr.setX(current_pose.transform.rotation.x);
            temp_qtr.setY(current_pose.transform.rotation.y);
            temp_qtr.setZ(current_pose.transform.rotation.z);
            mat.setRotation(temp_qtr);
            mat.getEulerYPR(yaw, pitch, roll);
            // sending the transform to carla
            vehicle_->SetTransform(cg::Transform(cg::Location(current_pose.transform.translation.x,
                                                             -current_pose.transform.translation.y,
                                                              current_pose.transform.translation.z + settings_.height_offset),
                                                    cg::Rotation(-pitch * TO_DEG, -yaw * TO_DEG, roll * TO_DEG)));

            std::this_thread::sleep_for(std::chrono::milliseconds(thread_sleep_ms));
        }
    }
}


/* Computes the extrinsics from a frame to another camera-frame(z-front, x-right) in carlas coordinate system (x-front, y-right) */
cg::Transform FreiCarCarlaProxy::LookupCameraExtrinsics(std::string from, std::string to) {
    geometry_msgs::TransformStamped base_cam_transform;
    try {
        base_cam_transform = tfBuffer_.lookupTransform(settings_.tf_name + from,
                                                       settings_.tf_name + to,
                                                       ros::Time(0),
                                                       ros::Duration(3.0));
    } catch (...) {
        ROS_ERROR("could not lookup transform %s w.r.t. %s", (settings_.tf_name + to).c_str(), (settings_.tf_name + from).c_str());
        DestroyCarlaAgent();
        std::exit(EXIT_FAILURE);
    }
    tf2::Stamped<tf2::Transform> base_cam_transform_tf;
    tf2::convert(base_cam_transform, base_cam_transform_tf);
    double base_cam_transform_roll, base_cam_transform_pitch, base_cam_transform_yaw;
    tf2::Matrix3x3(base_cam_transform_tf.getRotation()).getRPY(base_cam_transform_roll,
                                                                base_cam_transform_pitch,
                                                                base_cam_transform_yaw);
    tf2::Vector3 base_cam_transform_t = base_cam_transform_tf.getOrigin();

    // In Carla Y is pointing to the right in the local coordinate (car) coordinate system. In Freicar Y is pointing left.
    base_cam_transform_t.setY(-base_cam_transform_t.getY());
    base_cam_transform_yaw += M_PI / 2;
    base_cam_transform_roll += M_PI / 2;
    //std::cout << "Roll: " << base_cam_transform_roll << " Pitch: " << base_cam_transform_pitch << " Yaw: " << base_cam_transform_yaw << std::endl;

    // spawn the camera attached to the vehicle.
    auto camera_transform = cg::Transform{
            cg::Location{static_cast<float>(base_cam_transform_t.getX()),
                         static_cast<float>(base_cam_transform_t.getY()),
                         static_cast<float>(base_cam_transform_t.getZ())},
            // In Freicar the transform is given in the camera coordinate system convention (z -front). In Carla it's in the normal local coordinate frame.
            cg::Rotation{static_cast<float>(base_cam_transform_roll * TO_DEG),
                         static_cast<float>((-base_cam_transform_yaw) * TO_DEG),
                         static_cast<float>((base_cam_transform_pitch) * TO_DEG)}};
    // std::cout << "Pitch: " << static_cast<float>(base_cam_transform_roll * TO_DEG) << std::endl;
    // std::cout << "Yaw: " << static_cast<float>(base_cam_transform_yaw * TO_DEG) << std::endl;
    // std::cout << "Roll: " << static_cast<float>(base_cam_transform_pitch * TO_DEG) << std::endl;
    return camera_transform;
}

/* Computes the extrinsics from a frame to another lidar-frame(x-front, y-left) in carlas coordinate system (x-front, y-right) */
cg::Transform FreiCarCarlaProxy::LookupLidarExtrinsics(std::string from, std::string to) {

    geometry_msgs::TransformStamped base_cam_transform = tfBuffer_.lookupTransform(settings_.tf_name + from,
                                                                                   settings_.tf_name + to,
                                                                                   ros::Time(0),
                                                                                   ros::Duration(3.0));
    tf2::Stamped<tf2::Transform> base_cam_transform_tf;
    tf2::convert(base_cam_transform, base_cam_transform_tf);
    double base_cam_transform_roll, base_cam_transform_pitch, base_cam_transform_yaw;
    tf2::Matrix3x3(base_cam_transform_tf.getRotation()).getRPY(base_cam_transform_roll,
                                                                base_cam_transform_pitch,
                                                                base_cam_transform_yaw);
    tf2::Vector3 base_cam_transform_t = base_cam_transform_tf.getOrigin();

    // In Carla Y is pointing to the right in the local coordinate (car) coordinate system. In Freicar Y is pointing left.
    base_cam_transform_t.setY(-base_cam_transform_t.getY());
    base_cam_transform_t.setX(base_cam_transform_t.getX());

    // spawn the camera attached to the vehicle.
    auto camera_transform = cg::Transform{
            cg::Location{static_cast<float>(base_cam_transform_t.getX()),
                        -static_cast<float>(base_cam_transform_t.getY()),
                        static_cast<float>(base_cam_transform_t.getZ())},

            cg::Rotation{static_cast<float>(-base_cam_transform_pitch * TO_DEG),
                        static_cast<float>((base_cam_transform_yaw) * TO_DEG),
                        static_cast<float>((base_cam_transform_roll) * TO_DEG)}};

    return camera_transform;
}
/* halt topic callback */
void FreiCarCarlaProxy::HaltCallback(const freicar_common::FreiCarHalt::ConstPtr &msg) {
    if (msg->target_agent == settings_.name) {
        if (!suspended_) {
            std::cout << "suspended..." << std::endl;
        }
        suspended_ = true;
    }
}

/* control callback function */
    void FreiCarCarlaProxy::ControlCallback(const raiscar_msgs::ControlCommand::ConstPtr& control_command) {
        // Apply control to vehicle.
        static cc::Vehicle::Control control;
        control.manual_gear_shift = false;

        if (!suspended_) {
            control.hand_brake = false;
            control.throttle = control_command->throttle;
            control.brake = 0.0f;
            if (control_command->throttle >= 0.0f) {
                control.reverse = false;
            } else {
                control.reverse = true;
            }
            // else {
            //     if (vehicle_->GetVelocity().x == 0.0f) {
            //         control.reverse = true;
            //         control.throttle = control_command->throttle;
            //         control.brake = 0.0f;
            //     } else {
            //         control.throttle = 0.0f;
            //         control.brake = settings_.brake_multiplier * std::fabs(control_command->throttle);
            //     }
            // }
            control.steer = -control_command->steering;
        } else {
            control.throttle = 0.0f;
            control.hand_brake = true;
        }
        vehicle_->ApplyControl(control);
    }

/* sync msg topic callback */
void FreiCarCarlaProxy::SyncCallback(const sensor_msgs::CameraInfo::ConstPtr &msg) {
    geometry_msgs::TransformStamped current_pose;
    tf2::Quaternion temp_qtr;
    tf2::Matrix3x3 mat;
    double roll, pitch, yaw;

    if (!(msg->header.stamp > send_sync_stamp_)) {
        return;
    }
    send_sync_stamp_ = msg->header.stamp;
    sync_trigger_rgb = true;
    sync_trigger_sem = true;
    sync_trigger_sem = true;

    if (running_) {
        // updating current transform
        try {
            current_pose = tfBuffer_.lookupTransform(FIXED_FRAME, settings_.tf_name + "/base_link",
                                                        msg->header.stamp);
        } catch (tf2::TransformStorage &ex) {
            ROS_WARN("WARNING: got some buffer error while looking up %s w.r.t. map",
                        (settings_.tf_name + "/base_link").c_str());
            ros::Duration(0.2).sleep();
        } catch (tf2::LookupException &ex) {
            ROS_WARN("WARNING: could not find transform %s w.r.t. map", (settings_.tf_name + "/base_link").c_str());
            ros::Duration(0.2).sleep();
        } catch (tf2::ExtrapolationException &ex) {
            ROS_WARN("WARNING: got some extrapolation error while looking up %s w.r.t. map",
                        (settings_.tf_name + "/base_link").c_str());
            ros::Duration(0.2).sleep();
        }
        temp_qtr.setW(current_pose.transform.rotation.w);
        temp_qtr.setX(current_pose.transform.rotation.x);
        temp_qtr.setY(current_pose.transform.rotation.y);
        temp_qtr.setZ(current_pose.transform.rotation.z);
        mat.setRotation(temp_qtr);
        mat.getEulerYPR(yaw, pitch, roll);
        // sending the transform to carla
        vehicle_->SetTransform(cg::Transform(cg::Location(current_pose.transform.translation.x,
                                                         -current_pose.transform.translation.y,
                                                          current_pose.transform.translation.z +
                                                          settings_.height_offset),
                                                cg::Rotation(-pitch * TO_DEG, -yaw * TO_DEG, roll * TO_DEG)));
        carla_client_->GetWorld().Tick(10s);
        // std::cout << "Tick" << std::endl;
    }
}

/* resume topic callback */
void FreiCarCarlaProxy::ResumeCallback(const std_msgs::String::ConstPtr &msg) {
    if (msg->data == settings_.name) {
        if (suspended_)
            std::cout << "suspension lifted ..." << std::endl;
        suspended_ = false;
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
void FreiCarCarlaProxy::SetupSensors() {
    image_transport_ = std::make_unique<image_transport::ImageTransport>(*node_handle_);

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
        front_cam_.rgb_publisher = image_transport_->advertise("/sim/camera/rgb/front/image", 1);
        front_cam_.info_publisher = node_handle_->advertise<sensor_msgs::CameraInfo>(
                "/sim/camera/rgb/front/image/camera_info", 1);
        // usual camera info stuff
        front_cam_.cam_info.header.frame_id = settings_.name + "_sim_rgb_front_info";
        front_cam_.header.frame_id = settings_.name + "_sim_rgb_front";
        front_cam_.cam_info.header.seq = front_cam_.header.seq = 0;

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
        cg::Transform camera_transform = LookupCameraExtrinsics("/base_link", FRONT_CAMERA_FRAME);
        auto generic_actor = vehicle_->GetWorld().SpawnActor(cam_blueprint, camera_transform, vehicle_.get());
        front_cam_.sensor = boost::static_pointer_cast<cc::Sensor>(generic_actor);
        // register a callback to publish images
        front_cam_.sensor->Listen([this](boost::shared_ptr<carla::sensor::SensorData> data) {
            if (settings_.sync_topic != "") {
                if (sync_trigger_rgb) {
                    front_cam_.cam_info.header.stamp = front_cam_.header.stamp = send_sync_stamp_;
                    sync_trigger_rgb = false;
                } else {
                    return;
                }
            } else {
                if (settings_.type_code == type::REAL) {
                    front_cam_.cam_info.header.stamp = front_cam_.header.stamp =
                            ros::Time::now() - ros::Duration(settings_.sim_delay);
                } else {
                    front_cam_.cam_info.header.stamp = front_cam_.header.stamp =
                            ros::Time::now();
                }
            }

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

            front_cam_.cam_info.width = front_cam_.carla_image.cols;
            front_cam_.cam_info.height = front_cam_.carla_image.rows;
            sensor_msgs::ImagePtr rgb_msg = cv_bridge::CvImage(front_cam_.header, "rgb8",
                                                                front_cam_.carla_image).toImageMsg();
            front_cam_.rgb_publisher.publish(rgb_msg);
            front_cam_.info_publisher.publish(front_cam_.cam_info);
        });
    }
    // set up lidar
    YAML::Node lidar_node = base["lidar"];
    if (lidar_node.IsDefined()) {
        std::cout << "setting up lidar" << std::endl;
        auto lidar_blueprint = *bp_library->Find(lidar_node["type"].as<std::string>());
        // setting lidar attributes from the yaml
        for (YAML::const_iterator it = lidar_node.begin(); it != lidar_node.end(); ++it) {
            auto key = it->first.as<std::string>();
            if (lidar_blueprint.ContainsAttribute(key))
                lidar_blueprint.SetAttribute(key, it->second.as<std::string>());
        }
        // spawn the lidar sensor attached to the vehicle.
        cg::Transform lidar_transform = LookupLidarExtrinsics("/base_link", "/lidar_link");
        auto lidar_actor = vehicle_->GetWorld().SpawnActor(lidar_blueprint, lidar_transform, vehicle_.get());
        lidar_.sensor = boost::static_pointer_cast<cc::Sensor>(lidar_actor);
        lidar_.publisher = node_handle_->advertise<CarlaRGBPointCloud>("/sim/lidar", 20);
        // register a callback to publish point cloud
        lidar_.sensor->Listen([this](boost::shared_ptr<carla::sensor::SensorData> data) {
            auto pointcloud = boost::static_pointer_cast<csd::LidarMeasurement>(data);
            auto pc_size = pointcloud->size();
            lidar_.pcl_message.header.frame_id = settings_.tf_name + "/lidar_link";
            lidar_.pcl_message.height = pointcloud->GetChannelCount();
            lidar_.pcl_message.width = pc_size / lidar_.pcl_message.height;
            lidar_.pcl_message.points.reserve(pc_size);
            pcl::PointXYZRGB point;
            for (size_t i = 0; i < pc_size; ++i) {
                float fr = static_cast<float>(i) / static_cast<float>(pc_size);
                point.r = fr * 255;
                point.g = 255 - fr * 255;
                point.b = 18 + fr * 20;
                point.x = -pointcloud->at(i).y;
                point.y = -pointcloud->at(i).x;
                point.z = -pointcloud->at(i).z;
                lidar_.pcl_message.points.push_back(point);
            }
            ros::Time stamp_lidar;
            if (settings_.type_code == type::REAL) {
                stamp_lidar =
                        ros::Time::now() - ros::Duration(settings_.sim_delay);
            } else {
                stamp_lidar =
                        ros::Time::now();
            }
            pcl_conversions::toPCL(stamp_lidar, lidar_.pcl_message.header.stamp);
            lidar_.publisher.publish(lidar_.pcl_message);
            lidar_.pcl_message.points.clear();
        });
    }
    YAML::Node depth_cam_node = base["camera-depth"];
    if (depth_cam_node.IsDefined()) {
        std::cout << "setting up front camera" << std::endl;
        depth_cam_.rgb_publisher = image_transport_->advertise("/sim/camera/depth/front/image", 1);
        depth_cam_.float_publisher = image_transport_->advertise("/sim/camera/depth/front/image_float", 1);
        depth_cam_.info_publisher = node_handle_->advertise<sensor_msgs::CameraInfo>(
                "/sim/camera/depth/front/image/camera_info", 1);
        // usual camera info stuff
        depth_cam_.cam_info.header.frame_id = settings_.name + "_sim_depth_front_info";
        depth_cam_.header.frame_id = settings_.name + "_sim_depth_front";
        depth_cam_.cam_info.header.seq = depth_cam_.header.seq = 0;

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
        cg::Transform camera_transform = LookupCameraExtrinsics("/base_link", FRONT_CAMERA_FRAME);
        auto generic_actor = vehicle_->GetWorld().SpawnActor(cam_blueprint, camera_transform, vehicle_.get());
        depth_cam_.sensor = boost::static_pointer_cast<cc::Sensor>(generic_actor);
        // register a callback to publish images
        depth_cam_.sensor->Listen([this](boost::shared_ptr<carla::sensor::SensorData> data) {
            if (settings_.sync_topic != "") {
                if (sync_trigger_depth) {
                    depth_cam_.cam_info.header.stamp = depth_cam_.header.stamp = send_sync_stamp_;
                    sync_trigger_sem = false;
                } else {
                    return;
                }
            } else {
                if (settings_.type_code == type::REAL) {
                    depth_cam_.cam_info.header.stamp = depth_cam_.header.stamp =
                            ros::Time::now() - ros::Duration(settings_.sim_delay);
                } else {
                    depth_cam_.cam_info.header.stamp = depth_cam_.header.stamp = ros::Time::now();
                }
            }
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
            // auto image_view = carla::image::ImageView::MakeColorConvertedView(carla::image::ImageView::MakeView(*image),
            // 																  carla::image::ColorConverter::LogarithmicDepth());
            // // auto image_view = carla::image::ImageView::MakeView(*image);
            // auto grayscale_view = boost::gil::color_converted_view<boost::gil::gray8_pixel_t>(image_view);
            // typedef decltype(grayscale_view)::value_type pixel;
            // static_assert(sizeof(pixel) == 1, "single channel");
            // pixel raw_data[grayscale_view.width() * grayscale_view.height()];
            // boost::gil::copy_pixels(grayscale_view, boost::gil::interleaved_view(grayscale_view.width(),
            // 																	 grayscale_view.height(),
            // 																	 raw_data,
            // 																	 grayscale_view.width() * sizeof(pixel)));
            // depth_cam_.carla_image = cv::Mat(grayscale_view.height(), grayscale_view.width(), CV_8UC1, raw_data);
            // sensor_msgs::ImagePtr rgb_msg = cv_bridge::CvImage(depth_cam_.header, "mono8", depth_cam_.carla_image).toImageMsg();
            // depth_cam_.rgb_publisher.publish(rgb_msg);
            depth_cam_.cam_info.width = image->GetWidth();
            depth_cam_.cam_info.height = image->GetHeight();
            depth_cam_.float_publisher.publish(
            cv_bridge::CvImage(depth_cam_.header, "32FC1", cvm_f).toImageMsg());
            depth_cam_.info_publisher.publish(depth_cam_.cam_info);
        });
    }
    YAML::Node semseg_cam_node = base["camera-semseg"];
    if (semseg_cam_node.IsDefined()) {
        std::cout << "setting up front camera" << std::endl;
        semseg_cam_.rgb_publisher = image_transport_->advertise("/sim/camera/semseg/front/image", 1);
        semseg_cam_.raw_publisher = image_transport_->advertise("/sim/camera/semseg/front/image_raw", 1);
        semseg_cam_.info_publisher = node_handle_->advertise<sensor_msgs::CameraInfo>(
            "/sim/camera/semseg/front/image/camera_info", 1);
        // usual camera info stuff
        semseg_cam_.cam_info.header.frame_id = settings_.name + "_sim_semseg_front_info";
        semseg_cam_.header.frame_id = settings_.name + "_sim_semseg_front";
        semseg_cam_.cam_info.header.seq = semseg_cam_.header.seq = 0;

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
        cg::Transform camera_transform = LookupCameraExtrinsics("/base_link", FRONT_CAMERA_FRAME);
        auto generic_actor = vehicle_->GetWorld().SpawnActor(cam_blueprint, camera_transform, vehicle_.get());
        semseg_cam_.sensor = boost::static_pointer_cast<cc::Sensor>(generic_actor);
        // register a callback to save images to disk.
        semseg_cam_.sensor->Listen([this](boost::shared_ptr<carla::sensor::SensorData> data) {
            if (settings_.sync_topic != "") {
                if (sync_trigger_sem) {
                    semseg_cam_.cam_info.header.stamp = semseg_cam_.header.stamp = send_sync_stamp_;
                    sync_trigger_sem = false;
                } else {
                    return;
                }
            } else {
                if (settings_.type_code == type::REAL) {
                    semseg_cam_.cam_info.header.stamp = semseg_cam_.header.stamp =
                            ros::Time::now() - ros::Duration(settings_.sim_delay);
                } else {
                    semseg_cam_.cam_info.header.stamp = semseg_cam_.header.stamp =
                            ros::Time::now();
                }
            }
            auto image = boost::static_pointer_cast<csd::Image>(data);
            cv::Mat cvm(image->GetHeight(), image->GetWidth(), CV_8UC4, image->data());
            cvm = translateImg(cvm, semseg_cam_.shift_x, semseg_cam_.shift_y);
            const cv::Rect roi(semseg_cam_.margin_x, semseg_cam_.margin_y, int(cvm.cols - (semseg_cam_.margin_x*2)), int(cvm.rows - (semseg_cam_.margin_y*2)));
            cvm = cvm(roi);
            std::vector<cv::Mat> splits;
            cv::split(cvm, splits);
            // auto image_view = carla::image::ImageView::MakeColorConvertedView(carla::image::ImageView::MakeView(*image),
            // 																  carla::image::ColorConverter::CityScapesPalette());
            // auto rgb_view = boost::gil::color_converted_view<boost::gil::rgb8_pixel_t>(image_view);
            // typedef decltype(rgb_view)::value_type pixel;
            // static_assert(sizeof(pixel) == 3, "R, G & B");
            // pixel raw_data[rgb_view.width() * rgb_view.height()];
            // boost::gil::copy_pixels(rgb_view, boost::gil::interleaved_view(rgb_view.width(),
            // 															   rgb_view.height(),
            // 															   raw_data,
            // 															   rgb_view.width() * sizeof(pixel)));
            // semseg_cam_.carla_image = cv::Mat(rgb_view.height(), rgb_view.width(), CV_8UC3, raw_data);
            // sensor_msgs::ImagePtr rgb_msg = cv_bridge::CvImage(semseg_cam_.header, "rgb8", semseg_cam_.carla_image).toImageMsg();
            // semseg_cam_.rgb_publisher.publish(rgb_msg);
            semseg_cam_.cam_info.width = semseg_cam_.carla_image.cols;
            semseg_cam_.cam_info.height = semseg_cam_.carla_image.rows;
            semseg_cam_.info_publisher.publish(semseg_cam_.cam_info);
            semseg_cam_.raw_publisher.publish(cv_bridge::CvImage(semseg_cam_.header, "8UC1", splits[2]).toImageMsg());
        });
    }
}

/* stops the agent thread */
void FreiCarCarlaProxy::StopAgentThread() {
    running_ = false;
    if (agent_thread_ && agent_thread_->joinable()) {
        agent_thread_->join();
        delete agent_thread_;
    }
}

/* destroys all the agents in the simulation & their sensors. usually called before the program
   is about to exit with a ROS_ERROR */
void FreiCarCarlaProxy::DestroyCarlaAgent() {
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
    if (lidar_.sensor) {
        lidar_.sensor->Destroy();
        lidar_.sensor = nullptr;
    }
    if (vehicle_) {
        vehicle_->Destroy();
        vehicle_ = nullptr;
    }
    if (carla_client_) {
        carla_client_.reset();
    }
    if (image_transport_)
        image_transport_.reset();
}

/* updates the spectator view in the simulator. this is a debug feature but could be useful */
void FreiCarCarlaProxy::UpdateSimSpectator(cg::Transform target_transform) {
    static cg::Transform spec_transform;
    spec_transform.rotation.pitch = -20.0f;
    target_transform.location -= 2.0f * target_transform.GetForwardVector();
    target_transform.location.z += 1.0f;
    spec_transform.location = config::kSpectatorCamSmoothFactor * spec_transform.location +
                              (1 - config::kSpectatorCamSmoothFactor) * target_transform.location;
    // angle correction (jumps between 180 <-> -180)
    auto ang_diff = spec_transform.rotation.yaw - target_transform.rotation.yaw;
    if (std::abs(ang_diff) > 200.0f) {
        ang_diff = (ang_diff < 0) ? ang_diff + 360.0f : ang_diff - 360.0f;
        spec_transform.rotation.yaw = config::kSpectatorCamSmoothFactor * ang_diff + target_transform.rotation.yaw;
        spec_transform.rotation.yaw = (spec_transform.rotation.yaw >= 180.0f) ? spec_transform.rotation.yaw - 360.0f :
                                      (spec_transform.rotation.yaw <= -180.0f) ? spec_transform.rotation.yaw + 360.0f :
                                       spec_transform.rotation.yaw;
    } else
        spec_transform.rotation.yaw = config::kSpectatorCamSmoothFactor * ang_diff + target_transform.rotation.yaw;
    carla_client_->GetWorld().GetSpectator()->SetTransform(spec_transform);
}

} // namespace agent
} // namespace freicar
