#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include "transform_conversions.h"
// CARLA stuff
#include <carla/client/Map.h>
#include <carla/geom/Location.h>
#include <carla/client/Waypoint.h>
#include <carla/client/TimeoutException.h>
#include <carla/geom/Transform.h>
#include <carla/client/World.h>
#include <carla/client/Client.h>

namespace cc = carla::client;
using namespace std::chrono_literals;

std::unique_ptr<cc::Client> carla_client;

struct Controls {
    float delta_x = 0.0f;
    float delta_y = 0.0f;
    float delta_z = 0.0f;
    float delta_pitch = 0.0f;
    float delta_yaw = 0.0f;
    bool print = false;
}controls;

carla::SharedPtr<carla::client::Actor> InitCARLA(const std::string& town) {
    carla_client = std::make_unique<cc::Client>("127.0.0.1", 2000);
    carla_client->SetTimeout(10s);
    std::cout << "client version: " << carla_client->GetClientVersion() << "\t"
              << "server version: " << carla_client->GetServerVersion() << std::endl;
    carla_client->LoadWorld(town);
    auto world = carla_client->GetWorld();
    return world.GetSpectator();
}

/* joystick callback. messages are published using the official ROS joy node */
void JoystickCallback(const sensor_msgs::Joy::ConstPtr& joy) {
    static float axes_deadzone = 0.1;
    // L left: +1.0, right: -1.0
    if (std::abs(joy->axes[0]) > axes_deadzone) {
        controls.delta_yaw = -joy->axes[0];
    } else {
        controls.delta_yaw = 0;
    }
    // L up: +1.0, down: -1.0
    if (std::abs(joy->axes[1]) > axes_deadzone) {
        controls.delta_pitch = joy->axes[1];
    } else {
        controls.delta_pitch = 0;
    }
    // R left: +1.0, right: -1.0
    if (std::abs(joy->axes[3]) > axes_deadzone) {
        controls.delta_y = joy->axes[3];
    } else {
        controls.delta_y = 0;
    }
    // R up: +1.0, down: -1.0
    if (std::abs(joy->axes[4]) > axes_deadzone) {
        controls.delta_x = -joy->axes[4];
    } else {
        controls.delta_x = 0;
    }
    // LB: +1.0 to -1.0 -> 0 to -1
    // RB: +1.0 to -1.0 -> 0 to 1
    controls.delta_z = (-joy->axes[5] + 1 / 2) - (-joy->axes[2] + 1 / 2);

    /* A button */
    if (joy->buttons[0]) {
        controls.print = true;
    }
}

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "carla_explorer");
    ros::NodeHandle node_handle;
    std::cout << "joystick controls:\n"
              << "L: camera rotation\n"
              << "R: camera movement\n"
              << "A: print coordinate data\n" << std::endl;
    ros::Subscriber joy_sub = node_handle.subscribe<sensor_msgs::Joy>("joy", 10, JoystickCallback);
    // connect & change town
    auto spectator = InitCARLA("/Game/Carla/Maps/Town10HD");
    /*  Towns
        /Game/Carla/Maps/Town01 small, no exceptions
        /Game/Carla/Maps/Town02 small, no exceptions
        /Game/Carla/Maps/Town03 default
        /Game/Carla/Maps/Town04 large but mostly highway
        /Game/Carla/Maps/Town05 medium. has underpass
        /Game/Carla/Maps/Town06 suburb. no occlusions
        /Game/Carla/Maps/Town07 small farmhouse
        /Game/Carla/Maps/Town10HD small very urban
    */
    auto world = carla_client->GetWorld();
    auto map = world.GetMap();

    while(ros::ok()) {
        auto current_transform = spectator->GetTransform();
        if (controls.print) {
            controls.print = false;
            auto loc = current_transform.location;
            auto waypoint = map->GetWaypoint(loc);
            std::cout << "(" << loc.x << ", " << loc.y << ", " << loc.z << ") -------------------\n"
                      << "road id: " << waypoint->GetRoadId() << ", "
                      << "lane id: " << waypoint->GetLaneId() << ", "
                      << "section id: " << waypoint->GetSectionId() << ", "
                      << "junction id: " << waypoint->GetJunctionId() << std::endl;
        }
        current_transform.rotation.pitch += controls.delta_pitch;
        current_transform.rotation.yaw += controls.delta_yaw;
        current_transform.location.x += controls.delta_x;
        current_transform.location.y += controls.delta_y;
        current_transform.location.z += controls.delta_z;
        spectator->SetTransform(current_transform);
        ros::Duration(0.01).sleep();
        ros::spinOnce();
    }
    return 0;
}
