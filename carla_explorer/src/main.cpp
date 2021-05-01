#include <ros/ros.h>
#include <sensor_msgs/Joy.h>

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
carla::SharedPtr<carla::client::Actor> spectator;

void InitCARLA(const std::string& town) {
    carla_client = std::make_unique<cc::Client>("127.0.0.1", 2000);
    carla_client->SetTimeout(2s);
    std::cout << "client version: " << carla_client->GetClientVersion() << "\t"
              << "server version: " << carla_client->GetServerVersion() << std::endl;
    carla_client->LoadWorld(town);
    auto world = carla_client->GetWorld();
    spectator = world.GetSpectator();
}

/* joystick callback. messages are published using the official ROS joy node */
void JoystickCallback(const sensor_msgs::Joy::ConstPtr& joy) {
    if (joy->axes[5] < 1.0) {
        // axes[5] at 1, goes from +1 to -1
        
    } else if (joy->axes[2] < 1.0) {
        // axes[2] at 1, goes from +1 to -1

    } else {

    }
    // axes[0] at 0, goes from +1 to -1
    if (joy->buttons[9]) {
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

    ros::spin();
    return 0;
}
