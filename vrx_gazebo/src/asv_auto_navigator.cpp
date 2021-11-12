//
// Created by zihan on 10/1/21.
//

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>

class BoatAutoNavigator {
public:
    BoatAutoNavigator(ros::NodeHandle& nh, ros::NodeHandle& pnh): nh_(nh), pnh_(pnh), twist_(geometry_msgs::Twist()) {
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("cmd_vel", 10);
    }

    void navigate() {
        ros::Rate r(5);
        while (ros::ok()) {
            twist_.linear.x = 0.2;
            twist_.angular.z = 0;
            pub();
            r.sleep();
        }
    }

    void pub() {
        cmd_vel_pub_.publish(twist_);
    }

private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher cmd_vel_pub_;
    geometry_msgs::Twist twist_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "Boat Auto Navigator");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    BoatAutoNavigator ban(nh, pnh);
    ban.navigate();
}
