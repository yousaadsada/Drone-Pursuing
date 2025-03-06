#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("opencv_node");

    RCLCPP_INFO(node->get_logger(), "OpenCV version: %s", CV_VERSION);

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}