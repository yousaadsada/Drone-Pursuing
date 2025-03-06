#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <filesystem>
#include "std_msgs/msg/string.hpp"

class MarkerGeneratorNode : public rclcpp::Node
{
public:
    MarkerGeneratorNode() : Node("marker_generator_node")
    {
        // Define the directory to save the marker
        std::string dir = "/home/yousa/anafi_simulation/data/aruco";
        std::filesystem::create_directories(dir);  // Create the directory if it doesn't exist

        // Generate the ArUco marker
        cv::Mat markerImage;
        cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        cv::aruco::generateImageMarker(dictionary, 23, 200, markerImage, 1);

        // Save the image in the specified directory
        std::string filename = dir + "/marker23.png";
        cv::imwrite(filename, markerImage);

        RCLCPP_INFO(this->get_logger(), "ArUco marker saved to %s", filename.c_str());
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MarkerGeneratorNode>());
    rclcpp::shutdown();
    return 0;
}