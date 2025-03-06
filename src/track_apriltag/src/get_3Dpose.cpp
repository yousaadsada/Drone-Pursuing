#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

class ImageSubscriber : public rclcpp::Node
{
public:
    ImageSubscriber()
        : Node("image_subscriber")
    {
        image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "anafi/frames", 10,
            std::bind(&ImageSubscriber::image_callback, this, std::placeholders::_1));
        
        pose_publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("aruco_pose", 10);
        
        camMatrix = (cv::Mat_<double>(3, 3) << 899.288082, 0.000000, 623.516603,
                                              0.000000, 902.894688, 375.501706,
                                              0.000000, 0.000000, 1.000000);
        distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
        distCoeffs = (cv::Mat_<double>(5, 1) << -0.130160, 0.177506, -0.001448, 0.000741, 0.000000);

        markerLength = 0.0523;
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv::Mat inputImage = cv_bridge::toCvCopy(msg, "bgr8")->image;

        std::cout << "inputImage size: " << inputImage.rows << "x" << inputImage.cols << std::endl;
        std::cout << "inputImage type: " << inputImage.type() << std::endl;

        // Check if the image is empty
        if (inputImage.empty()) {
            std::cerr << "Error: inputImage is empty." << std::endl;
            return;
        }

        // Check if the image type is as expected
        if (inputImage.type() == CV_8UC4) {
            cv::cvtColor(inputImage, inputImage, cv::COLOR_BGRA2BGR);
            std::cout << "Converted inputImage to CV_8UC3." << std::endl;
        }

        // Check if the image type is now correct
        if (inputImage.type() != CV_8UC3) {
            std::cerr << "Error: inputImage type is not CV_8UC3 after conversion." << std::endl;
            return;
        }

        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

        // Use default detector parameters
        cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
        cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        cv::aruco::ArucoDetector detector(dictionary, detectorParams);

        try {
            // Detect markers
            detector.detectMarkers(inputImage, markerCorners, markerIds, rejectedCandidates);
        } catch (const cv::Exception& e) {
            std::cerr << "Error during detectMarkers: " << e.what() << std::endl;
            return;
        }

        // Debugging: Check the sizes of markerCorners and markerIds
        std::cout << "Number of markers detected: " << markerIds.size() << std::endl;
        for (size_t i = 0; i < markerCorners.size(); i++) {
            std::cout << "markerCorners[" << i << "] size: " << markerCorners[i].size() << std::endl;
        }
        
        std::cout << "inputImage size: " << inputImage.rows << "x" << inputImage.cols << std::endl;
        
        detector.detectMarkers(inputImage, markerCorners, markerIds, rejectedCandidates);

        cv::Mat outputImage = inputImage.clone();
        cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);
    
        

        size_t nMarkers = markerCorners.size();
        std::vector<cv::Vec3d> rvecs(nMarkers), tvecs(nMarkers);

        cv::Mat objPoints(4, 1, CV_32FC3);
        objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength/2.f, markerLength/2.f, 0);
        objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength/2.f, markerLength/2.f, 0);
        objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength/2.f, -markerLength/2.f, 0);
        objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength/2.f, -markerLength/2.f, 0);    
    
        if(!markerIds.empty()){
            for (size_t i = 0; i < nMarkers; i++){
                cv::solvePnP(objPoints, markerCorners[i], camMatrix, distCoeffs, rvecs[i], tvecs[i]);

                geometry_msgs::msg::Pose pose_msg;
                pose_msg.position.x = tvecs[i][2];
                pose_msg.position.y = -tvecs[i][0];
                pose_msg.position.z = -tvecs[i][1];

                cv::Mat rotationMatrix;
                cv::Rodrigues(rvecs[i], rotationMatrix);
                cv::Matx33d rotMat(rotationMatrix);
                tf2::Matrix3x3 tf2RotMat(
                    rotMat(0, 0), rotMat(0, 1), rotMat(0, 2),
                    rotMat(1, 0), rotMat(1, 1), rotMat(1, 2),
                    rotMat(2, 0), rotMat(2, 1), rotMat(2, 2)
                );
                tf2::Quaternion tf2Quat;
                tf2RotMat.getRotation(tf2Quat);

                pose_msg.orientation.x = tf2Quat.x();
                pose_msg.orientation.y = tf2Quat.y();
                pose_msg.orientation.z = tf2Quat.z();
                pose_msg.orientation.w = tf2Quat.w();

                pose_publisher_->publish(pose_msg);

                cv::aruco::drawDetectedMarkers(inputImage, markerCorners, markerIds);
                cv::drawFrameAxes(inputImage, camMatrix, distCoeffs, rvecs[i], tvecs[i], markerLength * 1.5f, 2);
            }
        }else {
        std::cout << "No ArUco markers detected in the current frame." << std::endl;
        }

        cv::imshow("Detected ArUco Markers", inputImage);
        cv::waitKey(1);  // Corrected waitKey capitalization
    } 

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_publisher_;
    cv::Mat camMatrix, distCoeffs;
    float markerLength;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ImageSubscriber>();

    rclcpp::Rate rate(10);

    while(rclcpp::ok()) {
        rclcpp::spin_some(node);
        rate.sleep();
    }

    rclcpp::shutdown();

    return 0;
}
