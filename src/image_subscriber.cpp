#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/String.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "b39vt_assignment/image_processing.hpp"

class ImageSubscriber
{
	// By default, variables and methods are private
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  
  ros::Publisher label_pub_;
  
  cv_bridge::CvImagePtr cv_ptr;
  
  //const std::string OPENCV_WINDOW;
  
public:
	// Public variables - it is a good practice to separate the declaration of
	// variables from the declaration/definition of methods
	bool data_valid;
  
public:
	// Public methods
  ImageSubscriber() : it_(nh_), data_valid(false)//, OPENCV_WINDOW("Image window")
  {
    // Subscribe to input video feed
    image_sub_ = it_.subscribe("/usb_cam/image_raw", 1, 
      &ImageSubscriber::imageCb, this);
    
    label_pub_ = nh_.advertise<std_msgs::String>("label", 1, this);

    //cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageSubscriber()
  {
    //cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      data_valid = true;
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    
    // Update GUI Window - uncomment if you want to visualize the subscribed
    // image...
    //cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    //cv::waitKey(3);
  }
  
  cv::Mat getImage()
  {
  	return cv_ptr->image;
  }
  
  void publish( std::string s )
  {
    std_msgs::String msg;
    msg.data = s;
    label_pub_.publish(msg);
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_subscriber");
  ImageSubscriber ic;
  
  cv::namedWindow("Templates");
  
  // Load templates
  std::vector<cv::Mat> templs = loadTemplates();
  
  /*for ( int i = 0; i < templs.size(); ++i )
  {
  	cv::imshow("Templates", templs[i]);
  	cv::waitKey(0);
  }*/
  
  ros::Rate r = 1;
  
  while (ros::ok())
  {
		if (ic.data_valid)
		{
			std::string label = templateMatching(ic.getImage(), templs);
			
			ic.publish(label);
		}
  	ros::spinOnce();
  	r.sleep();
  }
  return 0;
}
