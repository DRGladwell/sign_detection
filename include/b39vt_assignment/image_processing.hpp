#ifndef IMAGE_PROCESSING_B39VT_ASSIGNMENT_HPP
#define IMAGE_PROCESSING_B39VT_ASSIGNMENT_HPP

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <vector>
#include <string>
#include <stdio.h>

std::vector<cv::Mat> loadTemplates();

std::string templateMatching(const cv::Mat& im, const std::vector<cv::Mat>& templ);
// Your code goes here

# endif // IMAGE_PROCESSING_B39VT_ASSIGNMENT_HPP
