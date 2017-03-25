#include "b39vt_assignment/image_processing.hpp"

std::vector<cv::Mat> loadTemplates()
{
	std::vector<cv::Mat> templates;
	
	boost::filesystem::path images_path("src/b39vt_assignment/data/");
	boost::filesystem::directory_iterator end;

	for ( boost::filesystem::directory_iterator it(images_path); it != end; ++it )
		templates.push_back(cv::imread(it->path().string()));
	
	return templates;
}

void templateMatching(const cv::Mat& im, const std::vector<cv::Mat>& templ)
{
	// Your code goes here
	std::vector<int> num_matches;
	
	for ( int i = 0; i < templ.size(); ++i )
	{
		cv::Mat img_1 = im;
		cv::Mat img_2 = templ[i];
		
		cv::resize(img_2, img_2, cv::Size(200, 200));
	
		//-- Step 1: Detect the keypoints
		//cv::OrbFeatureDetector detector(25, 1.0f, 2, 10, 0, 2, 0, 10);
		cv::OrbFeatureDetector detector;
		//cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("BRISK");
	
		std::vector<cv::KeyPoint> keypoints_1, keypoints_2;

		detector.detect( img_1, keypoints_1 );
		detector.detect( img_2, keypoints_2 );

		//-- Step 2: Calculate descriptors (feature vectors)
		cv::OrbDescriptorExtractor extractor;
		//cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("BRISK");
	
		cv::Mat descriptors_1, descriptors_2;

		extractor.compute( img_1, keypoints_1, descriptors_1 );
		extractor.compute( img_2, keypoints_2, descriptors_2 );
		
		printf("-- keypoints_1 size = %d\n", keypoints_1.size());
		printf("-- keypoints_2 size = %d\n", keypoints_2.size());
		printf("-- descriptors_1 size = %d x %d\n", descriptors_1.size().height, descriptors_1.size().width);
		printf("-- descriptors_2 size = %d x %d\n", descriptors_2.size().height, descriptors_2.size().width);
		
		if (descriptors_1.type() != CV_32F)
			descriptors_1.convertTo(descriptors_1, CV_32F);
		
		if (descriptors_2.type() != CV_32F)
			descriptors_2.convertTo(descriptors_2, CV_32F);

		//-- Step 3: Matching descriptor vectors using FLANN matcher
		cv::FlannBasedMatcher matcher;
		std::vector< cv::DMatch > matches;
		matcher.match( descriptors_1, descriptors_2, matches );

		double max_dist = 0; double min_dist = 100;

		//-- Quick calculation of max and min distances between keypoints
		for( int i = 0; i < descriptors_1.rows; i++ )
		{ double dist = matches[i].distance;
		  if( dist < min_dist ) min_dist = dist;
		  if( dist > max_dist ) max_dist = dist;
		}

		printf("-- Max dist : %f \n", max_dist );
		printf("-- Min dist : %f \n", min_dist );

		//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
		//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
		//-- small)
		//-- PS.- radiusMatch can also be used here.
		std::vector< cv::DMatch > good_matches;

		for( int i = 0; i < descriptors_1.rows; i++ )
		{ //printf("Distance = %f\n", matches[i].distance);
			if( matches[i].distance <= std::max(2*min_dist, 0.02) )
			//if( matches[i].distance <= std::max(3*min_dist, 0.1) )
		  { good_matches.push_back( matches[i]); }
		}
		
		num_matches.push_back(good_matches.size());

		//-- Draw only "good" matches
		cv::Mat img_matches;
		drawMatches( img_1, keypoints_1, img_2, keypoints_2,
		             good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
		             std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

		//-- Show detected matches
		imshow( "Good Matches", img_matches );

		for( int i = 0; i < (int)good_matches.size(); i++ )
		{ printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }

		cv::waitKey(0);
	}
	
	int match = distance(num_matches.begin(), std::max_element(num_matches.begin(), num_matches.end()));
	
	switch (match)
	{
		case 0:
			printf("\nThe image is 'DEAD MAN'\n\n");
			break;
		case 1:
			printf("\nThe image is 'TOXIC'\n\n");
			break;
		case 2:
			printf("\nThe image is 'ALIVE MAN'\n\n");
			break;
		case 3:
			printf("\nThe image is 'RADIOACTIVE'\n\n");
			break;
		case 4:
			printf("\nThe image is 'FIRE'\n\n");
			break;
		case 5:
			printf("\nThe image is 'DANGER'\n\n");
			break;
		case 6:
			printf("\nThe image is 'BIOHAZARD'\n\n");
			break;
		case 7:
			printf("\nThe image is 'NO SMOKING'\n\n");
			break;
	}
}
