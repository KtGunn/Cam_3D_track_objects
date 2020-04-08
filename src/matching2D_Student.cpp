
#include <numeric>
#include "matching2D.hpp"


using namespace std;


// Find best matches for keypoints in two camera images based on several matching methods
//   For reference and to match 'previous'/'current' to 'Source'/'Ref'
//   matchDescriptors((dataBuffer.end() - 2)->keypoints,(dataBuffer.end() - 1)->keypoints, (dB.end()-2)->descs, (dBu.end()-1)->descriptors,
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
	// Bug in OpenCV! We must use floating point descriptors
	if (descSource.type() != CV_32F) {
	    descSource.convertTo (descSource, CV_32F);
	    descRef.convertTo (descRef, CV_32F);
	}
	matcher = cv::DescriptorMatcher::create (cv::DescriptorMatcher::FLANNBASED);
        // ...
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
	
	vector<vector<cv::DMatch>> knn_matches;
	matcher->knnMatch(descSource, descRef, knn_matches, 2); 
	
	// Ratio test
	double minDescDistRatio = 0.8;
	for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it) {
	    if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance) {
		// Push it into the returned matches
		matches.push_back((*it)[0]);
            }
        }
        // ...
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0) {
	
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
	
        extractor = cv::BRISK::create(threshold, octaves, patternScale);

    } else if (descriptorType.compare("AKAZE") == 0)
    {
	extractor = cv::AKAZE::create();
    } else if (descriptorType.compare("ORB") == 0)
    {
	extractor = cv::ORB::create();
    } else if (descriptorType.compare("BRISK") == 0)
    {
	extractor = cv::BRISK::create();
    } else if (descriptorType.compare("BRIEF") == 0)
    {
	extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    } else if (descriptorType.compare("FREAK") == 0)
    {
	extractor = cv::xfeatures2d::FREAK::create();
    } else if (descriptorType.compare("SIFT") == 0)
    {
	extractor = cv::xfeatures2d::SIFT::create();
    } else {
	
	cout << " descriptor type is unknown! \n";
	exit (0);
        //...
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    double ms = 1000.0 * t;
    cout << descriptorType << " descriptor extraction in " << ms << " ms" << endl;

    return (ms);
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       // size of an average block for computing a derivative
                             // covariation matrix over each pixel neighborhood

    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;
    
    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);
    
    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it) {
	
	cv::KeyPoint newKeyPoint;
	newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
	newKeyPoint.size = blockSize;
	keypoints.push_back(newKeyPoint);
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    double ms = 1000.0*t;
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return (ms);
}


// [KTG] Harris keypoint detector implementation
//
double detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
  // Detector parameters
  int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
  int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
  
  int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
  double k = 0.04;       // Harris parameter (see equation for details)
  
  double t = (double)cv::getTickCount();

  // Detect Harris corners and normalize output
  cv::Mat dst, dst_norm, dst_norm_scaled;
  dst = cv::Mat::zeros(img.size(), CV_32FC1);
  
  // 'img'=input, 'dst'=output
  cv::cornerHarris (img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
  
  // Scale-normalize the harris resonse matrix ; otuput remains CV_32F1
  cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  
  // Convert to CV_8U format
  cv::convertScaleAbs(dst_norm, dst_norm_scaled);
  
  // visualize results
  if (bVis)
    {
      std::string windowName = "Harris Corner Detector Response Matrix";
      cv::namedWindow(windowName, 4);
      cv::imshow(windowName, dst_norm_scaled);
      cv::waitKey(0);
    }
  
  double maxOverlap = 1.0; // 0->1kp; 1->299kp
  
  int cMin = 0;
  // Don't want to use the local copy
  //std::vector<cv::KeyPoint> vKeyPts;
  for (size_t r=0; r < dst_norm.rows; r++)
    {
      for (size_t c=0; c < dst_norm.cols; c++)
        {
          // Test the harris response
          int harrisValue = (int)dst_norm.at<float>(r,c);
          if (harrisValue > minResponse)
            {
              ++cMin;
              
              // Create a keypoint
              cv::KeyPoint newKp;
              newKp.pt = cv::Point2f(c,r); //NOTE: Point2f(x,y)!
              newKp.size = 2*apertureSize;
              newKp.response = harrisValue;
              
              // Test for overlap
              bool doesOverlap = false;
              if (true) {
                for (auto it=keypoints.begin(); it != keypoints.end() ;it++)
                  {
                    // If there's overlap
                    double overlap = cv::KeyPoint::overlap (newKp, *it);
                    if (overlap >= maxOverlap)
                      {
                        // There is overlap
                        doesOverlap = true;
                        if (newKp.response > (*it).response)
                          {
                            *it = newKp;
                            break;
                          }
                      }
                  }
              }
              // If there was no overlap, we add this point
              // Note, if there was overlap, we may have replaced an
              // exisiting key point
              if ( doesOverlap == false ) {
                keypoints.push_back (newKp);
              }
            }
        }
    }

  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
  // Now visualize the keypoints
  if (bVis)
    {
      std::string keyPointWindow ("Harris KeyPts");
      cv::namedWindow (keyPointWindow, 5);
      cv::Mat vizImage = dst_norm_scaled.clone();
      cv::drawKeypoints (dst_norm_scaled, keypoints, vizImage, cv::Scalar::all(-1),
                         cv::DrawMatchesFlags::DEFAULT);
      cv::imshow (keyPointWindow, vizImage);
      cv::waitKey (0);
    }
  
  return (1000.0*t);
}


/////////////////////////////////////////////////////////////////////////////////////
/// DETECTOR for modern descriptors, which are similar
//    ORB AKAZE SIFT FAST BRISK
//
double detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector;
    
    if ( detectorType.compare("ORB") == 0) {
	detector = cv::ORB::create();

    } else if ( detectorType.compare("AKAZE") == 0) {
	detector = cv::AKAZE::create();
	
    } else if ( detectorType.compare("SIFT") == 0) {
	detector = cv::xfeatures2d::SIFT::create();
	
    } else if ( detectorType.compare("FAST") == 0) {
	int threshold = 30; // intensity discriminator between target pixel and neighbors
	bool doNonMaxSuppress = true;
	cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
	detector = cv::FastFeatureDetector::create (threshold, doNonMaxSuppress, type);

    } else if ( detectorType.compare("BRISK") == 0) {
	detector = cv::BRISK::create();

    } else {
	cout << "Detector type '" << detectorType << "' is not recognized\n";
	exit (-1);
    }

    /////////////////////////////////////////////////////////////////////
    /// DETECT & Time
    //
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    double ms = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    ms *= 1000.0;
    
    if (bVis) {
	cv::Mat vizImage = img.clone ();
	cv::drawKeypoints (img, keypoints, vizImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow (detectorType, 2);
	cv::imshow (detectorType, vizImage);
	cv::waitKey (0);
    }

    return (ms);
}

