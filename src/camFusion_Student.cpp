
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iomanip> // std::setw

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor,
			 cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

	// pointers to all bounding boxes which enclose the current Lidar point
        vector<vector<BoundingBox>::iterator> enclosingBoxes;

        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    ///////////////////////////////////////////////////////////////////////////////
    /// PREVIOUS lidar scan
    ///
    // Note: lidarPt={x,y,z,r}[double]
    //       x-forward, x-up, y-left

    std::sort (lidarPointsPrev.begin(), lidarPointsPrev.end(), [] (const LidarPoint& lp1, const LidarPoint& lp2) {
	    return (lp1.x < lp2.x);
	});

    int count = lidarPointsPrev.size();
    double spread = (lidarPointsPrev[count-1].x-lidarPointsPrev[0].x)/count; 

    bool verbose = true;
    if (verbose) {
	cout << endl << "computeTTCLidar:\n";
	std::cout << "Prev pre: Small=" << lidarPointsPrev[0].x << " Large=" << lidarPointsPrev[count-1].x
		  << " count=" << count << " spread=" << spread << endl;
    }
    
    double accepanceLimit = 2*spread;
    double xMinPrev = 1;
    for (int n=0; n<(count/2); n++) {
	xMinPrev = lidarPointsPrev[n].x;
	if ( fabs(xMinPrev - lidarPointsPrev[n+1].x) < accepanceLimit) {
	    break;
	}
    }

    if (verbose) {
	std::cout << "Prev post: Small=" << xMinPrev << std::endl;
    }


    ///////////////////////////////////////////////////////////////////////////////
    /// CURRENT lidar scan
    ///
    std::sort (lidarPointsCurr.begin(), lidarPointsCurr.end(), [] (const LidarPoint& lp1, const LidarPoint& lp2) {
	    return (lp1.x < lp2.x);
	});

    count = lidarPointsCurr.size();
    spread = (lidarPointsCurr[count-1].x-lidarPointsCurr[0].x)/count; 

    if (verbose) {
	cout << endl << "computeTTCLidar:\n";
	std::cout << "Curr pre: Small="<< lidarPointsCurr[0].x << " Large="  << lidarPointsCurr[count-1].x
		  << " count=" << count << " spread=" << spread << endl;
    }


    accepanceLimit = 2*spread;
    double xMinCurr = 1;
    for (int n=0; n<(count/2); n++) {
	xMinCurr = lidarPointsCurr[n].x;
	if ( fabs(xMinCurr - lidarPointsCurr[n+1].x) < accepanceLimit) {
	    break;
	}
    }

    if (verbose) {
	std::cout << "Curr post: Small=" << xMinCurr << std::endl;
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    /// TTC in seconds
    ///
    TTC = xMinCurr/(xMinPrev-xMinCurr)/frameRate;
    if (verbose) {
	cout << " Lidar TTC= " << TTC << endl;
    }
    return;
}


// Purpose: Return a vector of KeyPoints contained w/i the
// supplied Bounding Box
//
// Note: This function is not part of the required work
//       It was created for test & debug purposes
//
std::vector<cv::KeyPoint> checkBoundingBoxes(std::vector<cv::DMatch> &matches, DataFrame &currFrame)
{
    std::vector<cv::KeyPoint> currKpts = currFrame.keypoints;
    cout << "Count prevKpts=" <<  currKpts.size() <<  " Count matches=" << matches.size() << endl;
    
    // All keypoints that are determined to be w/i bboxes go into this vector
    std::vector<cv::KeyPoint> vKps;

    int clean = 0;
    int lapping = 0;
    int nobox = 0;
    
    // Loop over matches
    for (auto itM = matches.begin(); itM != matches.end(); itM++) {

	// Pickup the current match
	cv::DMatch match = *itM;

	int prevId = match.queryIdx;
	int currId = match.queryIdx;
	int imgIdx = match.imgIdx;

	// Pickup the current matched keypoint
	cv::KeyPoint crrKpt = currKpts[currId];
	cv::Point2f p2f = crrKpt.pt;

	std::map<int,int> mapBBs;
	
	bool inbox = false;
	for (auto itBB = currFrame.boundingBoxes.begin(); itBB != currFrame.boundingBoxes.end(); itBB++) {
	    BoundingBox bb = *itBB;

	    if (bb.roi.contains (p2f)) {
                // This keypoint was in a box
		inbox = true;
		bool overlap = false;

		for (auto itBB2 = currFrame.boundingBoxes.begin(); itBB2 != currFrame.boundingBoxes.end(); itBB2++) {
		    // Skip the same box
		    if (itBB2 == itBB ) { continue; }

		    BoundingBox bb2 = *itBB2;
		    if (bb2.roi.contains (p2f)) {
			overlap = true;
			lapping++;
			break;
		    }
		    
		}
		if ( !overlap ) {

		    // This one makes the cut
		    vKps.push_back (crrKpt);
		    clean++;
		}
		// Done
		break;
	    }
	}
	if (!inbox) {
	    nobox++;
	}
    }
    cout << "Clean=" << clean << " lappers=" << lapping << " no box=" << nobox << endl;
    return (vKps);
}

