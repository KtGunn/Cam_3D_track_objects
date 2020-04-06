
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



// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI (BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // The objective is to complete
    //     boundingBox.kptMatches;
    // defined as
    //     std::vector<cv::DMatch> kptMatches;
    // This vector is the subset of all keypoint matches that belong to
    //     the bounding box in question
    // Outliers are to be removed
    
    std::vector<cv::DMatch> allmatches; // All possible matches, pre-pruning
    std::vector<float> euclidean; // List of euclidean distances of individual matches
    
    float sumLengths = 0.0f;   // Running sum of all euclidean distances
    float shortest = 1E10;   // Also note shortest and longest
    float longest = 0.0f;

    bool verbose = true;
    
    ///////////////////////////////////////////////////////////////////
    /// Cycle over all keypoint matches
    //
    for (auto it=kptMatches.begin(); it!= kptMatches.end(); it++) {
	
	cv::DMatch match = *it;
	int prevId = match.queryIdx; // IMPORTANT: query == previous !
	int currId = match.trainIdx; // IMPORTANT: train == current !
	int imgIdx = match.imgIdx;
	
	/////////////////////////////////////////////////////////////
	/// Keep those that are inside 'boundingBox'
	//
	cv::Point2f currPt = kptsCurr[currId].pt;
	if (boundingBox.roi.contains (currPt)) { 
	    
	    allmatches.push_back (*it);  // Inside the bounding box!
	    
	    ////////////////////////////////////////////////////////
	    /// Monitor the spread of match lengths
	    //
	    cv::Point2f prevPt = kptsPrev[prevId].pt;
	    float length = cv::norm (currPt - prevPt);
	    euclidean.push_back (length);  // Note the euclidean distance 
	    
	    sumLengths += length; // Maintain sum, min & max
	    if (length > longest ) { longest = length; }
	    if (length < shortest ) { shortest = length; }
	}
    }
    
    // Sanity check
    if (allmatches.size() != euclidean.size()) {
	exit (-1);
    }
    
    if (allmatches.size() == 0) {
	cout << "Fault in clusterKptMatchesWithROI!\n";
	exit(-1);
    }

    float avgEuclid = sumLengths/allmatches.size();
    
    if (verbose) {
	cout << "\nclusterKptMatchesWithROI:\n";
	cout << "AllMatches=" << kptMatches.size() << " possible=" << allmatches.size() << endl;
	cout << "Pre: longest=" << longest << " shortest=" << shortest << "  avg=" << avgEuclid << endl;
    }
    
    
    ///////////////////////////////////////////////////////////////////
    /// Prune away outliers
    //
    float factor = 0.1;    // I set pruning factor to '0.4'
    float minimumE = factor*avgEuclid;
    float maximumE = (1.0 + factor)*avgEuclid;

    // testing 
    minimumE = 0.005*avgEuclid;
    maximumE = 1.5*avgEuclid;

    shortest = 1E10; // For fun
    longest = 0.0f;
    float sum = 0.0;
    
    for (ushort n=0; n<allmatches.size(); n++) {
	float distance = euclidean[n];
	if (distance >= minimumE && distance <= maximumE) { // Within limits?
	    boundingBox.kptMatches.push_back (allmatches[n]); // Yes

	    sum += distance;
	    if (distance > longest ) { longest = distance; }
	    if (distance < shortest ) { shortest = distance; }
	}
    }

    if (verbose) {
	int survivors = boundingBox.kptMatches.size();
	cout << "Post: longest=" << longest << " shortest=" << shortest << "  avg="
	     << sum/survivors << " " << survivors << " survivors\n";
    }
    // ...
    return;
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera (std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // The objective is calculate the TTC using change
    //  in a 'characteristic' dimension between two images.
    // The characteristic dimension is the ratio of matched keypoints,
    //  computed for each image
    // Outliers are to be remved
    
    double minLength = 100.0; // Using same value as in lesson #2
    std::vector<double> charDim;

    double smallest = 1E10;
    double largest = 0.;
    double sum = 0.0;
    
    //////////////////////////////////////////////////////////////////////////
    /// Cycle over the keypoints for the object
    //
    for (auto itA=kptMatches.begin(); itA!=kptMatches.end();  itA++) {
	
	///////////////////////////////////////////////////
	/// Anchor
	//
	cv::KeyPoint kpCurr = kptsCurr.at (itA->trainIdx);
	cv::KeyPoint kpPrev = kptsPrev.at (itA->queryIdx);
	
	for (auto itB=kptMatches.begin()+1; itB!=kptMatches.end();  itB++) {
	    
	    //if (itB == itA) { continue; } // Skip the identical match
	    
	    ///////////////////////////////////////////////
	    /// Reference
	    //
	    cv::KeyPoint kpCL = kptsCurr.at (itB->trainIdx);
	    cv::KeyPoint kpPL = kptsPrev.at (itB->queryIdx);
	    
	    double lenCurr = cv::norm (kpCurr.pt - kpCL.pt);
	    double lenPrev = cv::norm (kpPrev.pt - kpPL.pt);
	    
	    // Guard against division-by-zero and beyond-precision numbers.
            if (lenPrev > std::numeric_limits<double>::epsilon() && lenCurr >= minLength) {
                double distRatio = lenCurr / lenPrev;
                charDim.push_back (distRatio);

		sum += distRatio;
		if (distRatio > largest) { largest = distRatio; }
		if (distRatio < smallest) { smallest = distRatio; }
	    }
	    
	}
    }

    // Sanity check
    if (charDim.size() == 0) {
	TTC = NAN;
	return;
    }

    // Statistics
    std::sort (charDim.begin(),charDim.end());
    double sumUp = std::accumulate (charDim.begin(),charDim.end(), 0.0);
    int count = charDim.size();
    double avgRatio = sumUp / count;
    double median = charDim[floor(charDim.size()/2)];
    
    bool verbose = true;
    if (verbose) {
	cout << "\ncomputeTTCCamera:\n";
	cout << "Pre vec: Small=" << charDim[0] <<  " larg=" << charDim[count-1]
	     << " avg=" << avgRatio << " @ " << count
	     << " median=" << median << endl;
	for (auto it=charDim.begin();it!=charDim.end(); it++) {
	    //cout << *it << endl;
	}
    }


    ///////////////////////////////////////////////////////////////////
    /// Prune away outliers
    //
    double referenceDistance;
    if (false) {

	//**********************************************************
	// VERY surprising result that 'median' is better
	// than an 'average'. Indicates (to me) that this method
	// of estimating TTC is 'touchy'

	float factor = 0.75;    // Acceptance range
	float minimumR = factor*avgRatio;
	float maximumR = (1.0 + factor)*avgRatio;
	
	count = 0;  // Keep track of pruning effect
	sumUp = 0.0;
	smallest = 1E10;
	largest = 0.0;
	
	for (auto it=charDim.begin();it!=charDim.end(); it++) {
	    double ratio = *it;
	    if (ratio <= maximumR && ratio >= minimumR) {
		sumUp += ratio;
		
		count++;
		if (ratio > largest) { largest = ratio; }
		if (ratio < smallest) { smallest = ratio; }
	    }
	}
	// Adjusted average ration w/outliers removed
	avgRatio = sumUp / count;
	referenceDistance = avgRatio;

    } else {
	// Adjust for an even count
	if ( (charDim.size() % 2) == 0 ) {
	    int m = floor(charDim.size())/2;
	    median = (charDim[m-1]+charDim[m+0])/2;
	}
	referenceDistance = median;
    }


    // Estimated time to collision (TTC)
    double dT = -1.0 / frameRate;
    TTC = dT / (1.0 - referenceDistance);

    if (verbose) {
	cout << "Post vec: Small=" << smallest <<  " larg=" << largest
	     << " avg=" << avgRatio << " @ " << count << " TTC=" << TTC
	     << " frame rate=" << frameRate << endl;
    }

    // ...
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

//===================================================

/*
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    for(auto i = prevFrame.boundingBoxes.begin(); i != prevFrame.boundingBoxes.end(); i++) {
	vector<vector<cv::DMatch>::iterator> outerLoop;
	
	/-
	Matched keypoint in the previous frame, 
	FInding all key points that are in the same bounding box
	-/
	
	for(auto j = matches.begin(); j != matches.end(); j++){
	    int indexPreviousKP = j->queryIdx; //using pointer operator here.
	    if(i->roi.contains(prevFrame.keypoints.at(indexPreviousKP).pt)) outerLoop.push_back(j);
	}
	
	//To find the corresponding pair of keypoints and bounding boxes here.
	
	multimap<int, int> multiMapping;
	for(auto k = outerLoop.begin(); k != outerLoop.end(); k++) {
	    //using pointer operator here.
	    int indexCurrentKP = (*k)->trainIdx;
	    for(auto l = currFrame.boundingBoxes.begin(); l != currFrame.boundingBoxes.end(); l++) {
		if(l->roi.contains(currFrame.keypoints.at(indexCurrentKP).pt)) {
		    int bbId = l->boxID;
		    multiMapping.insert(std::pair<int, int>(bbId, indexCurrentKP));
		}
	    }
	}
	
	//To find occurance of matched keypoints in current frame
	//BB.
	int threshold = 0, index = 10000;
	if(multiMapping.size() > 0)  {
	    for(auto m = multiMapping.begin(); m != multiMapping.end(); m++) {
		if(multiMapping.count(m->first) > threshold) {
		    threshold = multiMapping.count(m->first);
		    index = m->first;
		}
	    }
	    bbBestMatches.insert(pair<int, int>(i->boxID, index));
	}
    }
}
*/
//===================================================






//   For reference and to match 'previous'/'current' to 'Source'/'Ref'
//
/**/
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int,int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{

    /////////////////////////////////////////////////////////
    /// Call keypoints out to save some typing
    //
    std::vector<cv::KeyPoint> prevKpts = prevFrame.keypoints;
    std::vector<cv::KeyPoint> currKpts = currFrame.keypoints;

    bool verbose = false;
    if (verbose) {
	cout << "Count prevKpts=" << prevKpts.size() << " Count currKpts=" << currKpts.size()
	     <<  " Count matches=" << matches.size() << endl;
    }
    

    /////////////////////////////////////////////////////////////////////////
    ///**********************************************************************
    /// Loop over bounding boxes of Previous frame
    //
    for (auto itP=prevFrame.boundingBoxes.begin(); itP!=prevFrame.boundingBoxes.end(); itP++ ) {

	BoundingBox prevBox = (*itP);
	int prevBBIndex = prevBox.boxID;
	std::vector<cv::DMatch> prevBoxMatches; // Matches inside previous box

	////////////////////////////////////////////////////////////////
	/// Search all matches for having a Kpt in present box
	//
	for (auto itM=matches.begin(); itM!=matches.end(); itM++) {

	    cv::DMatch match = (*itM);
	    cv::KeyPoint prevKpt = prevKpts[match.queryIdx];

	    if (prevBox.roi.contains (prevKpt.pt)) {
		prevBoxMatches.push_back(match); // This match belongs
	    }
	}

	
	// All possible matches for previous box
	std::multimap<int,int> currBoxKpts; // multimap< <currentBoxID>,<currentKptID>>
	
	////////////////////////////////////////////////////////////////
	/// Search prevBoxMatches for counterpart in Current Frame
	//
	for (auto itBM=prevBoxMatches.begin(); itBM!=prevBoxMatches.end(); itBM++) {
	    
	    cv::DMatch match = (*itBM); // Get the current KeyPt from prev match
	    cv::KeyPoint currKpt = currKpts[match.trainIdx];

	    ///////////////////////////////////////////////////////////
	    /// Find the bounding box in Current Frame that's a match
	    //
	    for (auto itC=currFrame.boundingBoxes.begin(); itC!=currFrame.boundingBoxes.end(); itC++) {
		BoundingBox currBox = (*itC);
		
		if (currBox.roi.contains (currKpt.pt)) {
		    // We have a match
		    currBoxKpts.insert (std::pair<int,int>(itC->boxID,match.trainIdx));
		}
	    }
	}

	int maxCount = -1;
	int maxCountID = -1;
	if (currBoxKpts.size() > 0) {
	    for (auto i=currBoxKpts.begin(); i!=currBoxKpts.end(); i++) {
		// Is the count of present boxID higher than maxCount?
		if (currBoxKpts.count(i->first) > maxCount) {
		    // Yes, 
		    maxCount = currBoxKpts.count(i->first); // update max count
		    maxCountID = i->first; // associate ID w/max count
		}
	    }
	    bbBestMatches.insert (std::pair<int,int>(prevBBIndex,maxCountKpt));
	}

    }

    // Structure to hold bb matches and their populuations:
    //   map<pair<bbPrev, bbCurr>, count_of_kpts>
    //    pair<bbPrev, bbCurr> : indices of bounding boxes
    std::map<std::pair<int,int>,int> bbPopulation;
    

    /////////////////////////////////////////////////////////////////////////
    ///**********************************************************************
    /// Loop over matches
    //
    for (auto itM = matches.begin(); itM != matches.end(); itM++) {

	//////////////////////////////////////////////////////////////////////
	/// Pickup the current match
	cv::DMatch match = *itM;

	int prevId = match.queryIdx; // IMPORTANT: query == previous !
	int currId = match.trainIdx; // IMPORTANT: train == current !
	int imgIdx = match.imgIdx;

	//////////////////////////////////////////////////////////////////////
	/// Box for Current keypoint
	cv::KeyPoint crrKpt = currKpts[currId];
	cv::Point2f p2f = crrKpt.pt;

	bool inbox = false;
	int currBBindex = -1;
	for (auto itCur = currFrame.boundingBoxes.begin(); itCur != currFrame.boundingBoxes.end(); itCur++) {
	    BoundingBox bb = *itCur;

	    if (bb.roi.contains (p2f)) {
		if (!inbox) {
		    // First time found w/i a box!
		    inbox = true;
		    currBBindex = itCur - currFrame.boundingBoxes.begin();
		} else {
		    // Ooops! Keypoint in multiple boxes; rejecting this one;
		    currBBindex = -1;
		    break;
		}
	    }
	}
	if ( currBBindex == -1) {
	    // We got no good bounding box in the current frame
	    continue;
	}

	//////////////////////////////////////////////////////////////////////
	/// Box for Previous keypoint
	cv::KeyPoint prvKpt = prevKpts[prevId];
	cv::Point2f p2 = prvKpt.pt;

	inbox = false;
	int prevBBindex = -1;
	for (auto itPrv = prevFrame.boundingBoxes.begin(); itPrv != prevFrame.boundingBoxes.end(); itPrv++) {
	    BoundingBox bb = *itPrv;

	    if (bb.roi.contains (p2)) {
		if (!inbox) {
		    // This keypoint is in a box
		    inbox = true;
		    prevBBindex = itPrv - prevFrame.boundingBoxes.begin();
		} else {
		    // Multiple boxes occupancy! We are rejecting this one;
		    prevBBindex = -1;
		    break;
		}
	    }
	}
	if ( prevBBindex == -1) {
	    // We failed to find unique boxes for this match
	    continue;
	}
	
	/////////////////////////////////////////////////////////////////
	/// GOOD MATCH we insert it
	//                                                      pair<bbPrev, bbCurr>
	auto It = bbPopulation.insert (std::make_pair(std::make_pair(prevBBindex,currBBindex),1));
	if (!It.second) {
	    // The key already existed so we up the count
	    It.first->second += 1;
	}
    }

    if (bbPopulation.size() == 0) {
	cout << " ATTN! No Matching Bounding Boxes pre-pruning!\n";
    }
    
    // Let's see the matches
    if (verbose) {
	std::cout << "_UNIQUELY matched bounding boxes:\n";
	int countBBPrev = prevFrame.boundingBoxes.size();
	int countBBCurr = currFrame.boundingBoxes.size();
	std::cout << countBBPrev << " boxes in Prev, " << countBBCurr << " boxes in Curr\n";
    }
    
    // Selection criteria ; these are rather arbitrary
    int minCountKpts = 10;
    float maxRatio = 1.35;
    float minRatio = 0.65;
    
    maxRatio = 2;
    minRatio = 0.2;

    for (auto it=bbPopulation.begin(); it!=bbPopulation.end(); it++) {
	// Get the pair and count
	std::pair<int,int> bbPrvCurr = it->first;
	int count = it->second;
	
	// Get the areas
	cv::Rect bbPrev = prevFrame.boundingBoxes[bbPrvCurr.first].roi;
	float areaPrev = bbPrev.width*bbPrev.height;
	
	cv::Rect bbCurr = prevFrame.boundingBoxes[bbPrvCurr.second].roi;
	float areaCurr = bbCurr.width*bbCurr.height;
	
	float aRatio = areaCurr/areaPrev;
	
	if (verbose) {
	    std::cout << "(" << bbPrvCurr.second << ", " << bbPrvCurr.first << "), " << count << ": "
		      << areaCurr << "/" << areaPrev
		      << " ratio=" << std::setprecision(3) << aRatio << std::endl;
	}

	//////////////////////////////////////////////////////////////////////
	/// PICK out the keepers
	//
	if (aRatio < maxRatio && aRatio > minRatio && count > minCountKpts) {
	    bbBestMatches.insert (bbPrvCurr);
	}
    }
    
    if (bbBestMatches.size() == 0) {
	cout << " ATTN! No Matching Bounding Boxes Post-pruning!\n";
    } else {
	cout << "MBB: " << bbBestMatches.size() << " matching bounding boxes\n";
    }

    return;
}
/**/

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

