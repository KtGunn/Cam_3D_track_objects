
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iomanip> // std::setw

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;



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

    bool verbose = false;
    
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
    // These ideas didn't really work
    float factor = 0.1;    // I set pruning factor to '0.4'
    float minimumE = factor*avgEuclid;
    float maximumE = (1.0 + factor)*avgEuclid;

    // These rather arbitrary did, especially a low minimumE
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
