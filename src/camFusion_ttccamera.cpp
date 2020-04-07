
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iomanip> // std::setw

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;



// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera (std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // The objective is calculate the TTC using change
    //  in a 'characteristic' dimension between two images.
    // The characteristic dimension is the ratio of matched
    //  keypoint distances computed for each image
    // Outliers are to be removed
    
    double minLength = 100.0; // Using same value as in lesson #2
    std::vector<double> charDim;


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
    
    bool verbose = false;
    if (verbose) {
	cout << "\ncomputeTTCCamera:\n";
	cout << "Pre vec: Small=" << charDim[0] <<  " larg=" << charDim[count-1]
	     << " avg=" << avgRatio << " @ " << count << " median=" << median << endl;
	for (auto it=charDim.begin();it!=charDim.end(); it++) {
	    cout << *it << endl;
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
	
	int count = 0;  // Keep track of pruning effect
	float sumUp = 0.0;
	float smallest = 1E10;
	float largest = 0.0;
	
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

    // ...
}
