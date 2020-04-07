
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iomanip> // std::setw

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int,int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // The objective is to return a 'map' relating
    // the ID's of bounding boxes of two frames
    // representing the same object
    // 
    // We will create a 2-dim matrix of box ID's
    // first index for previous, second for current.
    // Entries in the matrix represent number of occurrences
    // of matched keypoints occupying the indicated box IDs.
    //
    // For each 'row' i.e. previous frame boxes, we select the
    // 'column' i.e. current frame boxes with the most occurrences
    // or number of matching keypoints


    ////////////////////////////////////////////////////////////////////////
    /// Create and initialize/zero the Matrix
    //
    int cPrevBBs = prevFrame.boundingBoxes.size();
    int cCurrBBs = currFrame.boundingBoxes.size();

    std::vector<std::vector<int> > vPrCr;
    for (int p=0; p<cPrevBBs;p++) {
	std::vector<int> vC;
	for (int c=0; c<cCurrBBs;c++) {
	    vC.push_back(0);
	}
	vPrCr.push_back (vC);
    }

    /////////////////////////////////////////////////////////
    /// Call keypoints out to save some typing
    //
    std::vector<cv::KeyPoint> prevKpts = prevFrame.keypoints;
    std::vector<cv::KeyPoint> currKpts = currFrame.keypoints;


    //////////////////////////////////////////////////////////////////
    /// Cycle over all keypoint matches
    //
    for (auto itM=matches.begin();itM!=matches.end();itM++) {

	// Process the current match	
	cv::DMatch match = (*itM);
	cv::KeyPoint kpPrev = prevKpts[match.queryIdx];
	cv::KeyPoint kpCurr = currKpts[match.trainIdx];

	//////////////////////////////////////////////////////////////
	/// Cycle over previous frame's bounding boxes
	//  to find its keypoints (prev keypoints)
	//
	for (auto ibP=prevFrame.boundingBoxes.begin();ibP!=prevFrame.boundingBoxes.end();ibP++) {
	    if (ibP->roi.contains(kpPrev.pt) ) {
		int preBoxID = ibP->boxID; // Note the previous box ID
		
		//////////////////////////////////////////////////////////////
		/// Cycle over current frame's bounding boxes
		//  for a match
		//
		for (auto ibC=currFrame.boundingBoxes.begin();ibC!=currFrame.boundingBoxes.end();ibC++) {
		    if (ibC->roi.contains(kpCurr.pt)) {
			int curBoxID = ibC->boxID;
			vPrCr[preBoxID][curBoxID] += 1; // Up the count
		    }
		}

	    }
	}
    }

    ///////////////////////////////////////////////////////////////
    /// Cycle over the Matrix to pick out the maximum count
    /// for each row
    //
    for (auto itP=vPrCr.begin();itP!=vPrCr.end();itP++){
	
	int prevBoxID = itP-vPrCr.begin(); // Match this box ID
	
	int max = 0;  // Initialize the search
	int currBoxID = 0;
	std::vector<int> vC = (*itP);

	///////////////////////////////////////////////////////////
	/// Look for maximum keypoints count in the row of
	//  matched boxes
	//
	for (auto itC=vC.begin();itC!=vC.end();itC++){
	    int boxID = itC-vC.begin();

	    if ( (*itC) > max ) {
		max = (*itC);
		currBoxID = boxID; // Best match currently
	    }
	}

	// Best match
	bbBestMatches.insert (std::pair<int,int>(prevBoxID, currBoxID));
    }
    return;
}
