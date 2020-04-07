
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iomanip> // std::setw

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


void computeTTCLidar (std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    ///////////////////////////////////////////////////////////////////////////////
    /// PREVIOUS lidar scan
    ///
    // Note: lidarPt={x,y,z,r}[double]
    //       x-forward, z-up, y-left

    // The objective is to esimate TTC by comparing
    //   shortest distance
    // to an object in two consecutive scans.
    //
    // Given the time span between the two scans
    //   and
    // and assuming constant velocity
    //   the TTC can be estimates
    //
    // Outliers that may skew the TTC estimate are to be removed
    

    /////////////////////////////////////////////////////////////////////
    /// Process Previous scan by
    //
    // ..sorting the lidar points vector w.r.t x-coord
    std::sort (lidarPointsPrev.begin(), lidarPointsPrev.end(), [] (const LidarPoint& lp1, const LidarPoint& lp2) {
	    return (lp1.x < lp2.x);
	});
    
    // ..then calculating the Average Spread between lidar points
    int count = lidarPointsPrev.size();
    double spread = (lidarPointsPrev[count-1].x-lidarPointsPrev[0].x)/count; 

    bool verbose = false;
    if (verbose) {
	cout << endl << "computeTTCLidar:\n";
	std::cout << "Prev pre: Small=" << lidarPointsPrev[0].x << " Large=" << lidarPointsPrev[count-1].x
		  << " count=" << count << " spread=" << spread << endl;
    }
    
    /////////////////////////////////////////////////////////////////////
    /// Remove outliers by
    //   picking the first point that's w/i two times (arbitrary) the 
    //   average spread.
    //   This removes abnormally shortest lidar point(s)
    //
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


    /////////////////////////////////////////////////////////////////////
    /// Process Current scan by
    //
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


    /////////////////////////////////////////////////////////////////////
    /// Remove outliers by
    //
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
    if (1 || verbose) {
	cout << "Image Lidar TTC= " << TTC << " xMinC=" << xMinCurr << "  dx=" << (xMinPrev-xMinCurr) << endl;
    }
    return;
}

