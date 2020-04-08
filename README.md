# Estimating TTC with Camera Image and Lidar data

<img src="pngs.d/3DTrack.png" />


## Final Project of Camera Section
This write-up reports tasks that were performed in completion of the final project. The objective is to use camera images and lidar scan data to estimate time to collision (TTC) of a host vehicle ("ego car") from a sequence of synchronized lidar scans and rgb images. Investigations into lidar TTC estimate error sources, and performance of various keypoint matching methods used in camera based TTC estimation are also performed.

### Coding

The following software creation tasks were given and completed:

1. Match 3D Objects;
2. Compute lidar based TTC;
3. Associate matched keypoints to bounding boxes;
4. Compute camera based TTC.

These tasks are found in files in the 'src' directory. There is a separate file for each task to better isolate and review them.These files were broken out from the camFusion_Student.cpp file. The CMakeLists.txt file has been accordingly updated.

The source compiles and executes except that 'yolo' config files needed to run the application must be provided. They were too large for the free repository on gitub.


### Performance of Lidar based TTC Estimates

<img src="pngs.d/TTC-Cam-Lidar.png" />

The image above shows estimates of TTC using both camera images and lidar. We see for the most part reasonable agreement between the two methods. But there are oddities in the mix. To get a getter take on the Lidar TTC estimate we can look at the distance to the leading vehicle over the sequence of scans. This is shown in the image below.

<img src="pngs.d/Lidar-TTC.png" />

Lidar measurements indicate that the distance to the preceding vehicle is steadily decreasing from 7.913m to 6.814m in 17 steps. This is an average decrease of 0.065m/step. We see this average decrease ('dx=') in the image but with numerous instances of deviation. The image below shows the span from first to last lidar scan.

<img src="pngs.d/OverHead-1-18.png" />

A factor in these variations is simply defining what is the location of the preceding vehicle to the ego car. Our working definition is the nearest reasonable lidar point. We look at the spread of lidar points in the forward direction and use it to eliminate outliers. The figure below shows that the spread varies a lot from scan to scan, from tight and narrow--we get a good estimate of distance to the vehicle--to broad, and our estimate of distance is affected and deteriorates.

<img src="pngs.d/Lidar-Narr-Brd.png" />

As a strategy going forward, it is just as important to weed out far outliers as outliers abnormally close. Estimating where the bulk of the lidar readings are is called for.

### Performance of Camera based TTC Estimates

<img src="pngs.d/TTCCam.png" />










