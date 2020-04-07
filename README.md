# Estimating TTC with Camera Image and Lidar data

<img src="pngs.d/3DTrack.png" />


## Final Project of Camera Section
This write-up reports tasks that were performed in completion of the final project. The objective is to use camera images and lidar scan data to estimate time to collision (TTC) of a host vehicle ("ego car") from a sequence of synchronized lidar scans and rgb images. Investigations into lidar TTC estimate error sources, and performance of various keypoint matching methods used in camera based TTC estimation are also performed.

### Coding

The following software creation tasks were given and completed:

A. Match 3D Objects;
B. Compute lidar based TTC;
C. Associate matched keypoints to bounding boxes;
D. Compute camera based TTC.

These tasks are found in files in the 'src' directory. There is a separate file for each task to better isolate and review them.These files were broken out from the camFusion_Student.cpp file. The CMakeLists.txt file has been accordingly updated.

The source compiles and executes except that 'yolo' config files needed to run the application must be provided. They were too large for thefree repository on gitub.






