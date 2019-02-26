# Extended Kalman Filter Project Starter Code
Self-Driving Car Engineer Nanodegree Program

In this project you will utilize a kalman filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. Passing the project requires obtaining RMSE values that are lower than the tolerance outlined in the project rubric. 

This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and install [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for either Linux or Mac systems. For windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. Please see the uWebSocketIO Starter Guide page in the classroom within the EKF Project lesson for the required version and installation scripts.

Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF

Tips for setting up your environment can be found in the classroom lesson for this project.

Note that the programs that need to be written to accomplish the project are src/FusionEKF.cpp, src/FusionEKF.h, kalman_filter.cpp, kalman_filter.h, tools.cpp, and tools.h

The program main.cpp has already been filled out, but feel free to modify it.

Here is the main protocol that main.cpp uses for uWebSocketIO in communicating with the simulator.

## Project basics 

In this project, we used C++ to write a program taking in radar and lidar data to track position using Extended Kalman Filters.

The code will make a prediction based on the sensor measurement and then update the expected position. See files in the 'src' folder for the primary C++ files making up this project.

We can use a Kalman filter in any place where we have uncertain information about some dynamic system, and It can make an educated guess about what the system is going to do next.


A Kalman filter is an optimal estimation algorithm used to estimate states of a system from indirect and uncertain measurements. A Kalman filter is only defined for linear systems. If you have a nonlinear system and want to estimate system states, you need to use a nonlinear state estimator.

Kalman filter is extensively used in estimating the car position using Lidar and Radar sensors. Measurement of these sensors are not accurate as they are subject to drift or noisy. Kalman filter can be used to fuse the measurement of these sensors to find the optimal estimation of the exact position.


## Kalman Filter Cycle

The Kalman filter represents our distributions by Gaussians and iterates on two main cycles: Prediction and Measurement update.

## Prediction
Let’s say we know an object’s current position and velocity , which we keep in the x variable. Now one second has passed. We can predict where the object will be one second later because we knew the object position and velocity one second ago; we’ll just assume the object kept going at the same velocity.


But maybe the object didn’t maintain the exact same velocity. Maybe the object changed direction, accelerated or decelerated. So when we predict the position one second later, our uncertainty increases.


x is the mean state vector. For an extended Kalman filter, the mean state vector contains information about the object’s position and velocity that you are tracking. It is called the “mean” state vector because position and velocity are represented by a gaussian distribution with mean x.
P is the state covariance matrix, which contains information about the uncertainty of the object’s position and velocity. You can think of it as containing standard deviations.
Q is the Process Covariance Matrix. It is a covariance matrix associated with the noise in states. I didn’t specify the equation to calculate the matrix here.
F is the Transition Matrix (the one that deals with time steps and constant velocities)
Measurement Update
We are going to obtain the data from two kind of measurements (sensors):

Laser Measurements
Radar Measurements

## What is Extended Kalman Filters?
Extended Kalman Filters(EKF) linearize the distribution around the mean of the current estimate and then use this linearization in the predict and update states of the Kalman Filter algorithm.

Our predicated state x is represented by gaussian distribution which is mapped to Nonlinear function(Radar measurement), then resulting function will not be a gaussian function.

EKF uses a method called First Order Taylor Expansion to linearise the function.



## Results

In two different simulated runs, my Extended Kalman Filter produces the below results. The x-position is shown as 'px', y-position as 'py', velocity in the x-direction is 'vx', while velocity in the y-direction is 'vy'. Residual error is calculated by mean squared error (MSE).


Test One

Input	MSE
px	   0.0974
py	   0.0855
vx	   0.4517
vy	   0.4404


Test Two

Input	MSE
px	   0.0726
py	   0.0965
vx	   0.4216
vy	   0.4932


## References 

1 - https://medium.com/@serrano_223/extended-kalman-filters-for-dummies-4168c68e2117

2 - https://towardsdatascience.com/the-unscented-kalman-filter-anything-ekf-can-do-i-can-do-it-better-ce7c773cf88d

3 - https://github.com/JunshengFu/tracking-with-Extended-Kalman-Filter
