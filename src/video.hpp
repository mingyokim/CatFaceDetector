//
// Created by min on 21/10/17.
//

#ifndef DETECTOR_VIDEO_HPP
#define DETECTOR_VIDEO_HPP

#include <iostream>
#include <string>
#include <opencv2/tracking.hpp>
#include <pthread.h>

#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "detector.hpp"

class Video {
    struct updateTrackingThreadArgs {
        cv::Ptr<cv::Tracker> tracker;
        cv::Rect *detection;
        bool isInit;
        cv::Mat *frame;
    };

    struct timeStats {
        double total_tracking_time;
        int total_tracking_frame_num;
    } time_stats;

    Detector detector;

    cv::VideoCapture cap;
    cv::VideoWriter writer;

    std::vector<cv::Rect> detections;
    std::vector<cv::Rect> detections_resized;

    std::vector<cv::Ptr<cv::Tracker> > trackers;
    bool isInit[NUM_FEATURES];

    bool show;
    float fps;
    int detection_rate;

    bool use_tracking;

public:
    //Constructor
    Video(){
        detector.loadModels( "models/cat_face.cfg", "models/cat_features.cfg" );
    }
    void processVideo(std::string src_path, std::string dst_path, bool use_tracking);
    static void *updateTrackingThread(void *ptr);

private:
    std::vector<cv::Rect> resizeDetections(std::vector<cv::Rect> &detections, float rx, float ry);
    void initTracking(cv::Mat &frame);
    void updateTracking(cv::Mat &frame);
    void processEachFrame();
};


#endif //DETECTOR_VIDEO_HPP
