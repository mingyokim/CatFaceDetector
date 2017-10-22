//
// Created by min on 21/10/17.
//

#include "video.hpp"

void Video::processVideo( std::string src_path, std::string dst_path, bool track )
{
    show = dst_path.compare("") == 0 ? true : false;
    use_tracking = track;

    cap.open( src_path );

    if( !cap.isOpened() )
    {
        std::cout << "Cannot open " << src_path << std::endl;
        return;
    }

    double fps = cap.get( CV_CAP_PROP_FPS );

    std::cout << "Detection on video: " << src_path << std::endl;

    if( show )
    {
        cv::namedWindow( "video", cv::WINDOW_NORMAL );
        cv::resizeWindow( "video", 800, 800 );
    }
    else
    {
        std::string videoname = src_path.substr( src_path.find_last_of('/')+1 );
        std::string write_path = dst_path + "/" + videoname;

        std::cout << "Writing detection result to: " << write_path << std::endl;

        cv::Mat sample;
        cap >> sample;

        if( !sample.data )
        {
            std::cout << "ERROR! first frame is blank" << std::endl;
            return;
        }

        int codec = CV_FOURCC( 'M', 'J', 'P', 'G' );
        writer.open( write_path, codec, fps, sample.size(), true );

        if( !writer.isOpened() )
        {
            std::cout << "ERROR! could not open " << dst_path << " for writing" << std::endl;
            return;
        }
    }

    if( use_tracking ) {
        trackers = std::vector<cv::Ptr<cv::Tracker> >(NUM_FEATURES,cv::TrackerKCF::create());
        std::memset(isInit,false,NUM_FEATURES);
        time_stats.total_tracking_time = 0;
        time_stats.total_tracking_frame_num = 0;
    }

    detection_rate = fps;

    processEachFrame();
}

void Video::processEachFrame() {
    int frame_num = 0;

    while( true )
    {
        std::cout << "frame" << frame_num << std::endl;
        cv::Mat frame;
        cap.read(frame);

        if( !frame.data )  break;

        detections = std::vector<cv::Rect>(NUM_FEATURES, cv::Rect());
        detections_resized = std::vector<cv::Rect>(NUM_FEATURES, cv::Rect());

        // resize image for tracking
        cv::Mat frame_resized;
        cv::resize(frame, frame_resized, cv::Size(), 1, 1); // can adjust size here - so far just keep the original size

        float rx = (float)frame.cols / frame_resized.cols;
        float ry = (float)frame.rows / frame_resized.rows;

        if( use_tracking )
        {
            if( frame_num % detection_rate == 0 )
            {
                detections = detector.detect( frame );
                detections_resized = resizeDetections(detections, rx, ry);

                initTracking(frame_resized);
            }
            else
            {
                updateTracking(frame_resized);
                detections = resizeDetections(detections_resized, 1/rx, 1/ry);
            }
        }
        else
        {
            detections = detector.detect( frame );
        }

        Detector::drawDetections( frame, detections );

        if( show )
        {
            cv::imshow( "video", frame );
            cv::waitKey(1);
        }
        else
        {
            writer.write( frame );
        }

        frame_num++;
    }

    if( use_tracking ) {
        std::cout << "Average tracking time per frame: " << time_stats.total_tracking_time/time_stats.total_tracking_frame_num << " ms" << std::endl;
    }
}

void Video::updateTracking(cv::Mat &frame_resized) {
    boost::posix_time::ptime t1, t2;
    boost::posix_time::time_duration dur;

    t1 = boost::posix_time::microsec_clock::local_time();

    //pthread
    pthread_t pthreads[NUM_FEATURES-1];
    for( int i = FACE+1; i < trackers.size(); i++ ) {
        updateTrackingThreadArgs args;
        args.tracker = trackers[i];
        args.detection = &detections_resized[i];
        args.isInit = isInit[i];
        args.frame = &frame_resized;
        std::cout << "\tframe size: " << frame_resized.size << std::endl;
        struct updateTrackingThreadArgs *ptr = (struct updateTrackingThreadArgs *)calloc(1, sizeof( struct updateTrackingThreadArgs ) );
        *ptr = args;
        if( pthread_create(pthreads+i-1, NULL, updateTrackingThread, ptr)) {
            std::cout << "Thread creation failed!" << std::endl;
            return;
        }
    }

    for( int i = FACE+1; i < trackers.size(); i++ ) {
        if( pthread_join(pthreads[i-1],NULL) ) {
            std::cout << "Thread join failed!" << std::endl;
            return;
        }
    }

    // sequential
    // for( int i = FACE+1; i < trackers.size(); i++ )
    // {
    //   if( isInit[i] )
    //   {
    //     cv::Rect2d updated_rect_2d;
    //     trackers[i]->update( frame, updated_rect_2d );
    //     cv::Rect updated_rect( (int)updated_rect_2d.x, (int)updated_rect_2d.y, (int)updated_rect_2d.width, (int)updated_rect_2d.height );
    //     detections[i] = updated_rect;
    //   }
    // }

    t2 = boost::posix_time::microsec_clock::local_time();
    dur = t2 - t1;
    int dur_milli = dur.total_milliseconds();
    std::cout << "Tracking took: " << dur_milli << " ms" << std::endl;

    time_stats.total_tracking_time += dur_milli;
    time_stats.total_tracking_frame_num++;
}

void Video::initTracking(cv::Mat &frame_resized) {

    for( int i = FACE+1; i < trackers.size(); i++ )
    {
        if( isInit[i] )
        {
            trackers[i]->clear();
        }

        if( detections[i].area() > 0 )
        {
            trackers[i] = cv::TrackerMIL::create();
            trackers[i]->init(frame_resized, detections_resized[i]);
            isInit[i] = true;
        }
        else
        {
            isInit[i] = false;
        }
    }
}

std::vector<cv::Rect> Video::resizeDetections(std::vector<cv::Rect> &detections, float rx, float ry) {
    std::vector<cv::Rect> result = std::vector<cv::Rect>(NUM_FEATURES,cv::Rect());

    for( int i = 0; i < detections.size(); i++ ) {
        cv::Rect resized;
        resized.x = detections[i].x / rx;
        resized.y = detections[i].y / ry;
        resized.width = detections[i].width / rx;
        resized.height = detections[i].height / ry;
        result[i] = resized;
    }

    return result;
}

void *Video::updateTrackingThread(void *ptr) {
    updateTrackingThreadArgs args = *(struct updateTrackingThreadArgs *) ptr;

    if( args.isInit )
    {
        cv::Rect2d updated_rect_2d;
        args.tracker->update( *(args.frame), updated_rect_2d );
        cv::Rect updated_rect( (int)updated_rect_2d.x, (int)updated_rect_2d.y, (int)updated_rect_2d.width, (int)updated_rect_2d.height );
        *(args.detection) = updated_rect;
    }

    free(ptr);
    return NULL;
}
