#include "detector.hpp"
#include "video.hpp"

#include <opencv2/tracking.hpp>
#include <pthread.h>

#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
using namespace boost::filesystem;

//struct updateTrackingThreadArgs {
//  cv::Ptr<cv::Tracker> tracker;
//  cv::Rect *detection;
//  bool isInit;
//  cv::Mat *frame;
//};

class Function
{

public:
  static void detectSingleImage( std::string src_path, std::string dst_path );
  static void detectMultipleImages( std::string src_path, std::string dst_path );
  static void detectVideo( std::string src_path, std::string dst_path, bool use_tracking );
  static std::vector<boost::filesystem::path> getImagePathsInFolder( const boost::filesystem::path &folder, const std::string &ext );
//  static void *updateTrackingThread(void *ptr);
};
