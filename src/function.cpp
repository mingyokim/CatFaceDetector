#include "function.hpp"

void Function::detectSingleImage( std::string src_path, std::string dst_path )
{
  bool show = dst_path.compare("") == 0 ? true : false;
  std::string imgname = src_path.substr( src_path.find_last_of('/')+1 );
  std::string write_path = dst_path + "/" + imgname;

  std::cout << "Detection on a single image: " << src_path << std::endl;

  if( show )    std::cout << "Showing detection result" << std::endl;
  else          std::cout << "Writing detection result to " << write_path << std::endl;

  Detector detector;
  detector.loadModels( "models/cat_face.cfg", "models/cat_features.cfg" );

  cv::Mat image = cv::imread( src_path );
  cv::Mat org = image.clone();
  std::vector<cv::Rect> detections = detector.detect( image );

  Detector::drawDetections( image, detections );

  if( show )
  {
    cv::namedWindow( "original", cv::WINDOW_NORMAL );
    cv::resizeWindow( "original", 800, 800 );
    cv::imshow( "original", org );

    cv::namedWindow( "detection", cv::WINDOW_NORMAL );
    cv::resizeWindow( "detection", 800, 800 );
    cv::imshow( "detection", image );
    cv::waitKey(0);
  }
  else
  {
    cv::imwrite( write_path, image );
  }
}

void Function::detectMultipleImages( std::string src_path, std::string dst_path )
{
  bool show = dst_path.compare("") == 0 ? true : false;

  std::cout << "Detection on multiple images in: " << src_path << std::endl;

  std::vector<boost::filesystem::path> paths = getImagePathsInFolder( src_path, ".jpg" );

  std::cout << "Found " << paths.size() << " images in the folder" << std::endl;

  Detector detector;
  detector.loadModels( "models/cat_face.cfg", "models/cat_features.cfg" );

  std::cout << std::endl << std::endl;

  for( std::vector<boost::filesystem::path>::iterator it = paths.begin(); it != paths.end(); ++it )
  {
    std::string imgpath = src_path + "/" + (*it).string();
    std::cout << "Reading from " << imgpath << std::endl;

    cv::Mat image = cv::imread( imgpath );
    cv::Mat org = image.clone();
    std::vector<cv::Rect> detections = detector.detect( image );

    Detector::drawDetections( image, detections );

    if( show )
    {
      cv::namedWindow( "original", cv::WINDOW_AUTOSIZE );
      cv::resizeWindow( "original", 800, 800 );
      cv::imshow( "original", org );

      cv::namedWindow( "detection", cv::WINDOW_AUTOSIZE );
      cv::resizeWindow( "detection", 800, 800 );
      cv::imshow( "detection", image );
      cv::waitKey(0);
    }
    else
    {
      std::string write_path = dst_path + "/" + (*it).string();
      cv::imwrite( write_path, image );
    }
  }
}

void Function::detectVideo( std::string src_path, std::string dst_path, bool use_tracking ) {
    Video video;
    video.processVideo(src_path, dst_path, use_tracking);
}

std::vector<boost::filesystem::path> Function::getImagePathsInFolder( const boost::filesystem::path &folder, const std::string &ext )
{
  std::vector<boost::filesystem::path>paths;

  if( !boost::filesystem::exists(folder) || !boost::filesystem::is_directory(folder) )  return paths;

  boost::filesystem::recursive_directory_iterator it(folder);
  boost::filesystem::recursive_directory_iterator endit;

  while( it != endit )
  {
    if( boost::filesystem::is_regular_file(*it) && it->path().extension() == ext ) paths.push_back( it->path().filename() );
    ++it;
  }

  return paths;
}
