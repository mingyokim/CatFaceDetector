#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "detector.hpp"

#include <boost/filesystem.hpp>
using namespace boost::filesystem;

using namespace std;

void help( char *argv[] );

//TODO make another class for these functions (probably "function.cpp")
void detectSingleImage( string src_path, string dst_path="" );
void detectMultipleImages( string src_path, string dst_path="" );
void detectVideo( string src_path, string dst_path="" );
vector<path> getImagePathsInFolder( const path &folder, const string &ext );

int main( int argc, char *argv[] )
{
  if( argc < 3 )  help( argv );
  else
  {
    string function( argv[1] );
    string src_path = argv[2];
    string dst_path="";
    if( argc > 3 )  dst_path = string( argv[3] );

    if( !function.compare( "image" ) )          detectSingleImage( src_path, dst_path );
    else if( !function.compare( "images" ) )    detectMultipleImages( src_path, dst_path );
    else if( !function.compare( "video" ) )     detectVideo( src_path, dst_path );
    else if( !function.compare( "help" ) )      help( argv );
    else
    {
      cout << "'" << function << "' is an invalid function!" << endl << endl;
      help( argv );
    }
  }

  return 0;
}

void detectSingleImage( string src_path, string dst_path )
{
  bool show = dst_path.compare("") == 0 ? true : false;
  string imgname = src_path.substr( src_path.find_last_of('/')+1 );
  string write_path = dst_path + "/" + imgname;

  cout << "Detection on a single image: " << src_path << endl;

  if( show )    cout << "Showing detection result" << endl;
  else          cout << "Writing detection result to " << write_path << endl;

  Detector detector;
  detector.loadModels( "models/cat_face.cfg", "models/cat_features.cfg" );

  cv::Mat image = cv::imread( src_path );
  cv::Mat org = image.clone();
  vector<cv::Rect> detections = detector.detect( image );

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

void detectMultipleImages( string src_path, string dst_path )
{
  bool show = dst_path.compare("") == 0 ? true : false;

  cout << "Detection on multiple images in: " << src_path << endl;

  vector<path> paths = getImagePathsInFolder( src_path, ".jpg" );

  cout << "Found " << paths.size() << " images in the folder" << endl;

  Detector detector;
  detector.loadModels( "models/cat_face.cfg", "models/cat_features.cfg" );

  cout << endl << endl;

  for( vector<path>::iterator it = paths.begin(); it != paths.end(); ++it )
  {
    string imgpath = src_path + "/" + (*it).string();
    cout << "Reading from " << imgpath << endl;

    cv::Mat image = cv::imread( imgpath );
    cv::Mat org = image.clone();
    vector<cv::Rect> detections = detector.detect( image );

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
      string write_path = dst_path + "/" + (*it).string();
      cv::imwrite( write_path, image );
    }
  }
}

void detectVideo( std::string src_path, std::string dst_path )
{
  bool show = dst_path.compare("") == 0 ? true : false;

  cv::VideoCapture cap( src_path );
  cv::VideoWriter writer;

  if( !cap.isOpened() )
  {
    std::cout << "Cannot open " << src_path << std::endl;
    return;
  }

  std::cout << "Detection on video: " << src_path << std::endl;

  Detector detector;
  detector.loadModels( "models/cat_face.cfg", "models/cat_features.cfg" );

  if( show )
  {
    cv::namedWindow( "video", cv::WINDOW_NORMAL );
    cv::resizeWindow( "video", 800, 800 );
  }
  else
  {
    string videoname = src_path.substr( src_path.find_last_of('/')+1 );
    string write_path = dst_path + "/" + videoname;

    std::cout << "Writing detection result to: " << write_path << std::endl;

    cv::Mat sample;
    cap >> sample;

    if( !sample.data )
    {
      std::cout << "ERROR! first frame is blank" << std::endl;
      return;
    }

    int codec = CV_FOURCC( 'M', 'J', 'P', 'G' );
    double fps = 25.0;
    writer.open( write_path, codec, fps, sample.size(), true );

    if( !writer.isOpened() )
    {
      std::cout << "ERROR! could not open " << dst_path << " for writing" << std::endl;
      return;
    }
  }

  while( true )
  {
    cv::Mat frame;
    cap >> frame;

    if( !frame.data )  break;

    vector<cv::Rect> detections = detector.detect( frame );
    Detector::drawDetections( frame, detections );

    if( show )
    {
      cv::imshow( "video", frame );
      cv::waitKey(5);
    }
    else
    {
      writer.write( frame );
    }
  }
}

vector<path> getImagePathsInFolder( const path &folder, const string &ext )
{
  vector<path> paths;

  if( !exists(folder) || !is_directory(folder) )  return paths;

  recursive_directory_iterator it(folder);
  recursive_directory_iterator endit;

  while( it != endit )
  {
    if( is_regular_file(*it) && it->path().extension() == ext ) paths.push_back( it->path().filename() );
    ++it;
  }

  return paths;
}

void help( char *argv[] )
{
  cout << "usage: " << string( argv[0] ) << " [function]" << " [src_path]" << " (dst_path)" << endl << endl;
  cout << "function: ['image', 'images', 'video', 'help']" << endl;
  cout << "\timage: run detection on a single image" << endl;
  cout << "\timages: run detection on all images in a folder" << endl;
  cout << "\tvideo: run detection on a video" << endl;
  cout << "\thelp: display this message" << endl << endl;
  cout << "src_path: path to element(s) to run detection on, depending on function" << endl;
  cout << "\timage: path to a single image" << endl;
  cout << "\timages: path to a folder containing one or more images" <<  endl;
  cout << "\tvideo: path to a video" << endl << endl;
  cout << "dst_path (optional): path to the folder in which you wish the program to write the detection results;" << endl;
  cout << "\tif not specified, the program will simply show the detection in a GUI" << endl << endl;
}
