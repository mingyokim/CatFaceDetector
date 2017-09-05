#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "detector.hpp"

#include <boost/filesystem.hpp>
using namespace boost::filesystem;

using namespace std;

void help( char *argv[] );
void detectSingleImage( string src_path, string dst_path="" );
void detectMultipleImages( string src_path, string dst_path="" );
vector<path> getImagePathsInFolder( const path &folder, const string &ext );

int main( int argc, char *argv[] )
{
  if( argc < 3 )  help( argv );
  else
  {
    string function( argv[1] );
    if( !function.compare( "image" ) )
    {
      string src_path = argv[2];
      string dst_path="";
      if( argc > 3 )  dst_path = string( argv[3] );
      detectSingleImage( src_path, dst_path );
    }
    else if( !function.compare( "images" ) )
    {
      // cout << "This function is not implemented yet. Sorry!" << endl;
      string src_path = argv[2];
      string dst_path = "";
      if( argc > 3 )  dst_path = string( argv[3] );
      detectMultipleImages( src_path, dst_path );
      return 0;
    }
    else if( !function.compare( "video" ) )
    {
      cout << "This function is not implemented yet. Sorry!" << endl;
      return 0;
    }
    else if( !function.compare( "help" ) )
    {
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
      cv::namedWindow( "original", cv::WINDOW_NORMAL );
      cv::resizeWindow( "original", 800, 800 );
      cv::imshow( "original", org );

      cv::namedWindow( "detection", cv::WINDOW_NORMAL );
      cv::resizeWindow( "detection", 800, 800 );
      cv::imshow( "detection", image );
      cv::waitKey(0);
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
