#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "function.hpp"

using namespace std;

void help( char *argv[] );

int main( int argc, char *argv[] )
{
  if( argc < 3 )  help( argv );
  else
  {
    string function( argv[1] );
    string src_path = argv[2];
    string dst_path="";
    if( argc > 3 )  dst_path = string( argv[3] );

    if( !function.compare( "image" ) )          Function::detectSingleImage( src_path, dst_path );
    else if( !function.compare( "images" ) )    Function::detectMultipleImages( src_path, dst_path );
    else if( !function.compare( "video" ) )
    {
      bool use_tracking = false;
      dst_path = "";
      if( argc > 3 )  use_tracking = atoi( argv[3] );
      if( argc > 4 )  dst_path = string( argv[4] );
      Function::detectVideo( src_path, dst_path, use_tracking );
    }
    else if( !function.compare( "help" ) )      help( argv );
    else
    {
      cout << "'" << function << "' is an invalid function!" << endl << endl;
      help( argv );
    }
  }

  return 0;
}

void help( char *argv[] )
{
  cout << "usage: " << string( argv[0] ) << " [function] [src_path] (video?)(tracking) (dst_path)" << endl << endl;
  cout << "function: ['image', 'images', 'video', 'help']" << endl;
  cout << "\timage: run detection on a single image" << endl;
  cout << "\timages: run detection on all images in a folder" << endl;
  cout << "\tvideo: run detection on a video" << endl;
  cout << "\thelp: display this message" << endl << endl;
  cout << "src_path: path to element(s) to run detection on, depending on function" << endl;
  cout << "\timage: path to a single image" << endl;
  cout << "\timages: path to a folder containing one or more images" <<  endl;
  cout << "\tvideo: path to a video" << endl << endl;
  cout << "tracking(for video only)[default = 0] if set to 1, detect less frequently and use tracking in between" << endl << endl;
  cout << "dst_path (optional): path to the folder in which you wish the program to write the detection results;" << endl;
  cout << "\tif not specified, the program will simply show the detection in a GUI" << endl << endl;
}
