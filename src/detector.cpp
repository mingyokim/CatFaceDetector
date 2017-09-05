#include "detector.hpp"

void Detector::loadModels( std::string path_to_face_model, std::string path_to_features_model )
{
  std::string path_to_face_weights = "models/cat_face.weights";
  _face_detector.loadModel( path_to_face_model, path_to_face_weights );

  std::string path_to_features_weights = "models/cat_features.weights";
  _feature_detector.loadModel( path_to_features_model, path_to_features_weights );
}

std::vector<cv::Rect> Detector::detect( cv::Mat image )
{
  std::vector<cv::Rect> features;
  features.push_back(detectFace( image ));

  if( features[FACE].area() == 0 )  return features;

  detectFeatures( image, features );

  return features;
}

cv::Rect Detector::detectFace( cv::Mat image )
{
  std::vector<Darknet::Detection> detections = _face_detector.detect( image );

  if( detections.size() == 0 )  return cv::Rect();

  int max_index = 0;
  float max_prob = 0;
  for( int i = 0; i < detections.size(); i++ )
  {
    if( detections[i].prob > max_prob )
    {
      max_prob = detections[i].prob;
      max_index = i;
    }
  }

  return detections[max_index].rect;
}

void Detector::detectFeatures( cv::Mat image, std::vector<cv::Rect> &features )
{
  cv::Rect bigger_face = enlargeRect( image, features[FACE], 0.1, 0.1 );
  cv::Mat crop = image( bigger_face );
  std::vector<Darknet::Detection> detections = _feature_detector.detect( crop );

  std::vector<Darknet::Detection> ears;
  std::vector<Darknet::Detection> eyes;
  std::vector<Darknet::Detection> mouths;

  for( int i = 0; i < detections.size(); i++ )
  {
    // detections[i].rect += features[FACE].tl();
    detections[i].rect += bigger_face.tl();
    if( detections[i].obj == 0 )        ears.push_back( detections[i] );
    else if( detections[i].obj == 1 )   eyes.push_back( detections[i] );
    else                                mouths.push_back( detections[i] );
  }

  //TODO 1: sort the vectors and extract two detections with highest probabilities
  //TODO 2: determine left and right ears and eyes
  std::sort ( ears.begin(), ears.end(), sort_prob );
  std::sort ( eyes.begin(), eyes.end(), sort_prob );
  std::sort ( mouths.begin(), mouths.end(), sort_prob );

  // for( int i = 0; i < 2 && i < ears.size(); i++ )    features.push_back( ears[i].rect );
  // for( int i = 0; i < 2 && i < eyes.size(); i++ )    features.push_back( eyes[i].rect );
  // if( mouths.size() != 0 )                           features.push_back( mouths[0].rect );

  for( int i = 0; i < 2; i++ )
  {
    if( i < ears.size() )   features.push_back( ears[i].rect );
    else                    features.push_back( cv::Rect() );
  }
  for( int i = 0; i < 2; i++ )
  {
    if( i < eyes.size() )   features.push_back( eyes[i].rect );
    else                    features.push_back( cv::Rect() );
  }
  if( mouths.size() != 0 )  features.push_back( mouths[0].rect );
  else                      features.push_back( cv::Rect() );
}

void Detector::drawDetections( cv::Mat &image, std::vector<cv::Rect> detections )
{
  for( int i = 0; i < detections.size(); i++ )
  {
    cv::Scalar color;
    if( i == FACE )             color = cv::Scalar(255,255,0);
    else if( i <= RIGHT_EAR )   color = cv::Scalar(255,0,0);
    else if( i <= RIGHT_EYE )   color = cv::Scalar(0,255,0);
    else                        color = cv::Scalar(0,0,255);
    cv::rectangle( image, detections[i], color, 4 );
  }
}

cv::Rect Detector::enlargeRect( const cv::Mat &image, cv::Rect rect, float wf, float hf )
{
  cv::Rect enlarged = rect;
  cv::Size deltaSize( rect.width * wf, rect.height * hf );
  cv::Point offset( deltaSize.width/2, deltaSize.height/2 );
  enlarged += deltaSize;
  enlarged -= offset;

  if( enlarged.x < 0 )  enlarged.x = 0;
  if( enlarged.y < 0 )  enlarged.y = 0;
  if( enlarged.br().x >= image.cols ) enlarged.width = image.cols - 1 - enlarged.x;
  if( enlarged.br().y >= image.rows ) enlarged.height = image.rows - 1 - enlarged.y;

  return enlarged;
}
