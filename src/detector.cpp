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
  cv::Mat crop = image( features[FACE] );
  std::vector<Darknet::Detection> detections = _feature_detector.detect( crop );

  std::vector<Darknet::Detection> ears;
  std::vector<Darknet::Detection> eyes;
  std::vector<Darknet::Detection> mouths;

  for( int i = 0; i < detections.size(); i++ )
  {
    detections[i].rect += features[FACE].tl();
    if( detections[i].obj == 0 )        ears.push_back( detections[i] );
    else if( detections[i].obj == 1 )   eyes.push_back( detections[i] );
    else                                mouths.push_back( detections[i] );
  }

  //TODO 1: sort the vectors and extract two detections with highest probabilities
  //TODO 2: determine left and right ears and eyes
  for( int i = 0; i < 2 && i < ears.size(); i++ )    features.push_back( ears[i].rect );
  for( int i = 0; i < 2 && i < eyes.size(); i++ )    features.push_back( eyes[i].rect );
  if( mouths.size() != 0 )                           features.push_back( mouths[0].rect );
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
