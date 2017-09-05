#ifndef _DARKNET_H
#define _DARKNET_H

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

#include <darknet.h>

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cudnn.h"

class Darknet
{
public:
  struct Detection
  {
    cv::Rect rect;
    float prob;
    int obj;
  };

private:
  network net;

  void loadCfg( std::string cfgfile );
  void loadWeights( std::string weightsfile );
  image convertImage( cv::Mat img );
  std::vector<Detection> getBoxes( int num, box *boxes, float **probs, int classes, cv::Size imgsize );

public:
  void loadModel( std::string cfgfile, std::string weightfile );
  void loadModel( char *cfgfile, char *weightfile );
  std::vector<Detection> detect( cv::Mat img );
};

#endif
