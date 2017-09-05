#include "darknet.hpp"

void Darknet::loadModel( std::string cfgfile, std::string weightsfile )
{
  cudaSetDevice(0);
  loadCfg( cfgfile);
  loadWeights( weightsfile );

  set_batch_network( &net, 1 );
}

void Darknet::loadCfg( std::string cfgfile )
{
  std::cout << std::endl << "Loading graph from " << cfgfile << std::endl;
  char *cfg = new char[cfgfile.length()+1];
  strcpy(cfg, cfgfile.c_str());
  net = parse_network_cfg( cfg );
  std::cout << "Done!"  << std::endl;
}

void Darknet::loadWeights( std::string weightsfile )
{
  char *weights = new char[weightsfile.length()+1];
  strcpy(weights, weightsfile.c_str());
  load_weights( &net, weights );
}

void Darknet::loadModel( char *cfg, char *weights )
{
  net = parse_network_cfg( cfg );
  load_weights( &net, weights );
  set_batch_network( &net, 1 );
}

std::vector<Darknet::Detection> Darknet::detect( cv::Mat cvimage )
{
  image im = convertImage( cvimage );

  float hier_thresh=0.5;
  float nms=.3;
  float thresh=0.24;

  layer l = net.layers[net.n-1];

  box *boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
  float **probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
  for(int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes + 1, sizeof(float *));
  float **masks = 0;
  if (l.coords > 4){
      masks = (float **)calloc(l.w*l.h*l.n, sizeof(float*));
      for(int j = 0; j < l.w*l.h*l.n; ++j) masks[j] = (float *)calloc(l.coords-4, sizeof(float *));
  }

  float *X = im.data;
  double time=what_time_is_it_now();
  network_predict(net, X);
  printf("Predicted in %f seconds.\n", what_time_is_it_now()-time);
  get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
  if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);

  return getBoxes( l.w*l.h*l.n, boxes, probs, l.classes, cv::Size(cvimage.cols, cvimage.rows) );
}

image Darknet::convertImage( cv::Mat cvimage )
{
  cv::Mat resized;
  cv::resize( cvimage, resized, cv::Size( net.w, net.h ) );

  int h = resized.rows;
  int w = resized.cols;
  int c = resized.channels();
  int step = resized.step;
  unsigned char *data = (unsigned char *) resized.data;

  image im = make_image( w, h, c );

  for( int i = 0; i < h; i++ )
  {
    for( int k = 0; k < c; k++ )
    {
      for( int j = 0; j < w; j++ )
      {
        im.data[ k*w*h + i*w + j ] = data[ i*step + j*c + k]/255.;
      }
    }
  }

  return im;
}

std::vector<Darknet::Detection> Darknet::getBoxes( int num, box *boxes, float  **probs, int classes, cv::Size imgsize )
{
  double thresh = 0.24;
  std::vector<Darknet::Detection> detections;

  for(int i = 0; i < num; ++i){
       int obj = max_index(probs[i], classes);
       float prob = probs[i][obj];
       if(prob > thresh){
           box b = boxes[i];

           int left  = (b.x-b.w/2.)*imgsize.width;
           int right = (b.x+b.w/2.)*imgsize.width;
           int top   = (b.y-b.h/2.)*imgsize.height;
           int bot   = (b.y+b.h/2.)*imgsize.height;

           if(left < 0) left = 0;
           if(right > imgsize.width-1) right = imgsize.width-1;
           if(top < 0) top = 0;
           if(bot > imgsize.height-1) bot = imgsize.height-1;

           Detection detection;
           detection.rect = cv::Rect( cv::Point(left,top), cv::Point(right,bot));
           detection.prob = prob;
           detection.obj = obj;

           detections.push_back( detection );
       }
   }

   return detections;
}
