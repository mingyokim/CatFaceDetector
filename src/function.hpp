#include "detector.hpp"

#include <boost/filesystem.hpp>
using namespace boost::filesystem;

class Function
{
public:
  static void detectSingleImage( std::string src_path, std::string dst_path );
  static void detectMultipleImages( std::string src_path, std::string dst_path );
  static void detectVideo( std::string src_path, std::string dst_path );
  static std::vector<boost::filesystem::path> getImagePathsInFolder( const boost::filesystem::path &folder, const std::string &ext );
};
