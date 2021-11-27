#include <glog/logging.h>

#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo.hpp>
#include <vitis/ai/facedetect.hpp>
#include "./face_swap.hpp"



using namespace std;
cv::Mat
process_result_faceswap(cv::Mat &in_image,
                  const cv::Mat &out_image,
                  bool is_jpeg) {
 return out_image;
}

using namespace std;
int main(int argc, char *argv[]) {
  return vitis::ai::main_for_video_demo(
    argc, argv, [] { return vitis::ai::FaceSwap::create(); },
    process_result_faceswap );

}
