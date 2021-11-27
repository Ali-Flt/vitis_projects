#pragma once
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


#include <vitis/ai/demo.hpp>
#include <vitis/ai/facedetect.hpp>
#include <iostream>

using namespace std;
namespace vitis {
namespace ai {

struct FaceSwap {
  static std::unique_ptr<FaceSwap> create();
  FaceSwap();
  std::vector<int> run(std::vector<cv::Mat> &input_image);
  int run(cv::Mat &input_image);

  int getInputWidth();
  int getInputHeight();
  size_t get_input_batch();

private:
  std::unique_ptr<vitis::ai::FaceDetect> face_detect_;
  dlib::shape_predictor sp;
  int frame_number = 0;
  bool debug;
  FaceDetectResult faces;
  bool faulty_face = false;
  std::vector<dlib::rectangle> dlibRects;

	std::vector<std::vector<cv::Point2f>> points;

  // Apply affine transform calculated using srcTri and dstTri to src
  void applyAffineTransform(cv::Mat &warpImage, cv::Mat &src, std::vector<cv::Point2f> &srcTri, std::vector<cv::Point2f> &dstTri)
  {
      // Given a pair of triangles, find the affine transform.
      cv::Mat warpMat = getAffineTransform( srcTri, dstTri );
      
      // Apply the Affine Transform just found to the src image
      warpAffine( src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
  }


  // Calculate Delaunay triangles for set of points
  // Returns the vector of indices of 3 points for each triangle
  static void calculateDelaunayTriangles(cv::Rect rect, std::vector<cv::Point2f> &points, std::vector< std::vector<int> > &delaunayTri){

    // Create an instance of Subdiv2D
      cv::Subdiv2D subdiv(rect);

    // Insert points into subdiv
      for( std::vector<cv::Point2f>::iterator it = points.begin(); it != points.end(); it++)
          subdiv.insert(*it);         

    std::vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    std::vector<cv::Point2f> pt(3);
    std::vector<int> ind(3);

    for( size_t i = 0; i < triangleList.size(); i++ )
    {
      cv::Vec6f t = triangleList[i];
      pt[0] = cv::Point2f(t[0], t[1]);
      pt[1] = cv::Point2f(t[2], t[3]);
      pt[2] = cv::Point2f(t[4], t[5]);

      if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])){
        for(int j = 0; j < 3; j++)
          for(size_t k = 0; k < points.size(); k++)
            if(abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)            
              ind[j] = k;         

        delaunayTri.push_back(ind);
      }
    }
  }


  // Warps and alpha blends triangular regions from img1 and img2 to img
  void warpTriangle(cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2f> &t1, std::vector<cv::Point2f> &t2)
  {
      cv::Rect r1 = boundingRect(t1);
      cv::Rect r2 = boundingRect(t2);
      
      // Offset points by left top corner of the respective rectangles
      std::vector<cv::Point2f> t1Rect, t2Rect;
      std::vector<cv::Point> t2RectInt;
      for(int i = 0; i < 3; i++)
      {
          t1Rect.push_back( cv::Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
          t2Rect.push_back( cv::Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
          t2RectInt.push_back( cv::Point(t2[i].x - r2.x, t2[i].y - r2.y) ); // for fillConvexPoly
      }
      
      // Get mask by filling triangle
      cv::Mat mask = cv::Mat::zeros(r2.height, r2.width, CV_32FC3);
      fillConvexPoly(mask, t2RectInt, cv::Scalar(1.0, 1.0, 1.0), 16, 0);
      
      // Apply warpImage to small rectangular patches
      cv::Mat img1Rect;
      img1(r1).copyTo(img1Rect);
      
      cv::Mat img2Rect = cv::Mat::zeros(r2.height, r2.width, img1Rect.type());
      
      applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);
      
      cv::multiply(img2Rect,mask, img2Rect);
      cv::multiply(img2(r2), cv::Scalar(1.0,1.0,1.0) - mask, img2(r2));
      img2(r2) = img2(r2) + img2Rect;
  }


void swap(cv::Mat img1, cv::Mat& img2, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2) {
      cv::Mat img1Warped = img2.clone();
      
      //convert Mat to float data type
      img1.convertTo(img1, CV_32F);
      img1Warped.convertTo(img1Warped, CV_32F);
      
      
      // Find convex hull
      std::vector<cv::Point2f> hull1;
      std::vector<cv::Point2f> hull2;
      std::vector<int> hullIndex;
      
      cv::convexHull(points2, hullIndex, false, false);
      
      for(unsigned int i = 0; i < hullIndex.size(); i++)
      {
          hull1.push_back(points1[hullIndex[i]]);
          hull2.push_back(points2[hullIndex[i]]);
      }

      
      // Find delaunay triangulation for points on the convex hull
      std::vector< std::vector<int> > dt;
      cv::Rect rect(0, 0, img1Warped.cols, img1Warped.rows);
      calculateDelaunayTriangles(rect, hull2, dt);
      
      // Apply affine transformation to Delaunay triangles
      for(size_t i = 0; i < dt.size(); i++)
      {
          std::vector<cv::Point2f> t1, t2;
          // Get points for img1, img2 corresponding to the triangles
          for(size_t j = 0; j < 3; j++)
          {
              t1.push_back(hull1[dt[i][j]]);
              t2.push_back(hull2[dt[i][j]]);
          }
          warpTriangle(img1, img1Warped, t1, t2);
      }
      
      // Calculate mask
      std::vector<cv::Point> hull8U;
      for(unsigned int i = 0; i < hull2.size(); i++)
      {
          cv::Point pt(hull2[i].x, hull2[i].y);
          hull8U.push_back(pt);
      }

      cv::Mat mask = cv::Mat::zeros(img2.rows, img2.cols, img2.depth());
      cv::fillConvexPoly(mask,&hull8U[0], hull8U.size(), cv::Scalar(255,255,255));

      // Clone seamlessly.
      cv::Rect r = boundingRect(hull2);
      cv::Point center = (r.tl() + r.br()) / 2;
      
      img1Warped.convertTo(img1Warped, CV_8UC3);
      seamlessClone(img1Warped,img2, mask, center, img2, cv::NORMAL_CLONE);
      return;
  }

  std::vector<std::vector<cv::Point2f>> landmark_detection(dlib::cv_image<dlib::rgb_pixel> dlibIm, std::vector<dlib::rectangle> rects, dlib::shape_predictor sp) {
      std::vector<std::vector<cv::Point2f>> both_points;
      for (unsigned long j = 0; j < 2; ++j)
      {
          dlib::full_object_detection shape = sp(dlibIm, rects[j]);
          if (shape.num_parts() == 68){
              std::vector<cv::Point2f> points;
              for(int i = 0; i < 68; ++i){
                  points.push_back(cv::Point2f((float)shape.part(i)(0),(float)shape.part(i)(1)));
              }
              both_points.push_back(points);
          }
          else{
        	  faulty_face = true;
        	  return both_points;
          }
      }
      faulty_face = false;
      return both_points;
  }
};

std::unique_ptr<FaceSwap> FaceSwap::create() {
  return std::unique_ptr<FaceSwap>(new FaceSwap());
}
int FaceSwap::getInputWidth() { return face_detect_->getInputWidth(); }
int FaceSwap::getInputHeight() { return face_detect_->getInputHeight(); }
size_t FaceSwap::get_input_batch() { return face_detect_->get_input_batch(); }

FaceSwap::FaceSwap()
    : face_detect_{vitis::ai::FaceDetect::create("densebox_640_360")}
{
  dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
  const char * val = std::getenv( "FACESWAP_DEBUG" );
  if ( val == nullptr ) {
    debug = false;
  }
  else {
    cout << "[INFO] FACESWAP_DEBUG" << endl;
    debug = true;
  }
}


std::vector<int>
FaceSwap::run(std::vector<cv::Mat> &input_image) {
  if ( debug == true ) {
    cout << "[INFO] batch mode" << endl;
  }
  auto batch_size = get_input_batch();
  std::vector<cv::Mat> input_tmp;
  std::vector<int> faceswap_results;
  size_t ind = 0;
  while(ind < input_image.size()) {
    size_t next = std::min(ind + batch_size, input_image.size());
    for(size_t i = ind; i < next; i++) {
      input_tmp.push_back(input_image[i]);
    }
    auto face_detect_results = face_detect_->run(input_tmp);
    input_tmp.clear();
    for (size_t i = 0; i < face_detect_results.size(); i++){
		if(face_detect_results[i].rects.size() >= 2){
			int good_faces = 0;
			for (const auto &r : face_detect_results[i].rects) {
				if ( r.score < 0.90 ) continue;
				good_faces++;
			}
			if (good_faces >= 2){
				cv::Mat image;
				image = input_image[ind+i];
				std::vector<dlib::rectangle> dlibRects;
				for (const auto &r : face_detect_results[i].rects) {
					std::vector<long> tl = {(long)(r.x * image.cols), (long)(r.y * image.rows)};
					std::vector<long> br = {tl[0] + (long)(r.width * image.cols), tl[1] + (long)(r.height * image.rows)};
					dlibRects.push_back(dlib::rectangle(tl[0],tl[1],br[0],br[1]));
				}
				std::vector<std::vector<cv::Point2f>> points;
				dlib::cv_image<dlib::rgb_pixel> dlibIm(image);
				points = landmark_detection(dlibIm,dlibRects,sp);
				cv::Mat imgClone = image.clone();
				swap(image,image, points[1], points[0]);
				swap(imgClone,image, points[0], points[1]);
				faceswap_results.push_back(0);
			}
			else{
				faceswap_results.push_back(1);
			}
		}
		else{
			faceswap_results.push_back(2);
		}
    }
	ind = ind + batch_size;
  }
  return faceswap_results;
}


int FaceSwap::run(cv::Mat &image) {
	frame_number++;
  // only process 110 frames
	if (frame_number > 110){
		std::cout << "EXIT" << std::endl;
		exit(2);
	}
	if ( debug == true ) {
		std::cout << "[INFO] single file mode" << endl;
		std::cout << "Frame " << frame_number << endl;
	}
	faces = face_detect_->run(image);
	if(faces.rects.size() >= 2){
		int good_faces = 0;
		for (const auto &r : faces.rects) {
			if ( r.score < 0.90 ) continue;
			good_faces++;
		}
		if (good_faces >= 2){
			dlibRects.clear();
			for (const auto &r : faces.rects) {
				std::vector<long> tl = {(long)(r.x * image.cols), (long)(r.y * image.rows)};
				std::vector<long> br = {tl[0] + (long)(r.width * image.cols), tl[1] + (long)(r.height * image.rows)};
				dlibRects.push_back(dlib::rectangle(tl[0],tl[1],br[0],br[1]));
			}
			dlib::cv_image<dlib::rgb_pixel> dlibIm(image);
			points = landmark_detection(dlibIm,dlibRects,sp);
			if (faulty_face == false){
				cv::Mat imageClone = image.clone();
				swap(image,image, points[1], points[0]);
				swap(imageClone,image, points[0], points[1]);
				return 0;
			}
		}
	}
	return 1;
}

} // namespace ai
} // namespace vitis
