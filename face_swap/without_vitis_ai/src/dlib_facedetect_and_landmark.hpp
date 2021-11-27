#ifndef _DLIB_FDETECT_AND_LANDMARK_HPP_
#define _DLIB_FDETECT_AND_LANDMARK_HPP_
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

#include <iostream>
#include <fstream>
#include <string> 

bool debug;

// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(cv::Mat &warpImage, cv::Mat &src, std::vector<cv::Point2f> &srcTri, std::vector<cv::Point2f> &dstTri)
{
    // Given a pair of triangles, find the affine transform.
    cv::Mat warpMat = getAffineTransform( srcTri, dstTri );
    
    // Apply the Affine Transform just found to the src image
    cv::warpAffine( src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
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


cv::Mat swap(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2) {
    cv::Mat img1Warped = img2.clone();
    
    //convert Mat to float data type
    img1.convertTo(img1, CV_32F);
    img1Warped.convertTo(img1Warped, CV_32F);
    
    
    // Find convex hull
    std::vector<cv::Point2f> hull1;
    std::vector<cv::Point2f> hull2;
    std::vector<int> hullIndex;
    
    cv::convexHull(points2, hullIndex, false, false);
    
    for(int i = 0; i < hullIndex.size(); i++)
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
    for(int i = 0; i < hull2.size(); i++)
    {
        cv::Point pt(hull2[i].x, hull2[i].y);
        hull8U.push_back(pt);
    }

    cv::Mat mask = cv::Mat::zeros(img2.rows, img2.cols, img2.depth());
    cv::fillConvexPoly(mask,&hull8U[0], hull8U.size(), cv::Scalar(255,255,255));

    // Clone seamlessly.
    cv::Rect r = boundingRect(hull2);
    cv::Point center = (r.tl() + r.br()) / 2;
    
    cv::Mat output;
    img1Warped.convertTo(img1Warped, CV_8UC3);
    cv::seamlessClone(img1Warped,img2, mask, center, output, cv::NORMAL_CLONE);
    return output;
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
                // std::cout << i << " " << shape.part(i)(0) << " " << shape.part(i)(1) << std::endl; 
            }
            both_points.push_back(points);
        }
    }

    return both_points;
}

std::tuple<dlib::cv_image<dlib::rgb_pixel>,std::vector<dlib::rectangle>> detectFaceDlibHog(dlib::frontal_face_detector hogFaceDetector, cv::Mat& frameDlibHog, int inHeight=300, int inWidth=0)
{

    int frameHeight = frameDlibHog.rows;
    dlib::cv_image<dlib::rgb_pixel> dlibIm(frameDlibHog);

    // Detect faces in the image
    std::vector<dlib::rectangle> faceRects = hogFaceDetector(dlibIm);
    if(debug){
        for ( size_t i = 0; i < faceRects.size(); i++ )
        {
            int x1 = (int)(faceRects[i].left());
            int y1 = (int)(faceRects[i].top());
            int x2 = (int)(faceRects[i].right());
            int y2 = (int)(faceRects[i].bottom());
            cv::rectangle(frameDlibHog, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,255,0), (int)(frameHeight/150.0), 4);
        }
    }
    return std::make_tuple(dlibIm,faceRects);
}

#endif