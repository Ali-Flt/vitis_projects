#include "dlib_facedetect_and_landmark.hpp"

int main( int argc, char** argv)
{   
    const char * val = std::getenv( "FACEAPP_DEBUG" );
    if ( val == nullptr ) {
        debug = false;
    }
    else {
        std::cout << "[INFO] FACEAPP_DEBUG" << std::endl;
        debug = true;
    }

    dlib::frontal_face_detector hogFaceDetector = dlib::get_frontal_face_detector();
    dlib::shape_predictor sp;
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
    cv::VideoCapture source;
    cv::VideoWriter videowriter("output.avi",cv::VideoWriter::fourcc('M','J','P','G'),10, cv::Size(640,360));
    if (argc == 1)
        source.open(0, cv::CAP_V4L);
    else
        source.open(argv[1]);

    cv::Mat frame;

    double tt_dlibHog = 0;
    double fpsDlibHog = 0;
    while (true)
    {

        source >> frame;
        if (frame.empty())
            break;

        double t = cv::getTickCount();
        std::vector<dlib::rectangle> rects;
        dlib::cv_image<dlib::rgb_pixel> dlibIm;

        auto face_detect_result = detectFaceDlibHog(hogFaceDetector, frame);
        dlibIm = std::get<0>(face_detect_result);
        rects = std::get<1>(face_detect_result);
        if (rects.size() == 2){
            std::vector<std::vector<cv::Point2f>> points;
            points = landmark_detection(dlibIm,rects,sp);
            cv::Mat output;
            output = swap(frame,frame, points[1], points[0]);
            output = swap(frame,output, points[0], points[1]);
            tt_dlibHog = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
            fpsDlibHog = 1/tt_dlibHog;
            cv::putText(output, cv::format("Face Swapped; FPS = %.2f",fpsDlibHog), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(0, 0, 255), 4);
            videowriter.write(output);
        }
        else {
            tt_dlibHog = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
            fpsDlibHog = 1/tt_dlibHog;
            cv::putText(frame, cv::format("Face Swapped; FPS = %.2f",fpsDlibHog), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(0, 0, 255), 4);
            videowriter.write(frame);
        }

        int k = cv::waitKey(5);
        if(k == 27)
        {
            source.release();
            videowriter.release();
            cv::destroyAllWindows();
            break;
        }
    }    
    return 0;
}
