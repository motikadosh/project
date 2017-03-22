#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

/// Global variables

// General variables
cv::Mat src, edges;
cv::Mat src_gray;
cv::Mat standard_hough, probabilistic_hough;

cv::Rect roi;

int min_threshold = 1;
int max_trackbar = 200;

const char *standard_name = "Standard Hough Lines";
const char *probabilistic_name = "Probabilistic Hough Lines";

int edgeThresh = 36;
int edgeThreshScharr = 390;
cv::Mat edge1, edge2, cedge;
const char* window_name1 = "Edge map : Canny default (Sobel gradient)";
const char* window_name2 = "Edge map : Canny with custom gradient (Scharr)";

int s_trackbar = 75;
int p_trackbar = 50;

void showStandardHough(int, void*)
{
    std::vector<cv::Vec2f> s_lines;
    cv::cvtColor(edges, standard_hough, cv::COLOR_GRAY2BGR);

    // 1. Use Standard Hough Transform
    cv::HoughLines(edges(roi), s_lines, 1, CV_PI / 180, min_threshold + s_trackbar, 0.5, 0.5);

    // Show the result
    for (size_t i = 0; i < s_lines.size(); i++)
    {
        float r = s_lines[i][0], t = s_lines[i][1];
        double cos_t = cos(t), sin_t = sin(t);
        double x0 = r * cos_t, y0 = r * sin_t;
        double alpha = 1000;

        cv::Point pt1(cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t));
        cv::Point pt2(cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t));
        cv::line(standard_hough, pt1, pt2, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        cv::circle(standard_hough, cv::Point(x0, y0), 3, cv::Scalar(0, 255, 0), cv::FILLED);
    }

    cv::rectangle(standard_hough, roi, cv::Scalar(0, 255, 0), 2);

    cv::imshow(standard_name, standard_hough);
}

void showProbabilisticHough(int, void*)
{
    std::vector<cv::Vec4i> p_lines;
    cv::cvtColor(edges, probabilistic_hough, cv::COLOR_GRAY2BGR);

    // 2. Use Probabilistic Hough Transform
    cv::HoughLinesP(edges(roi), p_lines, 1 /*rho*/, CV_PI / 180 /*theta*/, min_threshold + p_trackbar /*thresh*/,
        20 /*minLineLength*/, 30 /*maxLineGap*/);

    // Show the result
    for (size_t i = 0; i < p_lines.size(); i++)
    {
        cv::Vec4i l = p_lines[i];
        cv::line(probabilistic_hough, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 0), 1,
            cv::LINE_AA);
    }

    cv::rectangle(probabilistic_hough, roi, cv::Scalar(0, 255, 0), 2);

    cv::imshow(probabilistic_name, probabilistic_hough);
}

void onTrackbar(int, void*)
{
    // Run the edge detector on grayscale
    cv::Canny(src_gray, edge1, edgeThresh, edgeThresh * 3, 3);
    cedge = cv::Scalar::all(0);

    src.copyTo(cedge, edge1);
    cv::imshow(window_name1, cedge);

    /// Canny detector with scharr
    cv::Mat dx,dy;
    cv::Scharr(src_gray, dx, CV_16S, 1, 0);
    cv::Scharr(src_gray, dy, CV_16S, 0, 1);
    cv::Canny(dx, dy, edge2, edgeThreshScharr, edgeThreshScharr * 3);
    /// Using Canny's output as a mask, we display our result
    cedge = cv::Scalar::all(0);
    src.copyTo(cedge, edge2);
    cv::imshow(window_name2, cedge);
}

void help()
{
    printf("\t Hough Transform to detect lines \n ");
    printf("\t---------------------------------\n ");
    printf(" Usage: ./HoughLines <image_name> \n");
}

int main(int, char** argv)
{
    src = cv::imread(argv[1], cv::IMREAD_COLOR);
    if(src.empty())
    {
        help();
        return -1;
    }

    cv::resize(src, src, cv::Size(), 0.75, 0.75);
    cv::blur(src, src, cv::Size(5, 5));
    //cv::imshow("src blur", src);
    //cv::waitKey(0);

    roi = cv::Rect(cv::Point(0, 0), cv::Size(src.cols, src.rows/(float)3.0));

    // Pass the image to gray
    cv::cvtColor(src, src_gray, cv::COLOR_RGB2GRAY);
    //cv::imshow("src_gray", src_gray);
    //cv::waitKey(0);

#if 1
    cedge.create(src.size(), src.type());

    // Create a window
    cv::namedWindow(window_name1, 1);
    cv::namedWindow(window_name2, 1);

    // create a toolbar
    cv::createTrackbar("Canny threshold default", window_name1, &edgeThresh, 100, onTrackbar);
    cv::createTrackbar("Canny threshold Scharr", window_name2, &edgeThreshScharr, 400, onTrackbar);

    // Show the image
    onTrackbar(0, 0);
    // Wait for a key stroke; the same function arranges events processing
    cv::waitKey(0);
#endif

    std::cout << "edgeThresh: " << edgeThresh << std::endl;

    // Apply Canny edge detector
    cv::Canny(src_gray, edges, edgeThresh, edgeThresh * 3, 3);
    //cv::imshow("edges", edges);
    //cv::waitKey(0);

    // Create Trackbars for Thresholds
    char thresh_label[50];
    sprintf(thresh_label, "Thres: %d + input", min_threshold);

    //float showFactor = 0.75;

    cv::namedWindow(standard_name, cv::WINDOW_AUTOSIZE);
    //cv::resizeWindow(standard_name, edges.cols*showFactor, edges.rows*showFactor);
    cv::createTrackbar(thresh_label, standard_name, &s_trackbar, max_trackbar, showStandardHough);

    showStandardHough(0, 0);
    cv::waitKey(0);

    cv::namedWindow(probabilistic_name, cv::WINDOW_AUTOSIZE);
    //cv::resizeWindow(probabilistic_name, edges.cols*showFactor, edges.rows*showFactor);
    cv::createTrackbar(thresh_label, probabilistic_name, &p_trackbar, max_trackbar, showProbabilisticHough);

    showProbabilisticHough(0, 0);
    cv::waitKey(0);
    return 0;
}

