// Homework1 Nadezhda Manuilova ID:2049581
#include <opencv2/opencv.hpp>
#include<iostream>
#include <opencv2/core/mat.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"


using namespace std;
using namespace cv;

int  kernel;  //median filter parameter
const int kernel_max = 200;

int sigma_gauss; //gaussian blur parameters
int kernel_gauss;
const int kernel_gauss_max = 100;
const int sigma_gauss_max = 100;

int range_bilateral; //bilateral filter parameters
int space_bilateral;
const int range_bilateral_max = 200;
const int space_bilateral_max = 50;

// copy and paste inside your code

// hists = vector of 3 cv::mat of size nbins=256 with the 3 histograms
// e.g.: hists[0] = cv:mat of size 256 with the red histogram
//       hists[1] = cv:mat of size 256 with the green histogram
//       hists[2] = cv:mat of size 256 with the blue histogram
void showHistogram(std::vector<cv::Mat>& hists)
{
    // Min/Max computation
    double hmax[3] = { 0,0,0 };
    double min;
    cv::minMaxLoc(hists[0], &min, &hmax[0]);
    cv::minMaxLoc(hists[1], &min, &hmax[1]);
    cv::minMaxLoc(hists[2], &min, &hmax[2]);

    std::string wname[3] = { "blue", "green", "red" };
    cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),
                             cv::Scalar(0,0,255) };

    std::vector<cv::Mat> canvas(hists.size());

    // Display each histogram in a canvas
    for (int i = 0, end = hists.size(); i < end; i++)
    {
        canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

        for (int j = 0, rows = canvas[i].rows; j < hists[0].rows - 1; j++)
        {
            cv::line(
                canvas[i],
                cv::Point(j, rows),
                cv::Point(j, rows - (hists[i].at<float>(j) * rows / hmax[i])),
                hists.size() == 1 ? cv::Scalar(200, 200, 200) : colors[i],
                1, 8, 0
            );
        }

        cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
    }
}


/*
Median
*/
void CallbackMedian(int value, void* userData)
{
    Mat image = *(Mat*)userData;
    Mat Median_image;

    int kernel = value;
    if (kernel % 2 == 0)
    {
        kernel = kernel + 1;
    }

    cv::medianBlur(image, Median_image, kernel);
    imshow("median", Median_image);
}

/*
Gaussian
*/

void CallbackGaussian_kernel(int value, void* userData)
{
    Mat image = *(Mat*)userData;
    Mat gaussian_image;

    kernel_gauss = value;
    if (kernel_gauss % 2 == 0)
    {
        kernel_gauss = kernel_gauss + 1;
    }
    cv::GaussianBlur(image, gaussian_image, Size(kernel_gauss, kernel_gauss), (double)sigma_gauss);
    imshow("gaussian", gaussian_image);
}
void CallbackGaussian_sigma(int value, void* userData)
{
    Mat image = *(Mat*)userData;
    Mat gaussian_image;
    sigma_gauss = value / 10;
    if (sigma_gauss == 0)
    {
        sigma_gauss = sigma_gauss + 1;
    }
    cv::GaussianBlur(image, gaussian_image, Size(kernel_gauss, kernel_gauss), sigma_gauss);
    imshow("gaussian", gaussian_image);


}
/*
Bilateral
*/

void CallbackBilateral_range(int value, void* userData)
{
    Mat image = *(Mat*)userData;
    Mat bilateral_image;
    range_bilateral = value;

    cv::bilateralFilter(image, bilateral_image, 6 * space_bilateral, (double)range_bilateral, (double)space_bilateral);
    imshow("bilateral", bilateral_image);
}
void CallbackBilateral_space(int value, void* userData)
{
    Mat image = *(Mat*)userData;
    Mat bilateral_image;
    space_bilateral = value;

    cv::bilateralFilter(image, bilateral_image, 6 * space_bilateral, (double)range_bilateral, (double)space_bilateral);
    imshow("bilateral", bilateral_image);
}

/*
Histogram Equalization
*/

int main(int argc, char** argv)
{
    cv::Mat original_image = cv::imread("barbecue.png"); // load the image barbecue.png

    /*
    calcHist function variables and parameters
    */

    vector<Mat> hists;
    split(original_image, hists); //split the image in three components B,R,G
    int histSize = 256; // the number of bins
    float range[] = { 0, 256 }; //range
    const float* histRange = { range };
    bool uniform = true, accumulate = false;

    Mat b_hist, g_hist, r_hist; //for histogram computing

    Mat b_eqhist, g_eqhist, r_eqhist; // for equalization
    Mat eq_image; //place for the equalized image


    //computing histogram for the original image and shows the histogram
    calcHist(&hists[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&hists[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&hists[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    vector<Mat> computedHist{ b_hist,g_hist,r_hist };
    showHistogram(computedHist);

    imshow("barbecue.png", original_image); //visualized original image

    //Equalizes separately the R, G and B channels by using cv::equalizeHist().

    //equalize histogram of the original image 
    equalizeHist(hists[0], b_eqhist);
    equalizeHist(hists[1], g_eqhist);
    equalizeHist(hists[2], r_eqhist);

    vector<Mat> eq_hists{ b_eqhist,g_eqhist ,r_eqhist }; //vector with the three equalized channel

    merge(eq_hists, eq_image); //merge the three equalized components B,R,G
    imshow("equalizedImage", eq_image); //visualized equalized image
    imwrite("equalized_image.png", eq_image);

    //computing histogram for the previous equalized image and shows the histogram
    split(eq_image, hists); //split the image in three component B,R,G
    calcHist(&hists[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate); //computing histogram
    calcHist(&hists[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&hists[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    vector<Mat> computed_eqHist{ b_hist,g_hist,r_hist };
    showHistogram(computed_eqHist);


    Mat converted_image;
    cv::cvtColor(original_image, converted_image, COLOR_BGR2Lab);

    split(converted_image, hists);

    //computing histogram for the image in the CIELab color space
    calcHist(&hists[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate); //computing histogram
    calcHist(&hists[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&hists[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    vector<Mat> computed_convertedHist{ b_hist,g_hist,r_hist };
    showHistogram(computed_convertedHist);

    Mat L_hist;

    //equalization of the luminance channel
    equalizeHist(hists[0], L_hist);

    Mat eqconverted_image; //place for the equalized image
    vector<Mat> eqconverted_hists{ L_hist,hists[1],hists[2] }; //vector with the three equalized channel
    merge(eqconverted_hists, eqconverted_image); //merge the equalized component L, and the oter two (a,b) that aren't normalized

    cvtColor(eqconverted_image, eqconverted_image, COLOR_Lab2BGR); //the image is converted back in the RGB color space

    imshow("converted_eqalized_Image", eqconverted_image); //visualized equalized image
    imwrite("converted_eqalized_Image.png", eqconverted_image);

    split(eqconverted_image, hists);
    //histogram of the image obtained by equalizing the luminance channel
    calcHist(&hists[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate); //computing histogram
    calcHist(&hists[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&hists[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    vector<Mat> computed_eqconvertedHist{ b_hist,g_hist,r_hist };
    showHistogram(computed_eqconvertedHist);

    /*
    generate the trackbars you can use the cv::createTrackbar() function
    */


    namedWindow("median", 1); //give a name
    namedWindow("gaussian", 1);
    namedWindow("bilateral", 1);

    cv::createTrackbar("kernel", "median", &kernel, kernel_max, CallbackMedian, (void*)&eqconverted_image); //median trackbar

    cv::createTrackbar("kernel_g", "gaussian", &kernel_gauss, kernel_gauss_max, CallbackGaussian_kernel, (void*)&eqconverted_image); //gaussian trackbar kernel
    cv::createTrackbar("sigma", "gaussian", &sigma_gauss, sigma_gauss_max, CallbackGaussian_sigma, (void*)&eqconverted_image); //gaussian trackbar sigma

    cv::createTrackbar("sigmaRange", "bilateral", &range_bilateral, range_bilateral_max, CallbackBilateral_range, (void*)&eqconverted_image); //bilateral trackbar sigma range
    cv::createTrackbar("sigmaSpace", "bilateral", &space_bilateral, space_bilateral_max, CallbackBilateral_space, (void*)&eqconverted_image); //bilateral trackbar sigma space


    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
