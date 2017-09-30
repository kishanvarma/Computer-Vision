
//
//  main.cpp
//  HelloWorldCV
//
//  Created by Kishan Varma on Sep/1/2017.
//  Copyright © 2017 Kishan Varma. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    cout << "Read  + Display : " << endl;
    
    Mat img = imread("K.jpg", CV_LOAD_IMAGE_COLOR);
    
    if (img.empty()) {
        cout << "\nLoad Image : FAILED\n";
        return -1;
    }
    
    namedWindow("myphoto", CV_WINDOW_AUTOSIZE);
    imshow("myphoto", img);
    
    cout << "Add : " << endl;
    Mat inc_bright_img;  // = img + Scalar(30,30,30);
    img.convertTo(inc_bright_img, -1, 1, 75);
    namedWindow("myphoto1", CV_WINDOW_AUTOSIZE);
    imshow("myphoto1", inc_bright_img);
    
    cout << "Subtract : " << endl;
    Mat dec_bright_img;  //= img + Scalar(-50,-50,-50);
    img.convertTo(dec_bright_img, -1, 1, -50);
    namedWindow("myphoto2", CV_WINDOW_AUTOSIZE);
    imshow("myphoto2", dec_bright_img);
    
    cout << "Multiply : " << endl;
    Mat inc_contrast_img;
    img.convertTo(inc_contrast_img, -1, 2, 0);
    namedWindow("myphoto3", CV_WINDOW_AUTOSIZE);
    imshow("myphoto3", inc_contrast_img);
    
    cout << "Divide : " << endl;
    Mat dec_contrast_img;
    img.convertTo(dec_contrast_img, -1, 0.5, 0);
    namedWindow("myphoto4", CV_WINDOW_AUTOSIZE);
    imshow("myphoto4", dec_contrast_img);
    
    cout << "Resize : " << endl;
    Mat resize_img;
    resize(img, resize_img, cv::Size(), 0.75, 0.75);
    namedWindow("myphoto5", CV_WINDOW_AUTOSIZE);
    imshow("myphoto5", resize_img);
    
    waitKey();
    return 0;
}
