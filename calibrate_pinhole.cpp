#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <stdio.h>
#include <iostream>
#include "popt_pp.h"

using namespace std;
using namespace cv;

vector< vector< Point3f > > object_points;
vector< vector< Point2f > > imagePoints1, imagePoints2;
vector< Point2f > corners1, corners2;
vector< vector< Point2f > > left_img_points, right_img_points;

Mat img1, img2, gray1, gray2, spl1, spl2;

void load_image_points(int board_width, int board_height, float square_size, int num_imgs, 
                      char* img_dir, char* leftimg_filename, char* rightimg_filename) {
  Size board_size = Size(board_width, board_height);
  int board_n = board_width * board_height;

  for (int i = 0; i <= num_imgs-1; i++) {
    char left_img[100], right_img[100];
    // char number[2] = (i<10?"0"+to_string(i):to_string(i)).c_str();
    // std::cout<<number<<std::endl;
    sprintf(left_img, "%s%s00%s.jpg", img_dir, leftimg_filename, (i<10?"0"+to_string(i):to_string(i)).c_str());
    sprintf(right_img, "%s%s00%s.jpg", img_dir, rightimg_filename, (i<10?"0"+to_string(i):to_string(i)).c_str());
    std::cout<<left_img<<std::endl;
    img1 = imread(left_img, IMREAD_COLOR);
    img2 = imread(right_img, IMREAD_COLOR);
    cv::cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, COLOR_BGR2GRAY);

    bool found1 = false, found2 = false;
    int chessboardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
    found1 = cv::findChessboardCorners(gray1, board_size, corners1, chessboardFlags);
    found2 = cv::findChessboardCorners(gray2, board_size, corners2, chessboardFlags);

    if (found1)
    {
      cv::cornerSubPix(gray1, corners1, cv::Size(5, 5), cv::Size(-1, -1),
        cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
      cv::drawChessboardCorners(gray1, board_size, corners1, found1);
      // cv::imshow("corners 1",gray1);
      // cv::imshow();
    }
    if (found2)
    {
      cv::cornerSubPix(gray2, corners2, cv::Size(5, 5), cv::Size(-1, -1),
        cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
      cv::drawChessboardCorners(gray2, board_size, corners2, found2);
      // cv::imshow("corners 2",gray2);
      // cv::waitKey(10000);
    }

    vector<cv::Point3f> obj;
    for( int i = 0; i < board_height; ++i )
      for( int j = 0; j < board_width; ++j )
        obj.push_back(Point3f( (float)j * square_size, (float)i * square_size, 0));

    if (found1 && found2) {
      cout << i << ". Found corners!" << endl;
      // drawChessboardCorners(gray1, board_size, corners1, found1);
      // drawChessboardCorners(gray2, board_size, corners2, found2);
      imagePoints1.push_back(corners1);
      imagePoints2.push_back(corners2);
      object_points.push_back(obj);
    }
  }
  for (int i = 0; i < imagePoints1.size(); i++) {
    vector< Point2f > v1, v2;
    for (int j = 0; j < imagePoints1[i].size(); j++) {
      v1.push_back(Point2f(imagePoints1[i][j].x, imagePoints1[i][j].y));
      v2.push_back(Point2f(imagePoints2[i][j].x, imagePoints2[i][j].y));
    }
    left_img_points.push_back(v1);
    right_img_points.push_back(v2);
  }
}

int main(int argc, char const *argv[])
{
  int board_width, board_height, num_imgs;
  float square_size;
  char* img_dir;
  char* leftimg_filename;
  char* rightimg_filename;
  char* out_file;

  static struct poptOption options[] = {
    { "board_width",'w',POPT_ARG_INT,&board_width,0,"Checkerboard width","NUM" },
    { "board_height",'h',POPT_ARG_INT,&board_height,0,"Checkerboard height","NUM" },
    { "square_size",'s',POPT_ARG_FLOAT,&square_size,0,"Checkerboard square size","NUM" },
    { "num_imgs",'n',POPT_ARG_INT,&num_imgs,0,"Number of checkerboard images","NUM" },
    { "img_dir",'d',POPT_ARG_STRING,&img_dir,0,"Directory containing images","STR" },
    { "leftimg_filename",'l',POPT_ARG_STRING,&leftimg_filename,0,"Left image prefix","STR" },
    { "rightimg_filename",'r',POPT_ARG_STRING,&rightimg_filename,0,"Right image prefix","STR" },
    { "out_file",'o',POPT_ARG_STRING,&out_file,0,"Output calibration filename (YML)","STR" },
    POPT_AUTOHELP
    { NULL, 0, 0, NULL, 0, NULL, NULL }
  };

  POpt popt(NULL, argc, argv, options, 0);
  int c;
  while((c = popt.getNextOpt()) >= 0) {}

  load_image_points(board_width, board_height, square_size, num_imgs, img_dir, leftimg_filename, rightimg_filename);

  printf("Starting Calibration\n");
  cv::Mat K1, K2, R, E, F;
  cv::Vec3d T;
  cv::Mat D1, D2;
  vector<Mat> rvecs, tvecs;
  vector<float> reprojErrs;
  double totalAvgErr = 0;
  // vector<Point3f> newObjPoints;
  K1 = Mat::eye(3, 3, CV_64F);
  K2 = Mat::eye(3, 3, CV_64F);
  D1 = Mat::zeros(8, 1, CV_64F);
  D2 = Mat::zeros(8, 1, CV_64F);
  int flag = 0;
  flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
  flag |= cv::fisheye::CALIB_CHECK_COND;
  flag |= cv::fisheye::CALIB_FIX_SKEW;
  // flag |= cv::fisheye::CALIB_FIX_PRINCIPAL_POINT;
  //flag |= cv::fisheye::CALIB_FIX_K2;
  //flag |= cv::fisheye::CALIB_FIX_K3;
  //flag |= cv::fisheye::CALIB_FIX_K4;
  double rms = cv::stereoCalibrate(object_points, left_img_points, right_img_points,
      K1, D1, K2, D2, img1.size(), R, T, E, F, CALIB_FIX_PRINCIPAL_POINT+ CALIB_RATIONAL_MODEL + CALIB_FIX_K4 + CALIB_FIX_K5+ CALIB_FIX_K6);
  // double rms = cv::fisheye::stereoCalibrate(object_points, left_img_points, right_img_points,
  //     K1, D1, K2, D2, img1.size(), R, T, flag,
  //     cv::TermCriteria(3, 12, 0));
  cout << "done with RMS error=" << rms << endl;
  cv::FileStorage fs1(out_file, cv::FileStorage::WRITE);
  fs1 << "K1" << Mat(K1);
  fs1 << "K2" << Mat(K2);
  fs1 << "D1" << D1;
  fs1 << "D2" << D2;
  fs1 << "R" << Mat(R);
  fs1 << "T" << T;
  printf("Done Calibration\n");

  printf("Starting Rectification\n");

  cv::Mat R1, R2, P1, P2, Q;
  cv::stereoRectify(K1, D1, K2, D2, img1.size(), R, T, R1, R2, P1, P2, 
    Q);
  // cv::stereoRectify(K1, D1, K2, D2, img1.size(), R, T, R1, R2, P1, P2,
  //   Q, CALIB_ZERO_DISPARITY);

  fs1 << "R1" << R1;
  fs1 << "R2" << R2;
  fs1 << "P1" << P1;
  fs1 << "P2" << P2;
  fs1 << "Q" << Q;

  printf("Done Rectification\n");
  return 0;
}
