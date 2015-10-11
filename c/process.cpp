#include <sstream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

const float FIELD_CORNERS[8] = {
  153, 36,
  456, 37,
  74, 461,
  518, 460
};

void find_ball_threshold(Mat& frame, Mat& mask) {
  Mat hsv;
  cvtColor(frame, hsv, COLOR_BGR2YCrCb);
  inRange(hsv, Scalar(0, 140, 110), Scalar(187, 197, 124), mask);
}

Mat get_board_mtx(Mat& frame, Mat& mtx) {
  Mat dest = Mat::zeros(frame.size(), frame.type());
  float corners[8] = {
    0, 0,
    (float)dest.rows, 0,
    0, (float)dest.cols,
    (float)dest.rows, (float)dest.cols,
  };
  mtx = getPerspectiveTransform(
    Mat(4, 1, CV_32FC2, (void*)FIELD_CORNERS),
    Mat(4, 1, CV_32FC2, (void*)corners)
    );
  return mtx;
}

void draw_board(Mat& frame, Mat& dest) {
  Mat mtx;
  get_board_mtx(frame, mtx);
  Size size = frame.size();
  warpPerspective(
    frame, dest, mtx,
    Size(size.height, size.width));
}

Rect match_template(Mat& mask) {
  Mat res, tpl;
  double min_val, max_val;
  Point min_loc, max_loc;
  tpl = Mat::zeros(Size(22, 22), CV_8UC1);
  circle(tpl, Point(11, 11), 11, Scalar(255), -1);
  matchTemplate(mask, tpl, res, TM_CCORR);
  minMaxLoc(res, &min_val, &max_val, &min_loc, &max_loc);
  return Rect(max_loc.x, max_loc.y, 22, 22);
}


int main(int argc, char** argv) {
  VideoCapture cap("../Horton_Beine vs. Atha_Loffredo at the 2014 Tornado Worlds.mp4");
  if(!cap.isOpened()) return -1;
  int frameNumber = 1860;
  int delay = 0;
  stringstream matchString;
  Point previousMatch, velocity, previousVelocity, acceleration;
  cap.set(CV_CAP_PROP_POS_FRAMES, frameNumber);

  namedWindow("image");
  moveWindow("image", 0, 200);

  for(;;) {
    Mat frame, mask, mtx, mtxi, board;
    vector<Point2f> t;
    cap >> frame; frameNumber++;
    get_board_mtx(frame, mtx);
    invert(mtx, mtxi);

    draw_board(frame, board);
    find_ball_threshold(board, mask);
    Rect match = match_template(mask);

    acceleration = velocity - previousVelocity;
    velocity = previousMatch - match.tl();
    if (norm(velocity) > 100) {
      cout << "too much acc, drawing another square\n";
      perspectiveTransform(
        (vector<Point2f>){
          previousMatch + previousVelocity,
            previousMatch + previousVelocity + Point(22, 22)
            },
        t, mtxi);
      rectangle(frame, Rect(t.at(0), t.at(1)), Scalar(0, 255, 0), 1);
      previousVelocity = velocity;
      previousMatch = match.tl();
      continue;
    }
    previousVelocity = velocity;
    previousMatch = match.tl();

    matchString.str("");
    matchString << norm(velocity) << " " << norm(acceleration) ;
    putText(frame, matchString.str(), Point(0, 15), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255));

    perspectiveTransform(
      (vector<Point2f>){match.tl(), match.br()},
      t, mtxi);
    rectangle(frame, Rect(t.at(0), t.at(1)), Scalar(255, 0, 0), 1);

    imshow("image", frame);
    int key = waitKey(delay);
    if (key == 127) {
      frameNumber = 1860;
      cap.set(CV_CAP_PROP_POS_FRAMES, frameNumber);
    } else if (key > -1) {
      delay = !delay;
      cout << key << " " << frameNumber << "\n";
    }
  }

  return 0;
}
