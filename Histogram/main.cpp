#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <deque>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <math.h>
#include <algorithm>

#define pii pair<int,int>
#define vii vector<pair<int,int> >
#define vint vector<int>
#define dii deque<pair<int,int> >


using namespace cv;
using namespace std;

Mat img, imgb, imgc, imgd, img_out, img_out2, img_out3;
int dilation_size;
int morph_elem = 0;
int morph_size = 0;
int morph_operator = 0;
int const max_operator = 4;
int const max_elem = 2;
int const max_kernel_size = 21;
int blur_size = 1, blur_max = 10;


int gsize = 1, gd = 0;
int bri = 150, con = 0;
int equ = 0;
int thresholdVal = 0;

int edgeThresh = 1;
int lowThreshold = 10;
int const max_lowThreshold = 100;
int canny_ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

 int minPts = 4;
 int epsDist = 2;

int minCluster = 3;
int blockSize = 10;
int meanSubtract = 2;

int abs(int x) {
    return x > 0 ? x : -x;
}

Mat frame, frame2, oldframe, diff, bg, bga[3];

int dist(int x1, int x2, int y1, int y2) {
    return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
}

vii neighborQuery(Mat &m, int mi, int mj) {
    int i,j;
    
    int row = m.rows;
    int col = m.cols;
    
    vii neighbor;
    
    for(i=mi-epsDist; i<=mi+epsDist;i++) {
        for(j=mj-epsDist;j<=mj+epsDist;j++) {
            
            if(i >= 0 && i < row && j >= 0 && j < col) {
                
                if(m.at<unsigned char>(i,j)) {
                    if(dist(i,mi,j,mj) <= epsDist*2) {
                        neighbor.push_back(pii(i,j));
                    }
                }
            }
            
        }
    }
    
    return neighbor;
}

bool mycompare(pii a, pii b) {
    return a.second > b.second;
}

Mat DBScan(Mat input) {
    Mat output_rgb[3];
    output_rgb[0] = Mat::zeros(input.size(), CV_8UC1);
    output_rgb[1] = Mat::zeros(input.size(), CV_8UC1);
    output_rgb[2] = Mat::zeros(input.size(), CV_8UC1);
    Mat output;
    Mat cluster = Mat::zeros(input.size(), CV_32SC1);
    Mat visit = Mat::zeros(input.size(), CV_8UC1);
    Mat noise = Mat::zeros(input.size(), CV_8UC1);
    
    
    int row = input.rows;
    int col = input.cols;
    
    int i,j,k,l;
    
    int clusterNo = 0;
    
    dii queue;
    vii count;
    count.push_back(pii(0,0));
    
    for(i=0;i<row;i++) {
        for(j=0;j<col;j++) {
            
            queue.clear();
            
            if(!visit.at<unsigned char>(i,j)) {
                
                visit.at<unsigned char>(i,j) = 255;
                
                vii neighbor = neighborQuery(input, i, j);
                
                if(neighbor.size() < minPts) {
                    noise.at<unsigned char>(i,j) = 255;
                    //cout << "noise" << rand() << endl;
                } else {
                    clusterNo++;
                    count.push_back(pii(clusterNo,0));
                    int thisCluster = clusterNo;
                    
                    
                    for(k=0;k<neighbor.size();k++) {
                        queue.push_back(neighbor[k]);
                    }
                    cluster.at<int>(i,j) = thisCluster;
                    
                    // Expand cluster
                    
                    while(!queue.empty()) {
                        pii p = queue.front();
                        queue.pop_front();
                        
                        if(!visit.at<unsigned char>(p.first, p.second)) {
                            visit.at<unsigned char>(p.first, p.second) = 255;
                            neighbor = neighborQuery(input, p.first, p.second);
                            
                            if(neighbor.size() >= minPts) {
                                for(k=0;k<neighbor.size();k++) {
                                    queue.push_back(neighbor[k]);
                                }
                            }
                        }
                        if(cluster.at<int>(p.first, p.second) == 0) {
                            cluster.at<int>(p.first, p.second) = thisCluster;
                            count[thisCluster].second++;
                        }
                        

                    }
                    

                }
            }
        }
    }
    
    sort(count.begin(), count.end(), mycompare);
    
    vint top5;
    
    for(i=0;i<5;i++) {
        top5.push_back(count[i].first);
    }
    
    int c;
    
    for(i=0;i<row;i++) {
        for(j=0;j<col;j++) {
            c = cluster.at<int>(i,j);
            if(c && find(top5.begin(), top5.end(), c) != top5.end()) {
                srand(c);
                output_rgb[0].at<unsigned char>(i,j) = rand() % 156 + 100;
                output_rgb[1].at<unsigned char>(i,j) = rand() % 156 + 100;
                output_rgb[2].at<unsigned char>(i,j) = rand() % 156 + 100;
            } else if(c) {
                srand(c);
                output_rgb[0].at<unsigned char>(i,j) = rand() % 50;
                output_rgb[1].at<unsigned char>(i,j) = rand() % 50;
                output_rgb[2].at<unsigned char>(i,j) = rand() % 50;
            }
        }
    }
    
    
    merge(output_rgb, 3, output);
    
//    imshow("cluster", output);
    
    return output;
}

void polyRegression(vii data, double lambda) {
    
    int m = data.size();
    
    // ax^2 + bx + c;
    double a,b,c;
    a = rand();
    b = rand();
    c = rand();
    
    double cost = 0, h, x, y;
    double d_cost = 0;
    
    for(int i=0;i<m;i++) {
        x = data[i].first;
        y = data[i].second;
        h = a*pow(x,2) + b*y + c;
        cost += pow(h - y,2);
    }
    
    
    
}

void render(int, void*) {
    if(!equ)
        img.copyTo(imgb);
    else
        equalizeHist(img, imgb);
    
    if(gsize>=0) {
        GaussianBlur(imgb, imgc, Size(gsize*2+1, gsize*2+1), gd);
    } else {
        imgb.copyTo(imgc);
    }
    
    imgc.convertTo(imgd, -1, 1 + con / 100.0, bri / 10.0 - 15);
    
    
    
    Canny( imgd, img_out, lowThreshold, lowThreshold*canny_ratio, kernel_size );
    
    
    
    int operation = morph_operator + 2;
    
    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    
    /// Apply the specified morphology operation
    morphologyEx( img_out, img_out2, operation, element );
    
    
    //dbscan(img_out2);
    imshow("blur", imgd);
    
    imshow("test", img_out2);
    
    
    threshold(imgd, img_out3, thresholdVal, 255, 1);
    imshow("threshold", img_out3);
    
    
//    Mat img_cluster = DBScan(img_out2);
    
//    imshow("cluster", img_cluster);
    
    imshow("invert", 255-imgd);
    
    Mat imgat, imgat_mean;
    Mat imgat2;
    adaptiveThreshold(imgd, imgat, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, blockSize*2+1, meanSubtract);
    adaptiveThreshold(imgd, imgat_mean, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize*2+1, meanSubtract);
    imshow("adaptive_gaussian", 255-imgat);
    imshow("adaptive_mean", 255-imgat_mean);
    
    imgat2 = DBScan(255-imgat_mean);
    
    imshow("adaptive_cluster", imgat2);
    
    int hist[256] = {0};
    
    for(int i = 0; i<imgd.rows; i++)
    {
        for(int k = 0; k<imgd.cols; k++ )
        {
            int value = imgd.at<unsigned char>(i,k);
            hist[value] = hist[value] + 1;
        }
    }

    Mat histPlot( 500, 256, CV_8UC3 );
    
    for(int i = 0; i < 256; i++) {

        int mag = hist[i];
        line(histPlot,Point(i,histPlot.rows-1),Point(i,histPlot.rows-1-mag/100),Scalar(255,0,0));
    }
    
    imshow("histogram", histPlot);
    
}

int main() {

    
    namedWindow("test", 1);
    namedWindow("blur", 2);
    namedWindow("cluster",4);
    namedWindow("threshold", 3);
    namedWindow("cluster",4);
    namedWindow("invert", 5);
    namedWindow("adaptive_gaussian",6);
    namedWindow("adaptive_mean", 7);
    namedWindow("adaptive_cluster",8);
    namedWindow("params",8);
    namedWindow("histogram", 9);
    
    Mat  params = Mat::zeros(1, 500, CV_8UC1);
    imshow("params", params);
    
    

    
    img = imread("/Users/nuttt/handtest5.jpg");
    
    resize(img, img, Size(400,400));
    
    cvtColor( img, img, CV_BGR2GRAY );
    
    createTrackbar( "Min Threshold:", "test", &lowThreshold, max_lowThreshold, render);
    createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat",
                   "test", &morph_operator, max_operator, render);
    createTrackbar( "Element:\n 0: Rect - 1: Cross - 2: Ellipse", "test",
                   &morph_elem, max_elem, render);
    createTrackbar( "Kernel size:\n 2n +1", "test",&morph_size, max_kernel_size, render);
    
    createTrackbar("GaussianSize", "blur", &gsize, 20, render);
    createTrackbar("SD", "blur", &gd, 20, render);
    createTrackbar("con", "blur", &con, 300, render);
    createTrackbar("bri", "blur", &bri, 300, render);
    createTrackbar("eq", "blur", &equ, 1, render);
    
    createTrackbar("threshold", "threshold", &thresholdVal, 255, render);
    
    createTrackbar("minPts", "params", &minPts, 20, render);
    createTrackbar("eps", "params", &epsDist, 20, render);
    createTrackbar("minCluster", "params", &minCluster, 100, render);
    createTrackbar("blockSize", "params", &blockSize, 20, render);
    createTrackbar("meanSubtract", "params", &meanSubtract, 20, render);

    cout << "Min Threshold: " << lowThreshold << endl;
    cout << "Operator: " << morph_operator << endl;
    cout << "Element: " << morph_elem << endl;
    cout << "Kernel Size: " << morph_size << endl;
    cout << "Gaussian Size: " << gsize << endl;
    cout << "SD: " << gd << endl;
    cout << "Contrast: " << con << endl;
    cout << "Brightness: " << bri << endl;
    cout << "Equalization: " << equ << endl;
    cout << "---------------------------" << endl;
    
    render(NULL, NULL);
    
    waitKey(0);
    
}