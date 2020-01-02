#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <npp.h>
#include <fstream>
#include <cufft.h>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <chrono>
#include "cuda_functions.h"
#include "cuda_impl.h"
#include <iomanip>
const int IMG_WIDTH = 512;
const int IMG_HEIGHT = 512;


cv::Mat ReadMatFromTxt(std::string filename, int rows,int cols)
{
    float m;
    cv::Mat out = cv::Mat::zeros(rows, cols, CV_32F);//Matrix to store values

    std::ifstream fileStream(filename);
    int cnt = 0; //index starts from 0
    while (fileStream >> m)
    {
        int temprow = cnt / cols;
        int tempcol = cnt % cols;
        out.at<float>(temprow, tempcol) = m;
        cnt++;
    }
    return out;
}

cv::Mat dct2(cv::Mat &img){
    int height = img.rows;
    int width = img.cols;

	auto gridCos = cv::Mat(height, width, CV_32FC1);
	auto gridSin = cv::Mat(height, width, CV_32FC1);

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
			if(i > 0 && j > 0){
                gridCos.at<float>(i, j) = (2/sqrt(height*width)/4)*cos(((j)*height + (i)*width)*(M_PI/(2*width*height)));
			    gridSin.at<float>(i, j) = (2/sqrt(height*width)/4)*sin(((j)*height + (i)*width)*(M_PI/(2*width*height)));
			} else if (i == 0 && j > 0){
                gridCos.at<float>(i, j) = (2/(sqrt(2)*width)/4)*cos(((j)*height + (i)*width)*(M_PI/(2*width*height)));
			    gridSin.at<float>(i, j) = (2/(sqrt(2)*width)/4)*sin(((j)*height + (i)*width)*(M_PI/(2*width*height)));
			} else if (i > 0 && j == 0){
                gridCos.at<float>(i, j) = (2/(sqrt(2)*height)/4)*cos(((j)*height + (i)*width)*(M_PI/(2*width*height)));
			    gridSin.at<float>(i, j) = (2/(sqrt(2)*height)/4)*sin(((j)*height + (i)*width)*(M_PI/(2*width*height)));
			
			} else if (i == 0 && j == 0){
                gridCos.at<float>(i, j) = (1/sqrt(height*width)/4)*cos(((j)*height + (i)*width)*(M_PI/(2*width*height)));
			    gridSin.at<float>(i, j) = (1/sqrt(height*width)/4)*sin(((j)*height + (i)*width)*(M_PI/(2*width*height)));
			
			}
        }
    }

    auto y = cv::Mat(2*height, 2*width, CV_32FC1);
	img.copyTo(y(cv::Rect(0, 0, height, width)));
    cv::Mat tempMat;
	cv::flip(img, tempMat, 0);
	tempMat.copyTo(y(cv::Rect(0, width, height, width)));
	cv::flip(img, tempMat, 1);
	tempMat.copyTo(y(cv::Rect(height, 0, height, width)));
	cv::flip(img, tempMat, -1);
	tempMat.copyTo(y(cv::Rect(height, width, height, width)));

	cv::Mat fftOutput;
	cv::dft(y, fftOutput, cv::DFT_COMPLEX_OUTPUT);

	cv::Mat output;
	fftOutput(cv::Rect(0, 0, height, width)).copyTo(output);
	
    std::vector<cv::Mat> complexArray(2);
    cv::split(output, complexArray);
    

	cv::multiply(complexArray[0], gridCos, gridCos);
	cv::multiply(complexArray[1], gridSin, gridSin);

	cv::add(gridCos, gridSin, gridSin);    

    std::cout << gridSin(cv::Rect(0, 0, 5, 5)) << std::endl;
	return gridSin;
}

cv::Mat idct(cv::Mat imgRow, bool flag = false){
    int width = imgRow.cols;

	auto gridCos = cv::Mat(1, width, CV_32FC1);
	auto gridSin = cv::Mat(1, width, CV_32FC1);
    
    for(int i=0; i<width; i++){
        if(i > 0){
            gridCos.at<float>(0, i) = sqrt(2*width)*cos(M_PI*i/(2*width));
            gridSin.at<float>(0, i) = sqrt(2*width)*sin(M_PI*i/(2*width));
        } else if (i == 0){
            gridCos.at<float>(0, i) = sqrt(width)*cos(M_PI*i/(2*width));
            gridSin.at<float>(0, i) = sqrt(width)*sin(M_PI*i/(2*width));
        }

    }
    
    cv::multiply(imgRow, gridCos, gridCos);
    cv::multiply(imgRow, gridSin, gridSin);

    std::vector<cv::Mat> channels;
    cv::Mat merged;
    channels.push_back(gridCos);
    channels.push_back(gridSin);
    cv::merge(channels, merged);
    
    cv::dft(merged, merged, cv::DFT_INVERSE+cv::DFT_SCALE);

    std::vector<cv::Mat> complexArray(2);
    cv::split(merged, complexArray);

    cv::Mat real = complexArray[0];

    auto y = cv::Mat(1, width, CV_32FC1);
    cv::Mat tempMat;
	cv::flip(real, tempMat, 1);
    for (int i = 0; i< width/2; i++)
    {
        y.at<float>(0, 2*i) = real.at<float>(0, i);
        y.at<float>(0, 2*i+1) = tempMat.at<float>(0, i);
    }
    if (flag)
        std::cout << gridCos(cv::Rect(120, 0, 20, 1)) << " ";
    return y;
}

cv::Mat idct2(cv::Mat img){

    int height = img.rows;
    int width = img.cols;

    cv::Mat b = cv::Mat(height, width, CV_32FC1);

    for (auto i=0; i < height; i++){
        if(i != 300)
            idct(img.row(i)).copyTo(b.row(i));
        else
            idct(img.row(i)).copyTo(b.row(i));
    }
    std::cout << std::endl;
    cv::transpose(b, b);
    for (auto i=0; i < width; i++){
        if(i != 300)
            idct(b.row(i)).copyTo(b.row(i));
        else
            idct(b.row(i)).copyTo(b.row(i));
    }

    cv::transpose(b, b);
    return b;
}

cv::Mat Laplacian(cv::Mat &img){

    int height = img.rows;
    int width = img.cols;

    cv::Mat grid = cv::Mat(height, width, CV_32FC1);
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            grid.at<float>(i,j) = (i+1)*(i+1) + (j+1)*(j+1);
        }
    }
    auto ca = dct2(img);
    
    cv::multiply(ca, grid, ca);
    ca = idct2(ca);
    ca *= -4*M_PI*M_PI/(width*height);

    return ca;
}

cv::Mat iLaplacian(cv::Mat &img){

    int height = img.rows;
    int width = img.cols;

    cv::Mat grid = cv::Mat(height, width, CV_32FC1);
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            grid.at<float>(i,j) = (i+1)*(i+1) + (j+1)*(j+1);
        }
    }
    auto ca = dct2(img);
    cv::divide(ca, grid, ca);

    ca = idct2(ca);
    ca *= (width*height)/(-4*M_PI*M_PI);

    return ca;
}

void CreateCudaGrids(cv::Mat img, cv::cuda::GpuMat &cudaCosDCT, cv::cuda::GpuMat &cudaSinDCT, cv::cuda::GpuMat &cudaCosIDCT, cv::cuda::GpuMat &cudaSinIDCT,  cv::cuda::GpuMat &cudaGridLaplacian){

    int height = img.rows;
    int width = img.cols;

    auto gridCosDCT = cv::Mat(height, width, CV_32FC1);
	auto gridSinDCT = cv::Mat(height, width, CV_32FC1);
    auto gridCosIDCT = cv::Mat(height, width, CV_32FC1);
	auto gridSinIDCT = cv::Mat(height, width, CV_32FC1);
    auto gridLaplacian = cv::Mat(height, width, CV_32FC1);

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
			if(i > 0 && j > 0){
                gridCosDCT.at<float>(i, j) = (2/sqrt(height*width)/4)*cos(((j)*height + (i)*width)*(M_PI/(2*width*height)));
			    gridSinDCT.at<float>(i, j) = (2/sqrt(height*width)/4)*sin(((j)*height + (i)*width)*(M_PI/(2*width*height)));
			} else if (i == 0 && j > 0){
                gridCosDCT.at<float>(i, j) = (2/(sqrt(2)*width)/4)*cos(((j)*height + (i)*width)*(M_PI/(2*width*height)));
			    gridSinDCT.at<float>(i, j) = (2/(sqrt(2)*width)/4)*sin(((j)*height + (i)*width)*(M_PI/(2*width*height)));
			} else if (i > 0 && j == 0){
                gridCosDCT.at<float>(i, j) = (2/(sqrt(2)*height)/4)*cos(((j)*height + (i)*width)*(M_PI/(2*width*height)));
			    gridSinDCT.at<float>(i, j) = (2/(sqrt(2)*height)/4)*sin(((j)*height + (i)*width)*(M_PI/(2*width*height)));
			} else if (i == 0 && j == 0){
                gridCosDCT.at<float>(i, j) = (1/sqrt(height*width)/4)*cos(((j)*height + (i)*width)*(M_PI/(2*width*height)));
			    gridSinDCT.at<float>(i, j) = (1/sqrt(height*width)/4)*sin(((j)*height + (i)*width)*(M_PI/(2*width*height)));
			}
            if(j > 0){
                gridCosIDCT.at<float>(i, j) = sqrt(2*width)*cos(M_PI*j/(2*width));
                gridSinIDCT.at<float>(i, j) = sqrt(2*width)*sin(M_PI*j/(2*width));
            } else if (j == 0){
                gridCosIDCT.at<float>(i, j) = sqrt(width)*cos(M_PI*j/(2*width));
                gridSinIDCT.at<float>(i, j) = sqrt(width)*sin(M_PI*j/(2*width));
            }
            gridLaplacian.at<float>(i,j) = (i+1)*(i+1) + (j+1)*(j+1);
        }
    }

    cudaCosDCT.upload(gridCosDCT);
    cudaSinDCT.upload(gridSinDCT);
    cudaCosIDCT.upload(gridCosIDCT);
    cudaSinIDCT.upload(gridSinIDCT);
    cudaGridLaplacian.upload(gridLaplacian);
}


int main(){
    
    auto img = ReadMatFromTxt("test.txt", IMG_HEIGHT, IMG_WIDTH);
    cv::cuda::GpuMat cudaImg;
    cudaImg.upload(img);

    cv::cuda::GpuMat cudaCosDCT, cudaSinDCT;
    cv::cuda::GpuMat cudaCosIDCT, cudaSinIDCT;
    cv::cuda::GpuMat cudaGridLaplacian;
    CreateCudaGrids(img, cudaCosDCT, cudaSinDCT, cudaCosIDCT, cudaSinIDCT, cudaGridLaplacian);

    auto ts = std::chrono::high_resolution_clock::now();
    phaseUnwrap(cudaImg, cudaCosDCT, cudaSinDCT, cudaCosIDCT, cudaSinIDCT, cudaGridLaplacian);
    auto te = std::chrono::high_resolution_clock::now();
    std::cout << "Time GPU Laplacian: " << std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count() <<std::endl;
    
    ts = std::chrono::high_resolution_clock::now();
    phaseUnwrap(cudaImg, cudaCosDCT, cudaSinDCT, cudaCosIDCT, cudaSinIDCT, cudaGridLaplacian);
    te = std::chrono::high_resolution_clock::now();
    std::cout << "Time GPU Laplacian: " << std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count() <<std::endl;

}