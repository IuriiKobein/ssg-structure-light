#include "cuda_functions.h"

cv::cuda::GpuMat cuda_dct2(cv::cuda::GpuMat &img, cv::cuda::GpuMat &cudaCos, cv::cuda::GpuMat &cudaSin){
    int height = img.rows;
    int width = img.cols;

    auto y = cv::cuda::GpuMat(2*height, 2*width, CV_32FC1);
	img.copyTo(y(cv::Rect(0, 0, height, width)));
    cv::cuda::GpuMat tempMat;
	cv::cuda::flip(img, tempMat, 0);
	tempMat.copyTo(y(cv::Rect(0, width, height, width)));
	cv::cuda::flip(img, tempMat, 1);
	tempMat.copyTo(y(cv::Rect(height, 0, height, width)));
	cv::cuda::flip(img, tempMat, -1);
	tempMat.copyTo(y(cv::Rect(height, width, height, width)));

	cv::cuda::GpuMat fftOutput = cv::cuda::GpuMat(2*height, 2*width, CV_32FC1);
	cv::cuda::dft(y, fftOutput, y.size());

	cv::cuda::GpuMat output;
	fftOutput(cv::Rect(0, 0, height, width)).copyTo(output);
	
    std::vector<cv::cuda::GpuMat> complexArray(2);
    cv::cuda::split(output, complexArray);

    cv::cuda::GpuMat ch1, ch2;
	cv::cuda::multiply(complexArray[0], cudaCos, ch1);
	cv::cuda::multiply(complexArray[1], cudaSin, ch2);

	cv::cuda::add(ch1, ch2, ch1);    

	return ch1;
}

cv::cuda::GpuMat idct(cv::cuda::GpuMat &img, cv::cuda::GpuMat &cudaCos, cv::cuda::GpuMat &cudaSin){

    int height = img.rows;
    int width = img.cols;
    cv::cuda::GpuMat ch1, ch2;
    cv::cuda::multiply(img, cudaCos, ch1);
    cv::cuda::multiply(img, cudaSin, ch2);

    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::GpuMat merged;
    channels.push_back(ch1);
    channels.push_back(ch2);
    cv::cuda::merge(channels, merged);
    
    cv::cuda::dft(merged, merged, merged.size(), cv::DFT_ROWS+cv::DFT_INVERSE+cv::DFT_SCALE);

    std::vector<cv::cuda::GpuMat> complexArray(2);
    cv::cuda::split(merged, complexArray);

    cv::cuda::GpuMat real = complexArray[0];
    real.convertTo(real, real.type(), 512);
    auto y = cv::cuda::GpuMat(height, width, CV_32FC1);
    cv::cuda::GpuMat tempMat;
	cv::cuda::flip(real, tempMat, 1);

    invertArray(real, tempMat, y);

    return y;
}

cv::cuda::GpuMat cuda_idct2(cv::cuda::GpuMat &img, cv::cuda::GpuMat &cudaCos, cv::cuda::GpuMat &cudaSin){

    auto b = idct(img, cudaCos, cudaSin);
    cv::cuda::GpuMat x;
    cv::cuda::transpose(b, x);
    x = idct(x, cudaCos, cudaSin);
    cv::cuda::GpuMat z;
    cv::cuda::transpose(x, z);

    return z;
}


cv::cuda::GpuMat cudaLaplacian(cv::cuda::GpuMat &img, cv::cuda::GpuMat &cudaCosDCT, cv::cuda::GpuMat &cudaSinDCT, cv::cuda::GpuMat &cudaCosIDCT, cv::cuda::GpuMat &cudaSinIDCT, cv::cuda::GpuMat &cudaGridLaplacian){

    auto ca = cuda_dct2(img, cudaCosDCT, cudaSinDCT);

    cv::cuda::multiply(ca, cudaGridLaplacian, ca);
    ca = cuda_idct2(ca, cudaCosIDCT, cudaSinIDCT);
    ca.convertTo(ca, ca.type(), -4*M_PI*M_PI/(img.cols*img.rows));

    return ca;
}

cv::cuda::GpuMat cudaiLaplacian(cv::cuda::GpuMat &img, cv::cuda::GpuMat &cudaCosDCT, cv::cuda::GpuMat &cudaSinDCT, cv::cuda::GpuMat &cudaCosIDCT, cv::cuda::GpuMat &cudaSinIDCT, cv::cuda::GpuMat &cudaGridLaplacian){

    auto ca = cuda_dct2(img, cudaCosDCT, cudaSinDCT);
    cv::cuda::divide(ca, cudaGridLaplacian, ca);
    ca = cuda_idct2(ca, cudaCosIDCT, cudaSinIDCT);
    ca.convertTo(ca, ca.type(), (img.cols*img.rows)/(-4*M_PI*M_PI));
    
    return ca;
}

cv::cuda::GpuMat deltaPhi(cv::cuda::GpuMat &img, cv::cuda::GpuMat &cudaCosDCT, cv::cuda::GpuMat &cudaSinDCT, cv::cuda::GpuMat &cudaCosIDCT, cv::cuda::GpuMat &cudaSinIDCT, cv::cuda::GpuMat &cudaGridLaplacian){

    cv::cuda::GpuMat imgSin = cv::cuda::GpuMat(512, 512, CV_32FC1);
    cv::cuda::GpuMat imgCos = cv::cuda::GpuMat(512, 512, CV_32FC1);
    cudaSin(img, imgSin);
    cudaCos(img, imgCos);

    auto a1 = cudaLaplacian(imgSin, cudaCosDCT, cudaSinDCT, cudaCosIDCT, cudaSinIDCT, cudaGridLaplacian);
    auto a2 = cudaLaplacian(imgCos, cudaCosDCT, cudaSinDCT, cudaCosIDCT, cudaSinIDCT, cudaGridLaplacian);

    cv::cuda::multiply(a1, imgCos, a1);
    cv::cuda::multiply(a2, imgSin, a2);
    cv::cuda::subtract(a1, a2, a1);

    auto x = cudaiLaplacian(a1, cudaCosDCT, cudaSinDCT, cudaCosIDCT, cudaSinIDCT, cudaGridLaplacian);

    return x;
}

cv::Scalar cudaMean(cv::cuda::GpuMat &img){
    return cv::Scalar(cv::cuda::sum(img)[0]/img.cols/img.rows);
}

void phaseUnwrap(cv::cuda::GpuMat &img, cv::cuda::GpuMat &cudaCosDCT, cv::cuda::GpuMat &cudaSinDCT, cv::cuda::GpuMat &cudaCosIDCT, cv::cuda::GpuMat &cudaSinIDCT, cv::cuda::GpuMat &cudaGridLaplacian){

    //auto phix = deltaPhi(img, cudaCosDCT, cudaSinDCT, cudaCosIDCT, cudaSinIDCT, cudaGridLaplacian);
    auto phi1 = deltaPhi(img, cudaCosDCT, cudaSinDCT, cudaCosIDCT, cudaSinIDCT, cudaGridLaplacian);
    cv::cuda::subtract(phi1, cudaMean(phi1), phi1);
    cv::cuda::add(phi1, cudaMean(img), phi1);

    cv::cuda::GpuMat k1;
    cv::cuda::subtract(phi1, img, k1);
    k1.convertTo(k1, k1.type(), 0.5/M_PI);

    cv::cuda::GpuMat k1round = cv::cuda::GpuMat(512, 512, CV_32FC1);
    cudaRound(k1, k1round);
    k1round.convertTo(k1round, k1round.type(), 2*M_PI);

    cv::cuda::GpuMat k2, phi2;
    cv::cuda::GpuMat k2round = cv::cuda::GpuMat(512, 512, CV_32FC1);
    cv::cuda::add(img, k1round, phi2);

    cv::cuda::GpuMat error, phi_error;

    for(auto i = 0; i < 3; i++){
        cv::cuda::subtract(phi2, phi1, error);
        phi_error = deltaPhi(error, cudaCosDCT, cudaSinDCT, cudaCosIDCT, cudaSinIDCT, cudaGridLaplacian);
        cv::cuda::subtract(phi1, cudaMean(phi1), phi1);
        cv::cuda::add(phi1, phi_error, phi1);
        cv::cuda::add(phi1, cudaMean(phi2), phi1);
        
        cv::cuda::subtract(phi1, img, k2);
        k2.convertTo(k2, k2.type(), 0.5/M_PI);
        cudaRound(k2, k2round);
        k2round.convertTo(k2round, k2round.type(), 2*M_PI);

        cv::cuda::add(img, k2round, phi2);
        k1round = k2round.clone();
    }
    /* SAVING IMG
    std::string s1 = "Reconstruction";
    std::string s2 = ".bmp";
    cv::Mat cudaResult;
    phi2.download(cudaResult);
    cudaResult += 5;
    cv::normalize(cudaResult, cudaResult, 0, 255, cv::NORM_MINMAX);
    std::cout << cudaResult(cv::Rect(0, 0, 5, 5)) << std::endl;
    std::string s = std::to_string(i);
    cv::imwrite( s1+s+s2, cudaResult);
    */
}