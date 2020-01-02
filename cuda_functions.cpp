#include "cuda_functions.h"

cv::cuda::GpuMat cuda_dct2(cv::cuda::GpuMat &img, ConstData &constGrids, VarMats &varMats){

	img.copyTo(varMats.doubledMat(cv::Rect(0, 0, constGrids.height, constGrids.width)));

	cv::cuda::flip(img, varMats.Mat, 0);
	varMats.Mat.copyTo(varMats.doubledMat(cv::Rect(0, constGrids.width, constGrids.height, constGrids.width)));
	cv::cuda::flip(img, varMats.Mat, 1);
	varMats.Mat.copyTo(varMats.doubledMat(cv::Rect(constGrids.height, 0, constGrids.height, constGrids.width)));
	cv::cuda::flip(img, varMats.Mat, -1);
	varMats.Mat.copyTo(varMats.doubledMat(cv::Rect(constGrids.height, constGrids.width, constGrids.height, constGrids.width)));

	cv::cuda::dft(varMats.doubledMat, varMats.fftOut, varMats.doubledMat.size());
	varMats.fftOut(cv::Rect(0, 0, constGrids.height, constGrids.width)).copyTo(varMats.complexMat);
    cv::cuda::split(varMats.complexMat, varMats.complexArray);

	cv::cuda::multiply(varMats.complexArray[0], constGrids.cudaCosDCT, varMats.ch1);
	cv::cuda::multiply(varMats.complexArray[1], constGrids.cudaSinDCT, varMats.ch2);
	cv::cuda::add(varMats.ch1, varMats.ch2, varMats.outMat);    

	return varMats.outMat;
}

cv::cuda::GpuMat idct(cv::cuda::GpuMat &img, ConstData &constGrids, VarMats &varMats){

    cv::cuda::multiply(img, constGrids.cudaCosIDCT, varMats.ch1);
    cv::cuda::multiply(img, constGrids.cudaSinIDCT, varMats.ch2);

    varMats.complexArray[0] = (varMats.ch1.clone());
    varMats.complexArray[1] = (varMats.ch2.clone());
    cv::cuda::merge(varMats.complexArray, varMats.ifftIn);
    
    cv::cuda::dft(varMats.ifftIn, varMats.ifftIn, varMats.ifftIn.size(), cv::DFT_ROWS+cv::DFT_INVERSE+cv::DFT_SCALE);
    cv::cuda::split(varMats.ifftIn, varMats.complexArray);

    varMats.outMat = varMats.complexArray[0];
    varMats.outMat.convertTo(varMats.outMat, varMats.outMat.type(), 512);
    
	cv::cuda::flip(varMats.outMat, varMats.Mat, 1);
    invertArray(varMats.outMat, varMats.Mat, varMats.ca);

    return varMats.ca;
}

cv::cuda::GpuMat cuda_idct2(cv::cuda::GpuMat &img, ConstData &constGrids, VarMats &varMats){

    varMats.z.release();
    cv::cuda::transpose(idct(img, constGrids, varMats), varMats.x);
    cv::cuda::transpose(idct(varMats.x, constGrids, varMats), varMats.z);
    varMats.x.release();
    
    return varMats.z;
}


cv::cuda::GpuMat cudaLaplacian(cv::cuda::GpuMat &img, ConstData &constGrids, VarMats &varMats){

    cv::cuda::multiply(cuda_dct2(img, constGrids, varMats), constGrids.cudaGridLaplacian, varMats.ca);
    varMats.ca = cuda_idct2(varMats.ca, constGrids, varMats);
    varMats.ca.convertTo(varMats.ca, varMats.ca.type(), -4*M_PI*M_PI/(constGrids.height*constGrids.width));

    return varMats.ca;
}

cv::cuda::GpuMat cudaiLaplacian(cv::cuda::GpuMat &img, ConstData &constGrids, VarMats &varMats){

    cv::cuda::divide(cuda_dct2(img, constGrids, varMats), constGrids.cudaGridLaplacian, varMats.ica);
    varMats.ica = cuda_idct2(varMats.ica, constGrids, varMats);
    varMats.ica.convertTo(varMats.ica, varMats.ica.type(), (constGrids.height*constGrids.width)/(-4*M_PI*M_PI));
    return varMats.ica;
}

cv::cuda::GpuMat deltaPhi(cv::cuda::GpuMat &img, ConstData &constGrids, VarMats &varMats){

    cudaSin(img, varMats.imgSin);
    cudaCos(img, varMats.imgCos);
    cv::cuda::multiply(cudaLaplacian(varMats.imgSin, constGrids, varMats), varMats.imgCos, varMats.a1);
    cv::cuda::multiply(cudaLaplacian(varMats.imgCos, constGrids, varMats), varMats.imgSin, varMats.a2);
    cv::cuda::subtract(varMats.a1, varMats.a2, varMats.a1);

    return cudaiLaplacian(varMats.a1, constGrids, varMats);
}


cv::Scalar cudaMean(cv::cuda::GpuMat &img){
    return cv::Scalar(cv::cuda::sum(img)[0]/img.cols/img.rows);
}

void phaseUnwrap(cv::cuda::GpuMat &img, ConstData &constGrids, VarMats &varMats){

    varMats.phi1 = deltaPhi(img, constGrids, varMats);
    cv::cuda::subtract(varMats.phi1, cudaMean(varMats.phi1), varMats.phi1);
    cv::cuda::add(varMats.phi1, cudaMean(img), varMats.phi1);

    cv::cuda::subtract(varMats.phi1, img, varMats.k1);
    varMats.k1.convertTo(varMats.k1, varMats.k1.type(), 0.5/M_PI);

    cudaRound(varMats.k1, varMats.k1round);
    varMats.k1round.convertTo(varMats.k1round, varMats.k1round.type(), 2*M_PI);

    cv::cuda::add(img, varMats.k1round, varMats.phi2);
    
    for(auto i = 0; i < 3; i++){
        cv::cuda::subtract(varMats.phi2, varMats.phi1, varMats.error);
        cv::cuda::subtract(varMats.phi1, cudaMean(varMats.phi1), varMats.phi1);
        cv::cuda::add(varMats.phi1, deltaPhi(varMats.error, constGrids, varMats), varMats.phi1);
        cv::cuda::add(varMats.phi1, cudaMean(varMats.phi2), varMats.phi1);
        
        cv::cuda::subtract(varMats.phi1, img, varMats.k2);
        varMats.k2.convertTo(varMats.k2, varMats.k2.type(), 0.5/M_PI);
        cudaRound(varMats.k2, varMats.k2round);
        varMats.k2round.convertTo(varMats.k2round, varMats.k2round.type(), 2*M_PI);

        cv::cuda::add(img, varMats.k2round, varMats.phi2);
        varMats.k1round = varMats.k2round.clone();
    }
    
    img = varMats.phi2;
    /*//SAVING IMG
    std::string s1 = "Reconstruction";
    std::string s2 = ".bmp";
    cv::Mat cudaResult;
    varMats.phi2.download(cudaResult);
    double min, max;
    cv::minMaxLoc(cudaResult, &min, &max);
    cudaResult += 5;
    cv::normalize(cudaResult, cudaResult, 255, 0, cv::NORM_MINMAX);
    std::string s = std::to_string(1);
    cv::imwrite( s1+s+s2, cudaResult);
    */
}
