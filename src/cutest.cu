




#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <unistd.h>

#include <thrust/sequence.h>
#define BLOCK_SIZE 16

double calc_theta(const int m){
    double nmsum=0;
    for(int i=1;i<2*m;i++){
        nmsum+=1.0/static_cast<double>(i);
    }
    return((1/nmsum)/(2*m+1/nmsum));
}


__global__
void ldshrink(float* S, const float* mapd, const int p, const float m, const float ne, const float cutoff, const double theta)
{
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    int i = p - 2 - std::floor(std::sqrt(-8*k + 4*p*(p-1)-7)/2.0 - 0.5);
    int j = k + i + 1 - p*(p-1)/2 + (p-i)*((p-i)-1)/2;
    if(i<p&&j<p) {
        auto tsi = S[i * p + i];
        auto tsj = S[j * p + j];

        auto shrinkage = std::exp(-(4 * ne * std::abs(mapd[j] - mapd[i]) / 100) / (2 * m));
        shrinkage = shrinkage < cutoff ? 0 : shrinkage;
        auto tS = 1 / std::sqrt(tsi + 0.5 * theta * (1 - 0.5 * theta)) * ((1 - theta) * (1 - theta)) * S[i * p + j] *
                  shrinkage * (1 / std::sqrt(tsj));
        S[j * p + i] = tS;
        S[i * p + j] = tS;
    }

}
__global__
void zero_diagonal(float* S,const int p) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    S[i*p+i]=1;
}
__global__
void row_diagonal(float* S,const int p) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    S[i*p+i]=i;
}

__global__
void idx_check(float* rowm,float *colm,const int p) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    int i = p - 2 - std::floor(std::sqrt(-8*k + 4*p*(p-1)-7)/2.0 - 0.5);
    int j = k + i + 1 - p*(p-1)/2 + (p-i)*((p-i)-1)/2;
    if(i<p&&j<p) {
        rowm[i * p + j] = i;
        rowm[j * p + i] = j;

        colm[i * p + j] = j;
        colm[j * p + i] = i;
    }
}

std::pair<std::vector<float>,std::vector<float> >id_check(const size_t p) {

    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
    // maximum occupancy for a full device launch
    int gridSize;
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                                        idx_check, 0, 0);
    // Round up according to array size
    int arrayCount = (p*p-p)/2;
    gridSize = (arrayCount + blockSize - 1) / blockSize;
    thrust::device_vector<float> d_cov1(p*p);
    thrust::device_vector<float> d_cov2(p*p);
    idx_check<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_cov1.data()),thrust::raw_pointer_cast(d_cov2.data()),p);
    row_diagonal<<<1,p>>>(thrust::raw_pointer_cast(d_cov1.data()),p);
    row_diagonal<<<1,p>>>(thrust::raw_pointer_cast(d_cov2.data()),p);
    std::vector<float> res_data1(p*p);
    std::vector<float> res_data2(p*p);
    thrust::copy(d_cov1.begin(),d_cov1.end(),res_data1.begin());
    thrust::copy(d_cov2.begin(),d_cov2.end(),res_data2.begin());
    return(std::make_pair(res_data1,res_data2));
}


    std::vector<float> cuda_cov(float*X, const size_t n, const size_t p,const float* mapd,float m,float ne,float cutoff) {
        cublasHandle_t handle; // CUBLAS context

    float alpha = 1.0f;
    float beta = 0.0f; // bet =1
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
        // maximum occupancy for a full device launch
    int gridSize;
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                                            ldshrink, 0, 0);
        int arrayCount = (p*p-p)/2;
        gridSize = (arrayCount + blockSize - 1) / blockSize;

        thrust::device_vector<float> d_cov1(p*p);
    thrust::device_vector<float> d_cov2(p*p);
    thrust::device_vector<float> d_covResult(p*p);
    const thrust::device_vector<float> d_map(mapd,mapd+p);
    thrust::device_vector<float> d_wholeMatrix(X,X+n*p);
    thrust::device_vector<float> d_meansVec(p); // rowVec of means of trials
    float *meanVecPtr = thrust::raw_pointer_cast(d_meansVec.data());
    float *device2DMatrixPtr = thrust::raw_pointer_cast(d_wholeMatrix.data());

    thrust::device_vector<float> deviceVector(n, 1.0f);

        cublasCreate(&handle);
    auto theta = calc_theta(m);
    auto dimensionSize=p;
    alpha = 1.0f / n;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, dimensionSize, dimensionSize, n, &alpha,
                device2DMatrixPtr, dimensionSize, device2DMatrixPtr, dimensionSize, &beta,
                thrust::raw_pointer_cast(d_cov1.data()), dimensionSize);

        // Mean vector of each column
        alpha = 1.0f;
        cublasSgemv(handle, CUBLAS_OP_N, dimensionSize, n, &alpha, device2DMatrixPtr,
                    dimensionSize, thrust::raw_pointer_cast(deviceVector.data()), 1, &beta, meanVecPtr, 1);

        // MeanVec * transpose(MeanVec) / N*N
        alpha = 1.0f / (n*n);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, dimensionSize, dimensionSize, 1, &alpha,
                    meanVecPtr, 1, meanVecPtr, 1, &beta,
                    thrust::raw_pointer_cast(d_cov2.data()), dimensionSize);

        alpha = 1.0f;
        beta = -1.0f;
        //  (X*transpose(X) / N) -  (MeanVec * transpose(MeanVec) / N*N)
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimensionSize, dimensionSize, &alpha,
                    thrust::raw_pointer_cast(d_cov1.data()), dimensionSize, &beta,
                    thrust::raw_pointer_cast(d_cov2.data()),
                    dimensionSize, thrust::raw_pointer_cast(d_covResult.data()), dimensionSize);

        // Go to other class and calculate its covarianceMatrix


        std::vector<float> res_data(p*p);
        ldshrink<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_covResult.data()),thrust::raw_pointer_cast(d_map.data()),p,m,ne,cutoff,theta);
        zero_diagonal<<<1,p>>>(thrust::raw_pointer_cast(d_covResult.data()),p);
        thrust::copy(d_covResult.begin(),d_covResult.end(),res_data.begin());

        cublasDestroy(handle);
    return res_data;
}
