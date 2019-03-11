
#include <numeric>                    // Standard library import for std::accumulate
#include "xtensor/xtensor.h"
#include "mkl.h"
#include "mkl_vsl.h"
#include "xarray.hpp"
#include "xindex_view.hpp"
#include <xtensor/xdynamic_view.hpp>
#include <xtensor/xview.hpp>


float calc_theta(const int m){
  float nmsum=0;
  for(int i=1;i<2*m;i++){
    nmsum+=1.0/static_cast<float>(i);
  }
  return((1/nmsum)/(2*m+1/nmsum));
}

xt::xtensor<float,2> mkldshrink(xt::xtensor<float,2> x,xt::xtensor<float,1> map,const float m,const float Ne,const float cutoff){

    int i;
    VSLSSTaskPtr task;
    //float x[DIM][N];  /* matrix of data block */
    const int DIM = x.shape()[1];
    const int N=x.shape()[0];
    //xt::xtensor<float,2> cp = xt::zeros({DIM, DIM});
    xt::xtensor<float,2> cr(xt::xtensor<float,2>::shape_type{DIM, DIM});
    xt::xtensor<float,1> mean(xt::xtensor<float,1>::shape_type{DIM});
    //const int p=map.size();

    std::fill(cr.begin(),cr.end(),0);
    std::fill(mean.begin(),mean.end(),0);
    //float cp[DIM*DIM], cor[DIM*DIM],;
    //float mean[DIM];
    float w[2];
    MKL_INT p, n, xstorage, corstorage, cpstorage;
    int status;

    /* Parameters of the task and initialization */
    p = DIM;
    n = N;
    xstorage   = VSL_SS_MATRIX_STORAGE_ROWS;
    corstorage = VSL_SS_MATRIX_STORAGE_FULL;
    //cp         = VSL_SS_MATRIX_STORAGE_FULL;

   //    w[0] = 0.0; /* sum of weights */
    //  w[1] = 0.0; /* sum of squares of weights */


    /* Create a task */
    status = vsldSSNewTask( &task, &p, &n, &xstorage, x.data(), nullptr, nullptr );
    //status = vsldSSEditTask  ( task, VSL_SS_ED_ACCUM_WEIGHT, w    );
    status = vsldSSEditCovCor( task, mean.data(), cr.data(), &corstorage, nullptr, nullptr );
    status = vsldSSCompute( task, VSL_SS_COV,
                                  VSL_SS_METHOD_FAST);
    status = vslSSDeleteTask( &task );

    //auto diff_a =xt::meshgrid(map,map);
    auto td = xt::abs(xt::view(map,xt::all(),xt::newaxis())-xt::reshape_view(map,{1,p}));
    xt::xtensor<float,2> shrinkage=xt::exp(-(4*Ne*td/100)/(2*m));
    xt::filter(shrinkage,shrinkage<=cutoff)=0;
    float theta = calc_theta(m);
    cr=(shrinkage*cr)*((1-theta)*(1-theta));
    //xt::diagional(cr)+=);
    auto S_d=1/xt::sqrt(xt::diagonal(cr)+(0.5*theta*(1-0.5*theta)));


    return(xt::view(S_d,xt::all(),xt::newaxis())*cr*xt::reshape_view(S_d,{1,p}));
}
