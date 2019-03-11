//
// Created by nwknoblauch on 3/8/19.
//

#ifndef CUDALD_CUDACOV_HPP
#define CUDALD_CUDACOV_HPP


std::vector<float> cuda_cov(float*X, const size_t n, const size_t p,const float* mapd,float m,float ne,float cutoff);
std::pair<std::vector<float>,std::vector<float> >id_check(const size_t p);
xt::xtensor<float,2> mkldshrink(xt::xtensor<float,2> x,xt::xtensor<float,1> map,const float m,const float Ne,const float cutoff);

#endif //CUDALD_CUDACOV_HPP
