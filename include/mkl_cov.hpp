#pragma once


xt::xtensor<float,2> mkldshrink(xt::xtensor<float,2> xtd,xt::xtensor<float,1> &map,const float m,const float Ne,const float cutoff);