//
// Created by nwknoblauch on 3/7/19.
//

#include <iostream>

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "cudacov.hpp"
#include "cxxopts.hpp"

#include "xtensor/xview.hpp"
#include <xtensor-io/xhighfive.hpp>
#include "zstd_h5plugin.h"
#include "xtensor/xadapt.hpp"
#include "mkl_cov.hpp"



int main(int argc, char* argv[]){

    register_zstd();


    cxxopts::Options options("cudald", "cuda implementation of ldshrink");

    options.add_options()("i,input", "Input", cxxopts::value<std::string>()->default_value("../data/genotype.h5"))("o,output", "Output", cxxopts::value<std::string>()->default_value("../data/S.h5"));

    auto result = options.parse(argc, argv);


    //    cxxopts::value<std::string>()
    const std::string input_f =result["input"].as<std::string>();
    const std::string output_f =result["output"].as<std::string>();

    HighFive::File file(output_f, HighFive::File::Overwrite);

    auto A = xt::load_hdf5<xt::xtensor<float,2>>(input_f, "dosage");
    auto map = xt::load_hdf5<xt::xtensor<float,1>>(input_f, "map");
    const float m=85;
    const float Ne=11490.672741;
    const float cutoff = 0.001;
    const size_t N=A.shape()[0];
    const size_t p=A.shape()[1];
    xt::xtensor<double,2> mS =mkldshrink(A,map,m,Ne,cutoff);
    auto res=cuda_cov(A.data(),N,p,map.data(),m,Ne,cutoff);
    xt::xtensor<double,2> S = xt::adapt(res,{p,p});
    xt::dump(file, "/S", S);
    xt::dump(file, "/mkl_S", mS);

    // auto resi=id_check(p);
    // xt::xtensor<double,2> rowI = xt::adapt(resi.first,{p,p});
    // xt::xtensor<double,2> colI = xt::adapt(resi.second,{p,p});


    // xt::dump(file, "/rowI", rowI);
    // xt::dump(file, "/colI", colI);





}
