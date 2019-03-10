//
// Created by nwknoblauch on 3/7/19.
//

#include <iostream>

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "cudacov.hpp"

#include "xtensor/xview.hpp"
#include <xtensor-io/xhighfive.hpp>
#include "zstd_h5plugin.h"
#include "xtensor/xadapt.hpp"



int main() {

    register_zstd();
    const std::string input_f = "/home/nwknoblauch/Dropbox/Repos/LD/data/genotype.h5";
    const std::string output_f = "/home/nwknoblauch/Dropbox/Repos/LD/data/S.h5";
    HighFive::File file(output_f, HighFive::File::Overwrite);

    auto A = xt::load_hdf5<xt::xtensor<float,2>>(input_f, "dosage");
    auto map = xt::load_hdf5<xt::xtensor<float,1>>(input_f, "map");
    const float m=85;
    const float Ne=11490.672741;
    const float cutoff = 0.001;
    const size_t N=A.shape()[0];
    const size_t p=A.shape()[1];
    auto res=cuda_cov(A.data(),N,p,map.data(),m,Ne,cutoff);
    auto resi=id_check(p);
    xt::xtensor<double,2> rowI = xt::adapt(resi.first,{p,p});
    xt::xtensor<double,2> colI = xt::adapt(resi.second,{p,p});
    xt::xtensor<double,2> S = xt::adapt(res,{p,p});
    xt::dump(file, "/S", S);
    xt::dump(file, "/rowI", rowI);
    xt::dump(file, "/colI", colI);





}
