#include <math.h>
/* */
#include <nvector/nvector_cuda.h>
#include <sunmatrix/sunmatrix_cusparse.h>
/* */
/*  */
#include "naunet_ode.h"
/*  */
#include "naunet_constants.h"
#include "naunet_macros.h"
#include "naunet_physics.h"

#define IJth(A, i, j) SM_ELEMENT_D(A, i, j)

// clang-format off
__device__ int EvalRates(realtype *k, realtype *y, NaunetData *u_data) {

    realtype nH = u_data->nH;
    realtype Tgas = u_data->Tgas;
    
    
    // clang-format on

    // Some variable definitions from krome
    realtype Te      = Tgas * 8.617343e-5;            // Tgas in eV (eV)
    realtype lnTe    = log(Te);                       // ln of Te (#)
    realtype T32     = Tgas * 0.0033333333333333335;  // Tgas/(300 K) (#)
    realtype invT    = 1.0 / Tgas;                    // inverse of T (1/K)
    realtype invTe   = 1.0 / Te;                      // inverse of T (1/eV)
    realtype sqrTgas = sqrt(Tgas);  // Tgas rootsquare (K**0.5)

    // reaaction rate (k) of each reaction
    // clang-format off
    k[0] = exp(-32.71396786e0 + 13.5365560e0*lnTe-5.73932875e0*(pow(lnTe,
        2)) + 1.56315498e0*(pow(lnTe, 3)) - 0.28770560e0*(pow(lnTe, 4)) +
        3.48255977e-2*(pow(lnTe, 5)) - 2.63197617e-3*(pow(lnTe, 6)) +
        1.11954395e-4*(pow(lnTe, 7)) - 2.03914985e-6*(pow(lnTe, 8)));
        
    if (Tgas>1.0 && Tgas<5500.0) { k[1] = 3.92e-13*pow(invTe, 0.6353e0);  }
        
    k[2] = exp(-28.61303380689232e0 -
        0.7241125657826851e0*lnTe-0.02026044731984691e0*pow(lnTe, 2) -
        0.002380861877349834e0*pow(lnTe, 3) - 0.0003212605213188796e0*pow(lnTe,
        4) - 0.00001421502914054107e0*pow(lnTe, 5) +
        4.989108920299513e-6*pow(lnTe, 6) + 5.755614137575758e-7*pow(lnTe, 7) -
        1.856767039775261e-8*pow(lnTe, 8) - 3.071135243196595e-9*pow(lnTe, 9));
        
    k[3] = exp(-44.09864886e0 + 23.91596563e0*lnTe-10.7532302e0*(pow(lnTe,
        2)) + 3.05803875e0*(pow(lnTe, 3)) - 0.56851189e0*(pow(lnTe, 4)) +
        6.79539123e-2*(pow(lnTe, 5)) - 5.00905610e-3*(pow(lnTe, 6)) +
        2.06723616e-4*(pow(lnTe, 7)) - 3.64916141e-6*(pow(lnTe, 8)));
        
    if (Tgas>1.0 && Tgas<9280.0) { k[4] = 3.92e-13*pow(invTe, 0.6353e0);  }
        
    k[5] = 1.54e-9*(1.e0 +
        0.3e0/exp(8.099328789667e0*invTe))/(exp(40.49664394833662e0*invTe)*pow(Te,
        1.5e0)) + 3.92e-13/pow(Te, 0.6353e0);
        
    k[6] = exp(-68.71040990212001e0 +
        43.93347632635e0*lnTe-18.48066993568e0*pow(lnTe, 2) +
        4.701626486759002e0*pow(lnTe, 3) - 0.7692466334492e0*pow(lnTe, 4) +
        0.08113042097303e0*pow(lnTe, 5) - 0.005324020628287001e0*pow(lnTe, 6) +
        0.0001975705312221e0*pow(lnTe, 7) - 3.165581065665e-6*pow(lnTe, 8));
        
    k[7] = 3.36e-10/sqrTgas/pow((Tgas/1.e3), 0.2e0)/(1 + pow((Tgas/1.e6),
        0.7e0));
        
    k[8] = 6.77e-15*pow(Te, 0.8779e0);
        
    if (Tgas>1.0 && Tgas<1160.0) { k[9] = 1.43e-9;  }
        
    k[10] = exp(-20.06913897587003e0 +
        0.2289800603272916e0*lnTe+0.03599837721023835e0*pow(lnTe, 2) -
        0.004555120027032095e0*pow(lnTe, 3) - 0.0003105115447124016e0*pow(lnTe,
        4) + 0.0001073294010367247e0*pow(lnTe, 5) -
        8.36671960467864e-6*pow(lnTe, 6) + 2.238306228891639e-7*pow(lnTe, 7));
        
    if (Tgas>1.0 && Tgas<6700.0) { k[11] = 1.85e-23*pow(Tgas, 1.8e0);  }
        
    k[12] = 5.81e-16*pow((Tgas/5.62e4), (-0.6657e0*log10(Tgas/5.62e4)));
        
    k[13] = 6.0e-10;
        
    k[14] = exp(-24.24914687731536e0 +
        3.400824447095291e0*lnTe-3.898003964650152e0*pow(lnTe, 2) +
        2.045587822403071e0*pow(lnTe, 3) - 0.5416182856220388e0*pow(lnTe, 4) +
        0.0841077503763412e0*pow(lnTe, 5) - 0.007879026154483455e0*pow(lnTe, 6)
        + 0.0004138398421504563e0*pow(lnTe, 7) - 9.36345888928611e-6*pow(lnTe,
        8));
        
    k[15] = 5.6e-11*exp(-102124.e0*invT)*pow(Tgas, 0.5e0);
        
    k[16] = 1.0670825e-10*pow(Te, 2.012e0)*exp(-4.463e0*invTe)/pow((1.e0 +
        0.2472e0*Te), 3.512e0);
        
    k[17] = exp(-18.01849334273e0 +
        2.360852208681e0*lnTe-0.2827443061704e0*pow(lnTe, 2) +
        0.01623316639567e0*pow(lnTe, 3) - 0.03365012031362999e0*pow(lnTe, 4) +
        0.01178329782711e0*pow(lnTe, 5) - 0.001656194699504e0*pow(lnTe, 6) +
        0.0001068275202678e0*pow(lnTe, 7) - 2.631285809207e-6*pow(lnTe, 8));
        
    if (Tgas>1.0 && Tgas<1160.0) { k[18] = 2.56e-9*pow(Te, 1.78186e0);  }
        
    k[19] = exp(-20.37260896533324e0 +
        1.139449335841631e0*lnTe-0.1421013521554148e0*pow(lnTe, 2) +
        0.00846445538663e0*pow(lnTe, 3) - 0.0014327641212992e0*pow(lnTe, 4) +
        0.0002012250284791e0*pow(lnTe, 5) + 0.0000866396324309e0*pow(lnTe, 6) -
        0.00002585009680264e0*pow(lnTe, 7) + 2.4555011970392e-6*pow(lnTe, 8) -
        8.06838246118e-8*pow(lnTe, 9));
        
    k[20] = 6.5e-9/sqrt(Te);
        
    k[21] = 1.e-8*pow(Tgas, (-0.4e0));
        
    if (Tgas>1.0 && Tgas<617.0) { k[22] = 1.e-8;  }
        
    k[23] = 1.32e-6*pow(Tgas, (-0.76e0));
        
    k[24] = 5.e-7*sqrt(1.e2*invT);
        
    if (Tgas>1.0 && Tgas<300.0) { k[25] = 1.3e-32*pow((T32), (-0.38e0));  }
        
    k[26] = 1.3e-32*pow((T32), (-1.00e0));
        
    if (Tgas>1.0 && Tgas<300.0) { k[27] = 1.3e-32*pow((T32),
        (-0.38e0))/8.e0;  }
        
    k[28] = 1.3e-32*pow((T32), (-1.00e0))/8.e0;
        
    k[29] = 2.00e-10*pow(Tgas, (0.402e0))*exp(-37.1e0*invT) -
        3.31e-17*pow(Tgas, (1.48e0));
        
    k[30] = 2.06e-10*pow(Tgas, (0.396))*exp(-33.e0*invT) + 2.03e-9*pow(Tgas,
        (-0.332));
        
    k[31] = 1.e-9*(0.417 + 0.846*log10(Tgas) - 0.137*pow((log10(Tgas)), 2));
        
    k[32] = 1.0e-9*exp(-4.57e2*invT);
        
    if (Tgas>1.0 && Tgas<2000.0) { k[33] = pow(10, (-56.4737 +
        5.88886*log10(Tgas) + 7.19692*pow((log10(Tgas)), 2) +
        2.25069*pow((log10(Tgas)), 3) - 2.16903*pow((log10(Tgas)), 4) +
        0.317887*pow((log10(Tgas)), 5)));  }
        
    k[34] = 3.17e-10*exp(-5207.*invT);
        
    k[35] = 5.25e-11*exp(-4430.*invT + 1.739e5*pow((invT), 2));
        
    k[36] = 1.5e-9*pow((T32), (-0.1e0));
        
    k[37] = 3.6e-12*pow((Tgas/300), (-0.75e0));
        
    
        // clang-format on

    return NAUNET_SUCCESS;
}

/* */
int InitJac(SUNMatrix jmatrix) {
    int rowptrs[NEQUATIONS + 1], colvals[NNZ];

    // Zero out the Jacobian
    SUNMatZero(jmatrix);

    // clang-format off
    // number of non-zero elements in each row
    rowptrs[0] = 0;
    rowptrs[1] = 8;
    rowptrs[2] = 15;
    rowptrs[3] = 20;
    rowptrs[4] = 29;
    rowptrs[5] = 38;
    rowptrs[6] = 44;
    rowptrs[7] = 53;
    rowptrs[8] = 59;
    rowptrs[9] = 66;
    rowptrs[10] = 69;
    rowptrs[11] = 73;
    rowptrs[12] = 76;
    rowptrs[13] = 87;
    rowptrs[14] = 93;
    
    // the column index of non-zero elements
    colvals[0] = 0;
    colvals[1] = 1;
    colvals[2] = 3;
    colvals[3] = 4;
    colvals[4] = 5;
    colvals[5] = 6;
    colvals[6] = 8;
    colvals[7] = 12;
    colvals[8] = 0;
    colvals[9] = 1;
    colvals[10] = 3;
    colvals[11] = 4;
    colvals[12] = 6;
    colvals[13] = 8;
    colvals[14] = 12;
    colvals[15] = 3;
    colvals[16] = 4;
    colvals[17] = 10;
    colvals[18] = 11;
    colvals[19] = 12;
    colvals[20] = 0;
    colvals[21] = 1;
    colvals[22] = 3;
    colvals[23] = 4;
    colvals[24] = 5;
    colvals[25] = 6;
    colvals[26] = 7;
    colvals[27] = 8;
    colvals[28] = 12;
    colvals[29] = 0;
    colvals[30] = 1;
    colvals[31] = 3;
    colvals[32] = 4;
    colvals[33] = 5;
    colvals[34] = 6;
    colvals[35] = 7;
    colvals[36] = 8;
    colvals[37] = 12;
    colvals[38] = 0;
    colvals[39] = 3;
    colvals[40] = 4;
    colvals[41] = 5;
    colvals[42] = 7;
    colvals[43] = 12;
    colvals[44] = 0;
    colvals[45] = 1;
    colvals[46] = 3;
    colvals[47] = 4;
    colvals[48] = 5;
    colvals[49] = 6;
    colvals[50] = 7;
    colvals[51] = 8;
    colvals[52] = 12;
    colvals[53] = 3;
    colvals[54] = 4;
    colvals[55] = 5;
    colvals[56] = 6;
    colvals[57] = 7;
    colvals[58] = 12;
    colvals[59] = 0;
    colvals[60] = 1;
    colvals[61] = 3;
    colvals[62] = 4;
    colvals[63] = 5;
    colvals[64] = 6;
    colvals[65] = 8;
    colvals[66] = 9;
    colvals[67] = 10;
    colvals[68] = 12;
    colvals[69] = 9;
    colvals[70] = 10;
    colvals[71] = 11;
    colvals[72] = 12;
    colvals[73] = 10;
    colvals[74] = 11;
    colvals[75] = 12;
    colvals[76] = 0;
    colvals[77] = 1;
    colvals[78] = 3;
    colvals[79] = 4;
    colvals[80] = 5;
    colvals[81] = 6;
    colvals[82] = 7;
    colvals[83] = 9;
    colvals[84] = 10;
    colvals[85] = 11;
    colvals[86] = 12;
    colvals[87] = 3;
    colvals[88] = 4;
    colvals[89] = 9;
    colvals[90] = 10;
    colvals[91] = 11;
    colvals[92] = 12;
    
    // clang-format on

    // copy rowptrs, colvals to the device
    SUNMatrix_cuSparse_CopyToDevice(jmatrix, NULL, rowptrs, colvals);
    cudaDeviceSynchronize();

    return NAUNET_SUCCESS;
}

__global__ void FexKernel(realtype *y, realtype *ydot, NaunetData *d_udata,
                          int nsystem) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gs   = blockDim.x * gridDim.x;

    // clang-format off
    realtype nH = d_udata->nH;
    realtype Tgas = d_udata->Tgas;
    
    realtype mu = d_udata->mu;
    realtype gamma = d_udata->gamma;
        
    if (mu < 0) mu = GetMu(y);
    if (gamma < 0) gamma = GetGamma(y);
    // clang-format on

    for (int cur = tidx; cur < nsystem; cur += gs) {
        int yistart            = cur * NEQUATIONS;
        realtype *y_cur        = y + yistart;
        realtype k[NREACTIONS] = {0.0};
        NaunetData *udata      = &d_udata[cur];

        EvalRates(k, y_cur, udata);

        // clang-format off
        ydot[yistart + IDX_DI] = 0.0 - k[29]*y_cur[IDX_HII]*y_cur[IDX_DI] +
            k[30]*y_cur[IDX_HI]*y_cur[IDX_DII] -
            k[33]*y_cur[IDX_H2I]*y_cur[IDX_DI] -
            k[34]*y_cur[IDX_H2I]*y_cur[IDX_DI] +
            k[35]*y_cur[IDX_HDI]*y_cur[IDX_HI] -
            k[36]*y_cur[IDX_DI]*y_cur[IDX_HM] +
            k[37]*y_cur[IDX_DII]*y_cur[IDX_eM];
        ydot[yistart + IDX_DII] = 0.0 + k[29]*y_cur[IDX_HII]*y_cur[IDX_DI] -
            k[30]*y_cur[IDX_HI]*y_cur[IDX_DII] -
            k[31]*y_cur[IDX_H2I]*y_cur[IDX_DII] +
            k[32]*y_cur[IDX_HDI]*y_cur[IDX_HII] -
            k[37]*y_cur[IDX_DII]*y_cur[IDX_eM];
        ydot[yistart + IDX_GRAINI] = 0.0 + k[1]*y_cur[IDX_HII]*y_cur[IDX_eM]
            + k[2]*y_cur[IDX_HII]*y_cur[IDX_eM] +
            k[4]*y_cur[IDX_HeII]*y_cur[IDX_eM] +
            k[5]*y_cur[IDX_HeII]*y_cur[IDX_eM] +
            k[7]*y_cur[IDX_HeIII]*y_cur[IDX_eM] +
            k[8]*y_cur[IDX_HI]*y_cur[IDX_eM] +
            k[11]*y_cur[IDX_HI]*y_cur[IDX_HII] +
            k[12]*y_cur[IDX_HI]*y_cur[IDX_HII];
        ydot[yistart + IDX_HI] = 0.0 - k[0]*y_cur[IDX_HI]*y_cur[IDX_eM] +
            k[1]*y_cur[IDX_HII]*y_cur[IDX_eM] +
            k[2]*y_cur[IDX_HII]*y_cur[IDX_eM] - k[8]*y_cur[IDX_HI]*y_cur[IDX_eM]
            - k[9]*y_cur[IDX_HM]*y_cur[IDX_HI] -
            k[10]*y_cur[IDX_HM]*y_cur[IDX_HI] -
            k[11]*y_cur[IDX_HI]*y_cur[IDX_HII] -
            k[12]*y_cur[IDX_HI]*y_cur[IDX_HII] -
            k[13]*y_cur[IDX_H2II]*y_cur[IDX_HI] +
            k[14]*y_cur[IDX_H2I]*y_cur[IDX_HII] +
            k[15]*y_cur[IDX_H2I]*y_cur[IDX_eM] +
            k[15]*y_cur[IDX_H2I]*y_cur[IDX_eM] -
            k[16]*y_cur[IDX_H2I]*y_cur[IDX_HI] +
            k[16]*y_cur[IDX_H2I]*y_cur[IDX_HI] +
            k[16]*y_cur[IDX_H2I]*y_cur[IDX_HI] +
            k[16]*y_cur[IDX_H2I]*y_cur[IDX_HI] +
            k[17]*y_cur[IDX_HM]*y_cur[IDX_eM] -
            k[18]*y_cur[IDX_HM]*y_cur[IDX_HI] +
            k[18]*y_cur[IDX_HM]*y_cur[IDX_HI] +
            k[18]*y_cur[IDX_HM]*y_cur[IDX_HI] -
            k[19]*y_cur[IDX_HM]*y_cur[IDX_HI] +
            k[19]*y_cur[IDX_HM]*y_cur[IDX_HI] +
            k[19]*y_cur[IDX_HM]*y_cur[IDX_HI] +
            k[20]*y_cur[IDX_HM]*y_cur[IDX_HII] +
            k[20]*y_cur[IDX_HM]*y_cur[IDX_HII] +
            k[22]*y_cur[IDX_H2II]*y_cur[IDX_eM] +
            k[22]*y_cur[IDX_H2II]*y_cur[IDX_eM] +
            k[23]*y_cur[IDX_H2II]*y_cur[IDX_eM] +
            k[23]*y_cur[IDX_H2II]*y_cur[IDX_eM] +
            k[24]*y_cur[IDX_H2II]*y_cur[IDX_HM] -
            k[25]*y_cur[IDX_HI]*y_cur[IDX_HI]*y_cur[IDX_HI] -
            k[25]*y_cur[IDX_HI]*y_cur[IDX_HI]*y_cur[IDX_HI] -
            k[25]*y_cur[IDX_HI]*y_cur[IDX_HI]*y_cur[IDX_HI] +
            k[25]*y_cur[IDX_HI]*y_cur[IDX_HI]*y_cur[IDX_HI] -
            k[26]*y_cur[IDX_HI]*y_cur[IDX_HI]*y_cur[IDX_HI] -
            k[26]*y_cur[IDX_HI]*y_cur[IDX_HI]*y_cur[IDX_HI] -
            k[26]*y_cur[IDX_HI]*y_cur[IDX_HI]*y_cur[IDX_HI] +
            k[26]*y_cur[IDX_HI]*y_cur[IDX_HI]*y_cur[IDX_HI] -
            k[27]*y_cur[IDX_H2I]*y_cur[IDX_HI]*y_cur[IDX_HI] -
            k[27]*y_cur[IDX_H2I]*y_cur[IDX_HI]*y_cur[IDX_HI] -
            k[28]*y_cur[IDX_H2I]*y_cur[IDX_HI]*y_cur[IDX_HI] -
            k[28]*y_cur[IDX_H2I]*y_cur[IDX_HI]*y_cur[IDX_HI] +
            k[29]*y_cur[IDX_HII]*y_cur[IDX_DI] -
            k[30]*y_cur[IDX_HI]*y_cur[IDX_DII] +
            k[33]*y_cur[IDX_H2I]*y_cur[IDX_DI] +
            k[34]*y_cur[IDX_H2I]*y_cur[IDX_DI] -
            k[35]*y_cur[IDX_HDI]*y_cur[IDX_HI];
        ydot[yistart + IDX_HII] = 0.0 + k[0]*y_cur[IDX_HI]*y_cur[IDX_eM] -
            k[1]*y_cur[IDX_HII]*y_cur[IDX_eM] -
            k[2]*y_cur[IDX_HII]*y_cur[IDX_eM] -
            k[11]*y_cur[IDX_HI]*y_cur[IDX_HII] -
            k[12]*y_cur[IDX_HI]*y_cur[IDX_HII] +
            k[13]*y_cur[IDX_H2II]*y_cur[IDX_HI] -
            k[14]*y_cur[IDX_H2I]*y_cur[IDX_HII] -
            k[20]*y_cur[IDX_HM]*y_cur[IDX_HII] -
            k[21]*y_cur[IDX_HM]*y_cur[IDX_HII] -
            k[29]*y_cur[IDX_HII]*y_cur[IDX_DI] +
            k[30]*y_cur[IDX_HI]*y_cur[IDX_DII] +
            k[31]*y_cur[IDX_H2I]*y_cur[IDX_DII] -
            k[32]*y_cur[IDX_HDI]*y_cur[IDX_HII];
        ydot[yistart + IDX_HM] = 0.0 + k[8]*y_cur[IDX_HI]*y_cur[IDX_eM] -
            k[9]*y_cur[IDX_HM]*y_cur[IDX_HI] - k[10]*y_cur[IDX_HM]*y_cur[IDX_HI]
            - k[17]*y_cur[IDX_HM]*y_cur[IDX_eM] -
            k[18]*y_cur[IDX_HM]*y_cur[IDX_HI] -
            k[19]*y_cur[IDX_HM]*y_cur[IDX_HI] -
            k[20]*y_cur[IDX_HM]*y_cur[IDX_HII] -
            k[21]*y_cur[IDX_HM]*y_cur[IDX_HII] -
            k[24]*y_cur[IDX_H2II]*y_cur[IDX_HM] -
            k[36]*y_cur[IDX_DI]*y_cur[IDX_HM];
        ydot[yistart + IDX_H2I] = 0.0 + k[9]*y_cur[IDX_HM]*y_cur[IDX_HI] +
            k[10]*y_cur[IDX_HM]*y_cur[IDX_HI] +
            k[13]*y_cur[IDX_H2II]*y_cur[IDX_HI] -
            k[14]*y_cur[IDX_H2I]*y_cur[IDX_HII] -
            k[15]*y_cur[IDX_H2I]*y_cur[IDX_eM] -
            k[16]*y_cur[IDX_H2I]*y_cur[IDX_HI] +
            k[24]*y_cur[IDX_H2II]*y_cur[IDX_HM] +
            k[25]*y_cur[IDX_HI]*y_cur[IDX_HI]*y_cur[IDX_HI] +
            k[26]*y_cur[IDX_HI]*y_cur[IDX_HI]*y_cur[IDX_HI] -
            k[27]*y_cur[IDX_H2I]*y_cur[IDX_HI]*y_cur[IDX_HI] +
            k[27]*y_cur[IDX_H2I]*y_cur[IDX_HI]*y_cur[IDX_HI] +
            k[27]*y_cur[IDX_H2I]*y_cur[IDX_HI]*y_cur[IDX_HI] -
            k[28]*y_cur[IDX_H2I]*y_cur[IDX_HI]*y_cur[IDX_HI] +
            k[28]*y_cur[IDX_H2I]*y_cur[IDX_HI]*y_cur[IDX_HI] +
            k[28]*y_cur[IDX_H2I]*y_cur[IDX_HI]*y_cur[IDX_HI] -
            k[31]*y_cur[IDX_H2I]*y_cur[IDX_DII] +
            k[32]*y_cur[IDX_HDI]*y_cur[IDX_HII] -
            k[33]*y_cur[IDX_H2I]*y_cur[IDX_DI] -
            k[34]*y_cur[IDX_H2I]*y_cur[IDX_DI] +
            k[35]*y_cur[IDX_HDI]*y_cur[IDX_HI];
        ydot[yistart + IDX_H2II] = 0.0 + k[11]*y_cur[IDX_HI]*y_cur[IDX_HII]
            + k[12]*y_cur[IDX_HI]*y_cur[IDX_HII] -
            k[13]*y_cur[IDX_H2II]*y_cur[IDX_HI] +
            k[14]*y_cur[IDX_H2I]*y_cur[IDX_HII] +
            k[21]*y_cur[IDX_HM]*y_cur[IDX_HII] -
            k[22]*y_cur[IDX_H2II]*y_cur[IDX_eM] -
            k[23]*y_cur[IDX_H2II]*y_cur[IDX_eM] -
            k[24]*y_cur[IDX_H2II]*y_cur[IDX_HM];
        ydot[yistart + IDX_HDI] = 0.0 + k[31]*y_cur[IDX_H2I]*y_cur[IDX_DII]
            - k[32]*y_cur[IDX_HDI]*y_cur[IDX_HII] +
            k[33]*y_cur[IDX_H2I]*y_cur[IDX_DI] +
            k[34]*y_cur[IDX_H2I]*y_cur[IDX_DI] -
            k[35]*y_cur[IDX_HDI]*y_cur[IDX_HI] +
            k[36]*y_cur[IDX_DI]*y_cur[IDX_HM];
        ydot[yistart + IDX_HeI] = 0.0 - k[3]*y_cur[IDX_HeI]*y_cur[IDX_eM] +
            k[4]*y_cur[IDX_HeII]*y_cur[IDX_eM] +
            k[5]*y_cur[IDX_HeII]*y_cur[IDX_eM];
        ydot[yistart + IDX_HeII] = 0.0 + k[3]*y_cur[IDX_HeI]*y_cur[IDX_eM] -
            k[4]*y_cur[IDX_HeII]*y_cur[IDX_eM] -
            k[5]*y_cur[IDX_HeII]*y_cur[IDX_eM] -
            k[6]*y_cur[IDX_HeII]*y_cur[IDX_eM] +
            k[7]*y_cur[IDX_HeIII]*y_cur[IDX_eM];
        ydot[yistart + IDX_HeIII] = 0.0 + k[6]*y_cur[IDX_HeII]*y_cur[IDX_eM]
            - k[7]*y_cur[IDX_HeIII]*y_cur[IDX_eM];
        ydot[yistart + IDX_eM] = 0.0 - k[0]*y_cur[IDX_HI]*y_cur[IDX_eM] +
            k[0]*y_cur[IDX_HI]*y_cur[IDX_eM] + k[0]*y_cur[IDX_HI]*y_cur[IDX_eM]
            - k[1]*y_cur[IDX_HII]*y_cur[IDX_eM] -
            k[2]*y_cur[IDX_HII]*y_cur[IDX_eM] -
            k[3]*y_cur[IDX_HeI]*y_cur[IDX_eM] +
            k[3]*y_cur[IDX_HeI]*y_cur[IDX_eM] +
            k[3]*y_cur[IDX_HeI]*y_cur[IDX_eM] -
            k[4]*y_cur[IDX_HeII]*y_cur[IDX_eM] -
            k[5]*y_cur[IDX_HeII]*y_cur[IDX_eM] -
            k[6]*y_cur[IDX_HeII]*y_cur[IDX_eM] +
            k[6]*y_cur[IDX_HeII]*y_cur[IDX_eM] +
            k[6]*y_cur[IDX_HeII]*y_cur[IDX_eM] -
            k[7]*y_cur[IDX_HeIII]*y_cur[IDX_eM] -
            k[8]*y_cur[IDX_HI]*y_cur[IDX_eM] + k[9]*y_cur[IDX_HM]*y_cur[IDX_HI]
            + k[10]*y_cur[IDX_HM]*y_cur[IDX_HI] -
            k[15]*y_cur[IDX_H2I]*y_cur[IDX_eM] +
            k[15]*y_cur[IDX_H2I]*y_cur[IDX_eM] -
            k[17]*y_cur[IDX_HM]*y_cur[IDX_eM] +
            k[17]*y_cur[IDX_HM]*y_cur[IDX_eM] +
            k[17]*y_cur[IDX_HM]*y_cur[IDX_eM] +
            k[18]*y_cur[IDX_HM]*y_cur[IDX_HI] +
            k[19]*y_cur[IDX_HM]*y_cur[IDX_HI] +
            k[21]*y_cur[IDX_HM]*y_cur[IDX_HII] -
            k[22]*y_cur[IDX_H2II]*y_cur[IDX_eM] -
            k[23]*y_cur[IDX_H2II]*y_cur[IDX_eM] +
            k[36]*y_cur[IDX_DI]*y_cur[IDX_HM] -
            k[37]*y_cur[IDX_DII]*y_cur[IDX_eM];
        ydot[yistart + IDX_TGAS] = (gamma - 1.0) * ( 0.0 - 1.27e-21 *
            sqrt(y_cur[IDX_TGAS]) / (1.0 + sqrt(y_cur[IDX_TGAS]/1e5)) *
            exp(-1.578091e5/y_cur[IDX_TGAS]) * y_cur[IDX_HI]*y_cur[IDX_eM] -
            9.38e-22 * sqrt(y_cur[IDX_TGAS]) / (1.0 + sqrt(y_cur[IDX_TGAS]/1e5))
            * exp(-2.853354e5/y_cur[IDX_TGAS]) * y_cur[IDX_HeI]*y_cur[IDX_eM] -
            4.95e-22 * sqrt(y_cur[IDX_TGAS]) / (1.0 + sqrt(y_cur[IDX_TGAS]/1e5))
            * exp(-6.31515e5/y_cur[IDX_TGAS]) * y_cur[IDX_HeII]*y_cur[IDX_eM] -
            5.01e-27 * pow(y_cur[IDX_TGAS], -0.1687) / (1.0 +
            sqrt(y_cur[IDX_TGAS]/1e5)) * exp(-5.5338e4/y_cur[IDX_TGAS]) *
            y_cur[IDX_HeII]*y_cur[IDX_eM]*y_cur[IDX_eM] - 8.7e-27 *
            sqrt(y_cur[IDX_TGAS]) * pow(y_cur[IDX_TGAS]/1e3, -0.2) /
            (1.0+pow(y_cur[IDX_TGAS]/1e6, 0.7)) * y_cur[IDX_HII]*y_cur[IDX_eM] -
            1.24e-13 * pow(y_cur[IDX_TGAS], -1.5) * exp(-4.7e5/y_cur[IDX_TGAS])
            * (1.0+0.3*exp(-9.4e4/y_cur[IDX_TGAS])) *
            y_cur[IDX_HeII]*y_cur[IDX_eM] - 1.55e-26 * pow(y_cur[IDX_TGAS],
            0.3647) * y_cur[IDX_HeII]*y_cur[IDX_eM] - 3.48e-26 *
            sqrt(y_cur[IDX_TGAS]) * pow(y_cur[IDX_TGAS]/1e3, -0.2) /
            (1.0+pow(y_cur[IDX_TGAS]/1e6, 0.7)) * y_cur[IDX_HeIII]*y_cur[IDX_eM]
            - 9.1e-27 * pow(y_cur[IDX_TGAS], -0.1687) /
            (1.0+sqrt(y_cur[IDX_TGAS]/1e5)) * exp(-1.3179e4/y_cur[IDX_TGAS]) *
            y_cur[IDX_HI]*y_cur[IDX_eM]*y_cur[IDX_eM] - 5.54e-17 *
            pow(y_cur[IDX_TGAS], -.0397) / (1.0+sqrt(y_cur[IDX_TGAS]/1e5))
            *exp(-4.73638e5/y_cur[IDX_TGAS]) * y_cur[IDX_HeII]*y_cur[IDX_eM] -
            5.54e-17 * pow(y_cur[IDX_TGAS], -.0397) /
            (1.0+sqrt(y_cur[IDX_TGAS]/1e5)) *exp(-4.73638e5/y_cur[IDX_TGAS]) *
            y_cur[IDX_HeII]*y_cur[IDX_eM] ) / kerg / GetNumDens(y);
        
                // clang-format on
    }
}

__global__ void JacKernel(realtype *y, realtype *data, NaunetData *d_udata,
                          int nsystem) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gs   = blockDim.x * gridDim.x;

    for (int cur = tidx; cur < nsystem; cur += gs) {
        int yistart            = cur * NEQUATIONS;
        int jistart            = cur * NNZ;
        realtype *y_cur        = y + yistart;
        realtype k[NREACTIONS] = {0.0};
        NaunetData *udata      = &d_udata[cur];

        // clang-format off
        realtype mu = udata->mu;
        realtype gamma = udata->gamma;
                
        if (mu < 0) mu = GetMu(y);
        if (gamma < 0) gamma = GetGamma(y);
        // clang-format on

        EvalRates(k, y_cur, udata);

        // clang-format off
        data[jistart + 0] = 0.0 - k[29]*y_cur[IDX_HII] - k[33]*y_cur[IDX_H2I] - k[34]*y_cur[IDX_H2I] - k[36]*y_cur[IDX_HM];
        data[jistart + 1] = 0.0 + k[30]*y_cur[IDX_HI] + k[37]*y_cur[IDX_eM];
        data[jistart + 2] = 0.0 + k[30]*y_cur[IDX_DII] + k[35]*y_cur[IDX_HDI];
        data[jistart + 3] = 0.0 - k[29]*y_cur[IDX_DI];
        data[jistart + 4] = 0.0 - k[36]*y_cur[IDX_DI];
        data[jistart + 5] = 0.0 - k[33]*y_cur[IDX_DI] - k[34]*y_cur[IDX_DI];
        data[jistart + 6] = 0.0 + k[35]*y_cur[IDX_HI];
        data[jistart + 7] = 0.0 + k[37]*y_cur[IDX_DII];
        data[jistart + 8] = 0.0 + k[29]*y_cur[IDX_HII];
        data[jistart + 9] = 0.0 - k[30]*y_cur[IDX_HI] - k[31]*y_cur[IDX_H2I] - k[37]*y_cur[IDX_eM];
        data[jistart + 10] = 0.0 - k[30]*y_cur[IDX_DII];
        data[jistart + 11] = 0.0 + k[29]*y_cur[IDX_DI] + k[32]*y_cur[IDX_HDI];
        data[jistart + 12] = 0.0 - k[31]*y_cur[IDX_DII];
        data[jistart + 13] = 0.0 + k[32]*y_cur[IDX_HII];
        data[jistart + 14] = 0.0 - k[37]*y_cur[IDX_DII];
        data[jistart + 15] = 0.0 + k[8]*y_cur[IDX_eM] + k[11]*y_cur[IDX_HII] + k[12]*y_cur[IDX_HII];
        data[jistart + 16] = 0.0 + k[1]*y_cur[IDX_eM] + k[2]*y_cur[IDX_eM] + k[11]*y_cur[IDX_HI] + k[12]*y_cur[IDX_HI];
        data[jistart + 17] = 0.0 + k[4]*y_cur[IDX_eM] + k[5]*y_cur[IDX_eM];
        data[jistart + 18] = 0.0 + k[7]*y_cur[IDX_eM];
        data[jistart + 19] = 0.0 + k[1]*y_cur[IDX_HII] + k[2]*y_cur[IDX_HII] + k[4]*y_cur[IDX_HeII] + k[5]*y_cur[IDX_HeII] + k[7]*y_cur[IDX_HeIII] + k[8]*y_cur[IDX_HI];
        data[jistart + 20] = 0.0 + k[29]*y_cur[IDX_HII] + k[33]*y_cur[IDX_H2I] + k[34]*y_cur[IDX_H2I];
        data[jistart + 21] = 0.0 - k[30]*y_cur[IDX_HI];
        data[jistart + 22] = 0.0 - k[0]*y_cur[IDX_eM] - k[8]*y_cur[IDX_eM] - k[9]*y_cur[IDX_HM] - k[10]*y_cur[IDX_HM] - k[11]*y_cur[IDX_HII] - k[12]*y_cur[IDX_HII] - k[13]*y_cur[IDX_H2II] - k[16]*y_cur[IDX_H2I] + k[16]*y_cur[IDX_H2I] + k[16]*y_cur[IDX_H2I] + k[16]*y_cur[IDX_H2I] - k[18]*y_cur[IDX_HM] + k[18]*y_cur[IDX_HM] + k[18]*y_cur[IDX_HM] - k[19]*y_cur[IDX_HM] + k[19]*y_cur[IDX_HM] + k[19]*y_cur[IDX_HM] - k[25]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[25]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[25]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[25]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[25]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[25]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[25]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[25]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[25]*y_cur[IDX_HI]*y_cur[IDX_HI] + k[25]*y_cur[IDX_HI]*y_cur[IDX_HI] + k[25]*y_cur[IDX_HI]*y_cur[IDX_HI] + k[25]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[26]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[26]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[26]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[26]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[26]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[26]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[26]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[26]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[26]*y_cur[IDX_HI]*y_cur[IDX_HI] + k[26]*y_cur[IDX_HI]*y_cur[IDX_HI] + k[26]*y_cur[IDX_HI]*y_cur[IDX_HI] + k[26]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[27]*y_cur[IDX_H2I]*y_cur[IDX_HI] - k[27]*y_cur[IDX_H2I]*y_cur[IDX_HI] - k[27]*y_cur[IDX_H2I]*y_cur[IDX_HI] - k[27]*y_cur[IDX_H2I]*y_cur[IDX_HI] - k[28]*y_cur[IDX_H2I]*y_cur[IDX_HI] - k[28]*y_cur[IDX_H2I]*y_cur[IDX_HI] - k[28]*y_cur[IDX_H2I]*y_cur[IDX_HI] - k[28]*y_cur[IDX_H2I]*y_cur[IDX_HI] - k[30]*y_cur[IDX_DII] - k[35]*y_cur[IDX_HDI];
        data[jistart + 23] = 0.0 + k[1]*y_cur[IDX_eM] + k[2]*y_cur[IDX_eM] - k[11]*y_cur[IDX_HI] - k[12]*y_cur[IDX_HI] + k[14]*y_cur[IDX_H2I] + k[20]*y_cur[IDX_HM] + k[20]*y_cur[IDX_HM] + k[29]*y_cur[IDX_DI];
        data[jistart + 24] = 0.0 - k[9]*y_cur[IDX_HI] - k[10]*y_cur[IDX_HI] + k[17]*y_cur[IDX_eM] - k[18]*y_cur[IDX_HI] + k[18]*y_cur[IDX_HI] + k[18]*y_cur[IDX_HI] - k[19]*y_cur[IDX_HI] + k[19]*y_cur[IDX_HI] + k[19]*y_cur[IDX_HI] + k[20]*y_cur[IDX_HII] + k[20]*y_cur[IDX_HII] + k[24]*y_cur[IDX_H2II];
        data[jistart + 25] = 0.0 + k[14]*y_cur[IDX_HII] + k[15]*y_cur[IDX_eM] + k[15]*y_cur[IDX_eM] - k[16]*y_cur[IDX_HI] + k[16]*y_cur[IDX_HI] + k[16]*y_cur[IDX_HI] + k[16]*y_cur[IDX_HI] - k[27]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[27]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[28]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[28]*y_cur[IDX_HI]*y_cur[IDX_HI] + k[33]*y_cur[IDX_DI] + k[34]*y_cur[IDX_DI];
        data[jistart + 26] = 0.0 - k[13]*y_cur[IDX_HI] + k[22]*y_cur[IDX_eM] + k[22]*y_cur[IDX_eM] + k[23]*y_cur[IDX_eM] + k[23]*y_cur[IDX_eM] + k[24]*y_cur[IDX_HM];
        data[jistart + 27] = 0.0 - k[35]*y_cur[IDX_HI];
        data[jistart + 28] = 0.0 - k[0]*y_cur[IDX_HI] + k[1]*y_cur[IDX_HII] + k[2]*y_cur[IDX_HII] - k[8]*y_cur[IDX_HI] + k[15]*y_cur[IDX_H2I] + k[15]*y_cur[IDX_H2I] + k[17]*y_cur[IDX_HM] + k[22]*y_cur[IDX_H2II] + k[22]*y_cur[IDX_H2II] + k[23]*y_cur[IDX_H2II] + k[23]*y_cur[IDX_H2II];
        data[jistart + 29] = 0.0 - k[29]*y_cur[IDX_HII];
        data[jistart + 30] = 0.0 + k[30]*y_cur[IDX_HI] + k[31]*y_cur[IDX_H2I];
        data[jistart + 31] = 0.0 + k[0]*y_cur[IDX_eM] - k[11]*y_cur[IDX_HII] - k[12]*y_cur[IDX_HII] + k[13]*y_cur[IDX_H2II] + k[30]*y_cur[IDX_DII];
        data[jistart + 32] = 0.0 - k[1]*y_cur[IDX_eM] - k[2]*y_cur[IDX_eM] - k[11]*y_cur[IDX_HI] - k[12]*y_cur[IDX_HI] - k[14]*y_cur[IDX_H2I] - k[20]*y_cur[IDX_HM] - k[21]*y_cur[IDX_HM] - k[29]*y_cur[IDX_DI] - k[32]*y_cur[IDX_HDI];
        data[jistart + 33] = 0.0 - k[20]*y_cur[IDX_HII] - k[21]*y_cur[IDX_HII];
        data[jistart + 34] = 0.0 - k[14]*y_cur[IDX_HII] + k[31]*y_cur[IDX_DII];
        data[jistart + 35] = 0.0 + k[13]*y_cur[IDX_HI];
        data[jistart + 36] = 0.0 - k[32]*y_cur[IDX_HII];
        data[jistart + 37] = 0.0 + k[0]*y_cur[IDX_HI] - k[1]*y_cur[IDX_HII] - k[2]*y_cur[IDX_HII];
        data[jistart + 38] = 0.0 - k[36]*y_cur[IDX_HM];
        data[jistart + 39] = 0.0 + k[8]*y_cur[IDX_eM] - k[9]*y_cur[IDX_HM] - k[10]*y_cur[IDX_HM] - k[18]*y_cur[IDX_HM] - k[19]*y_cur[IDX_HM];
        data[jistart + 40] = 0.0 - k[20]*y_cur[IDX_HM] - k[21]*y_cur[IDX_HM];
        data[jistart + 41] = 0.0 - k[9]*y_cur[IDX_HI] - k[10]*y_cur[IDX_HI] - k[17]*y_cur[IDX_eM] - k[18]*y_cur[IDX_HI] - k[19]*y_cur[IDX_HI] - k[20]*y_cur[IDX_HII] - k[21]*y_cur[IDX_HII] - k[24]*y_cur[IDX_H2II] - k[36]*y_cur[IDX_DI];
        data[jistart + 42] = 0.0 - k[24]*y_cur[IDX_HM];
        data[jistart + 43] = 0.0 + k[8]*y_cur[IDX_HI] - k[17]*y_cur[IDX_HM];
        data[jistart + 44] = 0.0 - k[33]*y_cur[IDX_H2I] - k[34]*y_cur[IDX_H2I];
        data[jistart + 45] = 0.0 - k[31]*y_cur[IDX_H2I];
        data[jistart + 46] = 0.0 + k[9]*y_cur[IDX_HM] + k[10]*y_cur[IDX_HM] + k[13]*y_cur[IDX_H2II] - k[16]*y_cur[IDX_H2I] + k[25]*y_cur[IDX_HI]*y_cur[IDX_HI] + k[25]*y_cur[IDX_HI]*y_cur[IDX_HI] + k[25]*y_cur[IDX_HI]*y_cur[IDX_HI] + k[26]*y_cur[IDX_HI]*y_cur[IDX_HI] + k[26]*y_cur[IDX_HI]*y_cur[IDX_HI] + k[26]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[27]*y_cur[IDX_H2I]*y_cur[IDX_HI] - k[27]*y_cur[IDX_H2I]*y_cur[IDX_HI] + k[27]*y_cur[IDX_H2I]*y_cur[IDX_HI] + k[27]*y_cur[IDX_H2I]*y_cur[IDX_HI] + k[27]*y_cur[IDX_H2I]*y_cur[IDX_HI] + k[27]*y_cur[IDX_H2I]*y_cur[IDX_HI] - k[28]*y_cur[IDX_H2I]*y_cur[IDX_HI] - k[28]*y_cur[IDX_H2I]*y_cur[IDX_HI] + k[28]*y_cur[IDX_H2I]*y_cur[IDX_HI] + k[28]*y_cur[IDX_H2I]*y_cur[IDX_HI] + k[28]*y_cur[IDX_H2I]*y_cur[IDX_HI] + k[28]*y_cur[IDX_H2I]*y_cur[IDX_HI] + k[35]*y_cur[IDX_HDI];
        data[jistart + 47] = 0.0 - k[14]*y_cur[IDX_H2I] + k[32]*y_cur[IDX_HDI];
        data[jistart + 48] = 0.0 + k[9]*y_cur[IDX_HI] + k[10]*y_cur[IDX_HI] + k[24]*y_cur[IDX_H2II];
        data[jistart + 49] = 0.0 - k[14]*y_cur[IDX_HII] - k[15]*y_cur[IDX_eM] - k[16]*y_cur[IDX_HI] - k[27]*y_cur[IDX_HI]*y_cur[IDX_HI] + k[27]*y_cur[IDX_HI]*y_cur[IDX_HI] + k[27]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[28]*y_cur[IDX_HI]*y_cur[IDX_HI] + k[28]*y_cur[IDX_HI]*y_cur[IDX_HI] + k[28]*y_cur[IDX_HI]*y_cur[IDX_HI] - k[31]*y_cur[IDX_DII] - k[33]*y_cur[IDX_DI] - k[34]*y_cur[IDX_DI];
        data[jistart + 50] = 0.0 + k[13]*y_cur[IDX_HI] + k[24]*y_cur[IDX_HM];
        data[jistart + 51] = 0.0 + k[32]*y_cur[IDX_HII] + k[35]*y_cur[IDX_HI];
        data[jistart + 52] = 0.0 - k[15]*y_cur[IDX_H2I];
        data[jistart + 53] = 0.0 + k[11]*y_cur[IDX_HII] + k[12]*y_cur[IDX_HII] - k[13]*y_cur[IDX_H2II];
        data[jistart + 54] = 0.0 + k[11]*y_cur[IDX_HI] + k[12]*y_cur[IDX_HI] + k[14]*y_cur[IDX_H2I] + k[21]*y_cur[IDX_HM];
        data[jistart + 55] = 0.0 + k[21]*y_cur[IDX_HII] - k[24]*y_cur[IDX_H2II];
        data[jistart + 56] = 0.0 + k[14]*y_cur[IDX_HII];
        data[jistart + 57] = 0.0 - k[13]*y_cur[IDX_HI] - k[22]*y_cur[IDX_eM] - k[23]*y_cur[IDX_eM] - k[24]*y_cur[IDX_HM];
        data[jistart + 58] = 0.0 - k[22]*y_cur[IDX_H2II] - k[23]*y_cur[IDX_H2II];
        data[jistart + 59] = 0.0 + k[33]*y_cur[IDX_H2I] + k[34]*y_cur[IDX_H2I] + k[36]*y_cur[IDX_HM];
        data[jistart + 60] = 0.0 + k[31]*y_cur[IDX_H2I];
        data[jistart + 61] = 0.0 - k[35]*y_cur[IDX_HDI];
        data[jistart + 62] = 0.0 - k[32]*y_cur[IDX_HDI];
        data[jistart + 63] = 0.0 + k[36]*y_cur[IDX_DI];
        data[jistart + 64] = 0.0 + k[31]*y_cur[IDX_DII] + k[33]*y_cur[IDX_DI] + k[34]*y_cur[IDX_DI];
        data[jistart + 65] = 0.0 - k[32]*y_cur[IDX_HII] - k[35]*y_cur[IDX_HI];
        data[jistart + 66] = 0.0 - k[3]*y_cur[IDX_eM];
        data[jistart + 67] = 0.0 + k[4]*y_cur[IDX_eM] + k[5]*y_cur[IDX_eM];
        data[jistart + 68] = 0.0 - k[3]*y_cur[IDX_HeI] + k[4]*y_cur[IDX_HeII] + k[5]*y_cur[IDX_HeII];
        data[jistart + 69] = 0.0 + k[3]*y_cur[IDX_eM];
        data[jistart + 70] = 0.0 - k[4]*y_cur[IDX_eM] - k[5]*y_cur[IDX_eM] - k[6]*y_cur[IDX_eM];
        data[jistart + 71] = 0.0 + k[7]*y_cur[IDX_eM];
        data[jistart + 72] = 0.0 + k[3]*y_cur[IDX_HeI] - k[4]*y_cur[IDX_HeII] - k[5]*y_cur[IDX_HeII] - k[6]*y_cur[IDX_HeII] + k[7]*y_cur[IDX_HeIII];
        data[jistart + 73] = 0.0 + k[6]*y_cur[IDX_eM];
        data[jistart + 74] = 0.0 - k[7]*y_cur[IDX_eM];
        data[jistart + 75] = 0.0 + k[6]*y_cur[IDX_HeII] - k[7]*y_cur[IDX_HeIII];
        data[jistart + 76] = 0.0 + k[36]*y_cur[IDX_HM];
        data[jistart + 77] = 0.0 - k[37]*y_cur[IDX_eM];
        data[jistart + 78] = 0.0 - k[0]*y_cur[IDX_eM] + k[0]*y_cur[IDX_eM] + k[0]*y_cur[IDX_eM] - k[8]*y_cur[IDX_eM] + k[9]*y_cur[IDX_HM] + k[10]*y_cur[IDX_HM] + k[18]*y_cur[IDX_HM] + k[19]*y_cur[IDX_HM];
        data[jistart + 79] = 0.0 - k[1]*y_cur[IDX_eM] - k[2]*y_cur[IDX_eM] + k[21]*y_cur[IDX_HM];
        data[jistart + 80] = 0.0 + k[9]*y_cur[IDX_HI] + k[10]*y_cur[IDX_HI] - k[17]*y_cur[IDX_eM] + k[17]*y_cur[IDX_eM] + k[17]*y_cur[IDX_eM] + k[18]*y_cur[IDX_HI] + k[19]*y_cur[IDX_HI] + k[21]*y_cur[IDX_HII] + k[36]*y_cur[IDX_DI];
        data[jistart + 81] = 0.0 - k[15]*y_cur[IDX_eM] + k[15]*y_cur[IDX_eM];
        data[jistart + 82] = 0.0 - k[22]*y_cur[IDX_eM] - k[23]*y_cur[IDX_eM];
        data[jistart + 83] = 0.0 - k[3]*y_cur[IDX_eM] + k[3]*y_cur[IDX_eM] + k[3]*y_cur[IDX_eM];
        data[jistart + 84] = 0.0 - k[4]*y_cur[IDX_eM] - k[5]*y_cur[IDX_eM] - k[6]*y_cur[IDX_eM] + k[6]*y_cur[IDX_eM] + k[6]*y_cur[IDX_eM];
        data[jistart + 85] = 0.0 - k[7]*y_cur[IDX_eM];
        data[jistart + 86] = 0.0 - k[0]*y_cur[IDX_HI] + k[0]*y_cur[IDX_HI] + k[0]*y_cur[IDX_HI] - k[1]*y_cur[IDX_HII] - k[2]*y_cur[IDX_HII] - k[3]*y_cur[IDX_HeI] + k[3]*y_cur[IDX_HeI] + k[3]*y_cur[IDX_HeI] - k[4]*y_cur[IDX_HeII] - k[5]*y_cur[IDX_HeII] - k[6]*y_cur[IDX_HeII] + k[6]*y_cur[IDX_HeII] + k[6]*y_cur[IDX_HeII] - k[7]*y_cur[IDX_HeIII] - k[8]*y_cur[IDX_HI] - k[15]*y_cur[IDX_H2I] + k[15]*y_cur[IDX_H2I] - k[17]*y_cur[IDX_HM] + k[17]*y_cur[IDX_HM] + k[17]*y_cur[IDX_HM] - k[22]*y_cur[IDX_H2II] - k[23]*y_cur[IDX_H2II] - k[37]*y_cur[IDX_DII];
        data[jistart + 87] = 0.0 - 1.27e-21 * sqrt(y_cur[IDX_TGAS]) / (1.0 + sqrt(y_cur[IDX_TGAS]/1e5)) * exp(-1.578091e5/y_cur[IDX_TGAS])*y_cur[IDX_eM] - 9.1e-27 * pow(y_cur[IDX_TGAS], -0.1687) / (1.0+sqrt(y_cur[IDX_TGAS]/1e5)) * exp(-1.3179e4/y_cur[IDX_TGAS])*y_cur[IDX_eM]*y_cur[IDX_eM];
        data[jistart + 88] = 0.0 - 8.7e-27 * sqrt(y_cur[IDX_TGAS]) * pow(y_cur[IDX_TGAS]/1e3, -0.2) / (1.0+pow(y_cur[IDX_TGAS]/1e6, 0.7))*y_cur[IDX_eM];
        data[jistart + 89] = 0.0 - 9.38e-22 * sqrt(y_cur[IDX_TGAS]) / (1.0 + sqrt(y_cur[IDX_TGAS]/1e5)) * exp(-2.853354e5/y_cur[IDX_TGAS])*y_cur[IDX_eM];
        data[jistart + 90] = 0.0 - 4.95e-22 * sqrt(y_cur[IDX_TGAS]) / (1.0 + sqrt(y_cur[IDX_TGAS]/1e5)) * exp(-6.31515e5/y_cur[IDX_TGAS])*y_cur[IDX_eM] - 5.01e-27 * pow(y_cur[IDX_TGAS], -0.1687) / (1.0 + sqrt(y_cur[IDX_TGAS]/1e5)) * exp(-5.5338e4/y_cur[IDX_TGAS])*y_cur[IDX_eM]*y_cur[IDX_eM] - 1.24e-13 * pow(y_cur[IDX_TGAS], -1.5) * exp(-4.7e5/y_cur[IDX_TGAS]) * (1.0+0.3*exp(-9.4e4/y_cur[IDX_TGAS]))*y_cur[IDX_eM] - 1.55e-26 * pow(y_cur[IDX_TGAS], 0.3647)*y_cur[IDX_eM] - 5.54e-17 * pow(y_cur[IDX_TGAS], -.0397) / (1.0+sqrt(y_cur[IDX_TGAS]/1e5)) *exp(-4.73638e5/y_cur[IDX_TGAS])*y_cur[IDX_eM] - 5.54e-17 * pow(y_cur[IDX_TGAS], -.0397) / (1.0+sqrt(y_cur[IDX_TGAS]/1e5)) *exp(-4.73638e5/y_cur[IDX_TGAS])*y_cur[IDX_eM];
        data[jistart + 91] = 0.0 - 3.48e-26 * sqrt(y_cur[IDX_TGAS]) * pow(y_cur[IDX_TGAS]/1e3, -0.2) / (1.0+pow(y_cur[IDX_TGAS]/1e6, 0.7))*y_cur[IDX_eM];
        data[jistart + 92] = (gamma - 1.0) * (0.0 - 1.27e-21 * sqrt(y_cur[IDX_TGAS]) / (1.0 + sqrt(y_cur[IDX_TGAS]/1e5)) * exp(-1.578091e5/y_cur[IDX_TGAS])*y_cur[IDX_HI] - 9.38e-22 * sqrt(y_cur[IDX_TGAS]) / (1.0 + sqrt(y_cur[IDX_TGAS]/1e5)) * exp(-2.853354e5/y_cur[IDX_TGAS])*y_cur[IDX_HeI] - 4.95e-22 * sqrt(y_cur[IDX_TGAS]) / (1.0 + sqrt(y_cur[IDX_TGAS]/1e5)) * exp(-6.31515e5/y_cur[IDX_TGAS])*y_cur[IDX_HeII] - 5.01e-27 * pow(y_cur[IDX_TGAS], -0.1687) / (1.0 + sqrt(y_cur[IDX_TGAS]/1e5)) * exp(-5.5338e4/y_cur[IDX_TGAS])*y_cur[IDX_HeII]*y_cur[IDX_eM] - 5.01e-27 * pow(y_cur[IDX_TGAS], -0.1687) / (1.0 + sqrt(y_cur[IDX_TGAS]/1e5)) * exp(-5.5338e4/y_cur[IDX_TGAS])*y_cur[IDX_HeII]*y_cur[IDX_eM] - 8.7e-27 * sqrt(y_cur[IDX_TGAS]) * pow(y_cur[IDX_TGAS]/1e3, -0.2) / (1.0+pow(y_cur[IDX_TGAS]/1e6, 0.7))*y_cur[IDX_HII] - 1.24e-13 * pow(y_cur[IDX_TGAS], -1.5) * exp(-4.7e5/y_cur[IDX_TGAS]) * (1.0+0.3*exp(-9.4e4/y_cur[IDX_TGAS]))*y_cur[IDX_HeII] - 1.55e-26 * pow(y_cur[IDX_TGAS], 0.3647)*y_cur[IDX_HeII] - 3.48e-26 * sqrt(y_cur[IDX_TGAS]) * pow(y_cur[IDX_TGAS]/1e3, -0.2) / (1.0+pow(y_cur[IDX_TGAS]/1e6, 0.7))*y_cur[IDX_HeIII] - 9.1e-27 * pow(y_cur[IDX_TGAS], -0.1687) / (1.0+sqrt(y_cur[IDX_TGAS]/1e5)) * exp(-1.3179e4/y_cur[IDX_TGAS])*y_cur[IDX_HI]*y_cur[IDX_eM] - 9.1e-27 * pow(y_cur[IDX_TGAS], -0.1687) / (1.0+sqrt(y_cur[IDX_TGAS]/1e5)) * exp(-1.3179e4/y_cur[IDX_TGAS])*y_cur[IDX_HI]*y_cur[IDX_eM] - 5.54e-17 * pow(y_cur[IDX_TGAS], -.0397) / (1.0+sqrt(y_cur[IDX_TGAS]/1e5)) *exp(-4.73638e5/y_cur[IDX_TGAS])*y_cur[IDX_HeII] - 5.54e-17 * pow(y_cur[IDX_TGAS], -.0397) / (1.0+sqrt(y_cur[IDX_TGAS]/1e5)) *exp(-4.73638e5/y_cur[IDX_TGAS])*y_cur[IDX_HeII] ) / kerg / GetNumDens(y);
                // clang-format on
    }
}

/* */

int Fex(realtype t, N_Vector u, N_Vector udot, void *user_data) {
    /* */

    realtype *y         = N_VGetDeviceArrayPointer_Cuda(u);
    realtype *ydot      = N_VGetDeviceArrayPointer_Cuda(udot);
    NaunetData *h_udata = (NaunetData *)user_data;
    NaunetData *d_udata;

    // check the size of system (number of cells/ a batch)
    sunindextype lrw, liw;
    N_VSpace_Cuda(u, &lrw, &liw);
    int nsystem = lrw / NEQUATIONS;

    // copy the user data for each system/cell
    cudaMalloc((void **)&d_udata, sizeof(NaunetData) * nsystem);
    cudaMemcpy(d_udata, h_udata, sizeof(NaunetData) * nsystem,
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    unsigned block_size = min(BLOCKSIZE, nsystem);
    unsigned grid_size =
        max(1, min(MAXNGROUPS / BLOCKSIZE, nsystem / BLOCKSIZE));
    FexKernel<<<grid_size, block_size>>>(y, ydot, d_udata, nsystem);

    cudaDeviceSynchronize();
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) {
        fprintf(stderr, ">>> ERROR in fex: cudaGetLastError returned %s\n",
                cudaGetErrorName(cuerr));
        return -1;
    }
    cudaFree(d_udata);

    /* */

    return NAUNET_SUCCESS;
}

int Jac(realtype t, N_Vector u, N_Vector fu, SUNMatrix jmatrix, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    /* */
    realtype *y         = N_VGetDeviceArrayPointer_Cuda(u);
    realtype *data      = SUNMatrix_cuSparse_Data(jmatrix);
    NaunetData *h_udata = (NaunetData *)user_data;
    NaunetData *d_udata;

    int nsystem = SUNMatrix_cuSparse_NumBlocks(jmatrix);

    cudaMalloc((void **)&d_udata, sizeof(NaunetData) * nsystem);
    cudaMemcpy(d_udata, h_udata, sizeof(NaunetData) * nsystem,
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    unsigned block_size = min(BLOCKSIZE, nsystem);
    unsigned grid_size =
        max(1, min(MAXNGROUPS / BLOCKSIZE, nsystem / BLOCKSIZE));
    JacKernel<<<grid_size, block_size>>>(y, data, d_udata, nsystem);

    cudaDeviceSynchronize();
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) {
        fprintf(stderr, ">>> ERROR in jac: cudaGetLastError returned %s\n",
                cudaGetErrorName(cuerr));
        return -1;
    }
    cudaFree(d_udata);

    /* */

    return NAUNET_SUCCESS;
}