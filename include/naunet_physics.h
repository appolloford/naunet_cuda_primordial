#ifndef __NAUNET_PHYSICS_H__
#define __NAUNET_PHYSICS_H__

/*  */
__device__ __host__ double GetMantleDens(double *y);
__device__ __host__ double GetMu(double *y);
__device__ __host__ double GetGamma(double *y);
__device__ __host__ double GetNumDens(double *y);
__device__ double GetShieldingFactor(int specidx, double h2coldens,
                                     double spcoldens, double tgas, int method);
__device__ double GetH2shielding(double coldens, int method);
__device__ double GetCOshielding(double tgas, double h2col, double coldens,
                                 int method);
__device__ double GetN2shielding(double tgas, double h2col, double coldens,
                                 int method);
__device__ double GetH2shieldingInt(double coldens);
__device__ double GetCOshieldingInt(double tgas, double h2col, double coldens);
__device__ double GetN2shieldingInt(double tgas, double h2col, double coldens);
/*  */

#endif