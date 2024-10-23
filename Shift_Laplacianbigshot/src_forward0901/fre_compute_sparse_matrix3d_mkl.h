#ifndef _FRE_COMPUTE_SPARSE_MATRIX3D_MKL_H_
#define _FRE_COMPUTE_SPARSE_MATRIX3D_MKL_H_
#include <mpi.h>
#include "stdio.h"
#include "stdlib.h"
#include "par.h"
#include "su.h"
#include "mkl.h"


	void SetPML3D(int nx,int ny,int nz,float dx,float dy,float dz,
        float ***vel,
        float ***dampx,float ***dampy,float ***dampz,
        float ***dampxdiff,float ***dampydiff,float ***dampzdiff,
        float R,float fpeak,float omega,
        float ***alphax,float ***alphay,float ***alphaz,
        float ***alphax_diff,float ***alphay_diff,float ***alphaz_diff,
        float alpha_max,
        complex ***deno_ex2,complex ***deno_ey2,complex ***deno_ez2,
        complex ***deno_ex3,complex ***deno_ey3,complex ***deno_ez3,
        int pml_thick);

	void stencil_six_order3D(int *nine_Ax,int *nine_Ay,int *nine_Az,int ix,int iy,int iz,int nx,int ny,int nz);

	void stencil_coefficient_six_order3D(complex *coefficient,
    float ***dampx,float ***dampy,float ***dampz,
    float ***dampxdiff,float ***dampydiff,float ***dampzdiff,
    int ix,int iy,int iz,
    float ***alphax,float ***alphay,float ***alphaz,
    float ***alphax_diff,float ***alphay_diff,float ***alphaz_diff,
    complex ***deno_ex2,complex ***deno_ey2,complex ***deno_ez2,
    complex ***deno_ex3,complex ***deno_ey3,complex ***deno_ez3,
    complex ***A,float omega,
    float *dux,float *duy,float *duz,
    float *duxx,float *duyy,float *duzz,
    float a,float b,float c,float d,float e,float *omegav_cofficient);

    void stencil_coefficient_six_order3D_shiftlaplace(complex *coefficient,
    float ***dampx,float ***dampy,float ***dampz,
    float ***dampxdiff,float ***dampydiff,float ***dampzdiff,
    int ix,int iy,int iz,float dx,
    float ***alphax,float ***alphay,float ***alphaz,
    float ***alphax_diff,float ***alphay_diff,float ***alphaz_diff,
    complex ***deno_ex2,complex ***deno_ey2,complex ***deno_ez2,
    complex ***deno_ex3,complex ***deno_ey3,complex ***deno_ez3,
    complex ***A,float omega,float ***nppw,
    float *dux,float *duy,float *duz,
    float *duxx,float *duyy,float *duzz,
    float a,float b,float c,float d,float e,float *omegav_cofficient);

    void compute_du_six_order3D(int nx,int ny,int nz,
                            float dx,float dy,float dz,
                            int ix,int iy,int iz,
                            float omega,float ***vel,
                            float *dux,float *duy,float *duz,
                            float *duxx,float *duyy,float *duzz,
                            float a,float b,float c,float d,float e,
                            float *omegav_cofficient);

    void Compute_matrix3D_sparse(int shift_laplace_option,int nx,int ny,int nz,float dx,float dy,float dz,int nxA,MKL_INT nzA,
    int pml_thick,
    float ***dampx,float ***dampy,float ***dampz,
    float ***dampxdiff,float ***dampydiff,float ***dampzdiff,
    float ***alphax,float ***alphay,float ***alphaz,
    float ***alphax_diff,float ***alphay_diff,float ***alphaz_diff,
    complex ***deno_ex2,complex ***deno_ey2,complex ***deno_ez2,
    complex ***deno_ex3,complex ***deno_ey3,complex ***deno_ez3,
    float *dux,float *duy,float *duz,
    float *duxx,float *duyy,float *duzz,
    complex ***A,float ***vel,float omega,float *omegav_cofficient,complex *coefficient,
    dcomplex *h_CsrValA,MKL_INT *h_CsrRowPtrA,MKL_INT *h_CsrColIndA,
    float ***nppw,
    dcomplex *h_ValA,MKL_INT *h_RowA,MKL_INT *h_ColA,
    MKL_INT *no_zero_number);

    void StencilR3d(float *R);

    void StencilOffsetR3d(int *OffsetRx,int *OffsetRy,int *OffsetRz);

    void StencilindexR3d(int *Rx,int *Ry, int *Rz,int number,
    int ix,int iy,int iz,
    int *OffsetRx,int *OffsetRy,int *OffsetRz);

    void ComputeMatrixR3d(int Level,int *NxIlevel,int *NyIlevel,int *NzIlevel,MKL_INT *RankIlevel,
    dcomplex **h_CsrValR,MKL_INT **h_CsrRowR,MKL_INT **h_CsrColR);

    void ComputeMatrixSR3d(
    int Level,int *NxIlevel,int *NyIlevel,int *NzIlevel,MKL_INT *RankIlevel,
    dcomplex **h_CsrValR,MKL_INT **h_CsrRowR,MKL_INT **h_CsrColR,
    int *OffsetSRx,int *OffsetSRy,int *OffsetSRz,float *SR,int srnum);

 
 
#endif

