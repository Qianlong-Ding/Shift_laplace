#ifndef _FRE_FORWARD_CPU_SUBFUNCTION_H_
#define _FRE_FORWARD_CPU_SUBFUNCTION_H_
#include <mpi.h>
#include "stdio.h"
#include "stdlib.h"
#include "par.h"
#include "su.h"

#include "mkl.h"

    int ComputeSR1d(int *OffsetSRx,int *OffsetSRy,int *OffsetSRz,float *SR,int srmaxnum);

    void IDFTt(int nt,complex *S1,float *x,int it);

    float ricker (float t, float fpeak);

    void IFFT_ft(int nt,complex *S,float *sif);

    void FFT_tf(int nt,int nfreq,float *s, complex *S);

    float Norm_2(float **A,int nx,int nz);

    void iter_solve_fun_gmres(int solve_option,int shift_laplace_option,int nter_max,int *nter_real,float *norm2_res,float tolerant,
    int nter_precond,int precond,int gmres_smoother,int fgmres_out,int gmres_out,int m_fg,int m_g,
    int myid,int npros,MPI_Comm comm,int mkl_parallel,int Level,
    dcomplex *h_ValA,MKL_INT *h_ColA,MKL_INT *h_RowA,
    float ***nppw,
    dcomplex **h_Val,MKL_INT **h_Col,MKL_INT **h_Row,dcomplex *h_b,dcomplex *h_x,
    dcomplex **h_ValR,MKL_INT **h_ColR,MKL_INT **h_RowR,
    dcomplex **h_ValP,MKL_INT **h_ColP,MKL_INT **h_RowP,
    dcomplex **h_ValSR,MKL_INT **h_ColSR,MKL_INT **h_RowSR,
    int *OffsetSRx,int *OffsetSRy,int *OffsetSRz,float *SR,int srnum,
    MKL_INT *RankIlevel,int *NxIlevel,int *NyIlevel,int *NzIlevel,
    float dx,float dy,float dz,int nxA,
    int nx,int ny,int nz,MKL_INT nzA,MKL_INT *no_zero_number,
    float R,float fpeak,float alpha_max,
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
    complex ***freq_slice,
    int ns,float *xs,float *ys,float *zs,int *ixs,int *iys,int *izs,
    complex *source_fre,int ifreq,float df,
    char *frslicepath,FILE* logfp);

    void Output_shotgather3d(int mkl_parallel,int nx,int ny,int nz,int nt,int nfreq,float dt,int nfft,float ***shot,
    int is,float *xs,float *ys,float *zs,int *ixs,int *iys,int *izs,
    float fx,float fy,float dx,float dy,float dz,float hrz,
    float tdelay,
    char *frslicepath,char *shotgatherpath);

    void Output_snapshot3d(int mkl_parallel,int nx,int ny,int nz,int nt,int nfreq,int it,float dt,int nfft,int is,
    char *frslicepath,char *slicepath);


#endif

