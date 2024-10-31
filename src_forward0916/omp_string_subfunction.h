#ifndef _OMP_STRING_SUBFUNCTION_H_
#define _OMP_STRING_SUBFUNCTION_H_
#include "par.h"
#include "su.h"
#include "mkl.h"

    void omp_memset_MKL_INT(int mkl_parallel,MKL_INT *p1,MKL_INT n);

    void omp_memset_float(int mkl_parallel,float *p1,MKL_INT n);

    void omp_memset_complex(int mkl_parallel,complex *p1,MKL_INT n);

    void omp_memset_dcomplex(int mkl_parallel,dcomplex *p1,MKL_INT n);

    void omp_memcpy_MKL_INT(int mkl_parallel,MKL_INT *p1,MKL_INT *p2,MKL_INT n);

    void omp_memcpy_dcomplex(int mkl_parallel,dcomplex *p1,dcomplex *p2,MKL_INT n);

    void omp_mklgemv(int mkl_parallel,dcomplex *h_CsrVal,MKL_INT *h_CsrRowPtr,MKL_INT *h_CsrColInd,
    dcomplex *X,dcomplex *B,MKL_INT n);


#endif


