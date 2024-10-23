#ifndef _FRE_SOLVER_MKL_H_
#define _FRE_SOLVER_MKL_H_
#include <mpi.h>
#include "stdio.h"
#include "stdlib.h"
#include "par.h"
#include "su.h"

#include "mkl.h"


	void Gauss_Seidel(int nRow,int nCol,
    dcomplex *h_Val,int *h_Row,int *h_Col,
    dcomplex *h_b,dcomplex *h_x);

    void Gauss_Seidel_BASE_ZERO(int nRow,int nCol,
    dcomplex *h_Val,int *h_Row,int *h_Col,
    dcomplex *h_b,dcomplex *h_x);

    void Gauss_Seidel_BASE_ONE(int nRow,int nCol,
    dcomplex *h_Val,int *h_Row,int *h_Col,
    dcomplex *h_b,dcomplex *h_x);

    void symmetric_Gauss_Seidel(MKL_INT nRow,
    dcomplex *h_Val,MKL_INT *h_Row,MKL_INT *h_Col,
    dcomplex *h_b,dcomplex *h_x);

    void gmres_cpu_dcomplex_restart(
    int nter_max,float tolerant,int out_put,int m,int mkl_parallel,
    int nx,int nz,MKL_INT nzA,
    dcomplex *h_CsrVal,MKL_INT *h_CsrRowPtr,MKL_INT *h_CsrColInd,
    dcomplex *h_b,dcomplex *h_x);

    void solve_matrix(int mkl_parallel,MKL_Complex16 *h_ValA,MKL_INT *h_RowA,MKL_INT *h_ColA,
    MKL_INT nnz,MKL_Complex16 *h_B,MKL_Complex16 *h_X);

 
 
#endif

