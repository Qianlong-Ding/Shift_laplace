#ifndef _FRE_GMG_SOLVER_MKL_H_
#define _FRE_GMG_SOLVER_MKL_H_
#include <mpi.h>
#include "stdio.h"
#include "stdlib.h"
#include "par.h"
#include "su.h"

#include "mkl.h"


        void FunctionSparseSpmm(int mkl_parallel,MKL_INT nRow,MKL_INT nCol,
            dcomplex **h_ValCoarse,MKL_INT **h_ColCoarse,MKL_INT **h_RowCoarse,
            dcomplex *h_ValFine,MKL_INT *h_ColFine,MKL_INT *h_RowFine,
            dcomplex *h_ValR,MKL_INT *h_ColR,MKL_INT *h_RowR,
            dcomplex *h_ValP,MKL_INT *h_ColP,MKL_INT *h_RowP);

        void CoarseVector(int mkl_parallel,MKL_INT nRow,MKL_INT nCol,
            dcomplex *h_xc,dcomplex *h_xf,
            dcomplex *h_ValR,MKL_INT *h_RowR,MKL_INT *h_ColR
            );

        void FineVector(int mkl_parallel,MKL_INT nRow,MKL_INT nCol,
            dcomplex *h_xf,dcomplex *h_xc,
            dcomplex *h_ValP,MKL_INT *h_RowP,MKL_INT *h_ColP
            );

        void MultiGridVcycle(int mkl_parallel,int ilevel,int Level,float tolerant,int gmres_out,int gmres_smoother,int m,
                                int *NxIlevel,int *NzIlevel,int nxA,MKL_INT *RankIlevel,
                                MKL_Complex16 **h_Val,MKL_INT **h_Row,MKL_INT **h_Col,
                                MKL_Complex16 **h_ValR,MKL_INT **h_RowR,MKL_INT **h_ColR,
                                MKL_Complex16 **h_ValP,MKL_INT **h_RowP,MKL_INT **h_ColP,
                                MKL_Complex16 **h_ValSR,MKL_INT **h_RowSR,MKL_INT **h_ColSR,
                                MKL_Complex16 *h_B,MKL_Complex16 *h_X,MKL_Complex16 *h_X0);

        void Multigrid_Solver(int mkl_parallel,int nter_max,float tolerant,int gmres_out,int gmres_smoother,int m,
                              int myid,int npros,MPI_Comm comm,
                              int nxA,int Level,
                              int *NxIlevel,int *NzIlevel,MKL_INT *RankIlevel,
                              dcomplex **h_Val,MKL_INT **h_Row,MKL_INT **h_Col,
                              dcomplex **h_ValR,MKL_INT **h_RowR,MKL_INT **h_ColR,
                              dcomplex **h_ValP,MKL_INT **h_RowP,MKL_INT **h_ColP,
                              dcomplex **h_ValSR,MKL_INT **h_RowSR,MKL_INT **h_ColSR,
                              dcomplex *h_B,dcomplex *h_X,dcomplex *h_X0);

        void matrix_sort(dcomplex *h_Val,MKL_INT *h_Row,MKL_INT *h_Col,MKL_INT nRow,int mkl_parallel);


        void BiCGStab_Z_MKL_Precond(int mkl_parallel,int nter_max,double tolerant,MKL_INT nnz,
                                    dcomplex *h_Val,MKL_INT *h_Row,MKL_INT *h_Col,
                                    int precond,int nter_gmg,
                                    int gmres_out,int fgmres_out,int gmres_smoother,int m,int myid,int npros,MPI_Comm comm,int nxA,int Level,
                                    int *NxIlevel,int *NzIlevel,MKL_INT *RankIlevel,
                                    dcomplex **h_ValGMG,MKL_INT **h_RowGMG,MKL_INT **h_ColGMG,
                                    dcomplex **h_ValR,MKL_INT **h_RowR,MKL_INT **h_ColR,
                                    dcomplex **h_ValP,MKL_INT **h_RowP,MKL_INT **h_ColP,
                                    dcomplex **h_ValSR,MKL_INT **h_RowSR,MKL_INT **h_ColSR,
                                    dcomplex *h_b,dcomplex *h_x,float *res);

        int fgmres(int mkl_parallel,
                                    int precond,int nter_precond,int gmres_smoother,int fgmres_out,
                                    int nter_max,float *norm2_res,float tolerant,int gmres_out,int m,int m_g,
                                    int myid,int npros,MPI_Comm comm,
                                    int nxA,MKL_INT nzA,int Level,
                                    int *NxIlevel,int *NzIlevel,MKL_INT *RankIlevel,
                                    dcomplex *h_CsrVal,MKL_INT *h_CsrRowPtr,MKL_INT *h_CsrColInd,
                                    dcomplex **h_ValPrecond,MKL_INT **h_RowPrecond,MKL_INT **h_ColPrecond,
                                    dcomplex **h_ValR,MKL_INT **h_RowR,MKL_INT **h_ColR,
                                    dcomplex **h_ValP,MKL_INT **h_RowP,MKL_INT **h_ColP,
                                    dcomplex **h_ValSR,MKL_INT **h_RowSR,MKL_INT **h_ColSR,
                                    dcomplex *h_b,dcomplex *h_x);


 
#endif

