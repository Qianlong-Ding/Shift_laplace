#include "fre_gmg_solver_mkl.h"
#include "fre_solver_mkl.h"
#include "omp_string_subfunction.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "par.h"
#include "su.h"
#include "segy.h"
#include "mkl.h"
#include <omp.h>

#define ALIGN 64

void solve_matrix(int mkl_parallel,MKL_Complex16 *h_ValA,MKL_INT *h_RowA,MKL_INT *h_ColA,
    MKL_INT nnz,MKL_Complex16 *h_B,MKL_Complex16 *h_X)
{
    MKL_INT n;
    MKL_Complex16 *h_Val=NULL;
    MKL_INT *h_Row=NULL;
    MKL_INT *h_Col=NULL;

    MKL_INT mtype=13;       /* Real complex unsymmetric matrix */
    MKL_INT nrhs=1;     /* Number of right hand sides. */
    MKL_INT *pt[64];
    MKL_INT iparm[64];
    MKL_INT maxfct, mnum, phase, error, msglvl;

    /* Auxiliary variables. */
    MKL_INT i, j;

    MKL_Complex16 ddum;       /* Double dummy */
    MKL_INT idum;         /* Integer dummy. */
    FILE* fp;
    int izero;

    for (i = 0; i < 64; i++)
    {
        iparm[i]=0;
    }
    iparm[0]=1;         /* No solver default */
    iparm[1]=2;         /* Fill-in reordering from METIS */
    iparm[3]=0;         /* No iterative-direct algorithm */
    iparm[4]=0;         /* No user fill-in reducing permutation */
    iparm[5]=0;         /* Write solution into x */
    iparm[6]=0;         /* Not in use */
    iparm[7]=2;         /* Max numbers of iterative refinement steps */
    iparm[8]=0;         /* Not in use */
    iparm[9]=13;        /* Perturb the pivot elements with 1E-13 */
    iparm[10]=1;        /* Use nonsymmetric permutation and scaling MPS */
    iparm[11]=0;        /* Conjugate transposed/transpose solve */
    iparm[12]=1;        /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
    iparm[13]=0;        /* Output: Number of perturbed pivots */
    iparm[14]=0;        /* Not in use */
    iparm[15]=0;        /* Not in use */
    iparm[16]=0;        /* Not in use */
    iparm[17]=-1;       /* Output: Number of nonzeros in the factor LU */
    iparm[18]=-1;       /* Output: Mflops for LU factorization */
    iparm[19]=0;        /* Output: Numbers of CG Iterations */

    maxfct=1;           /* Maximum number of numerical factorizations.  */
    mnum=1;         /* Which factorization to use. */

    msglvl=0;           /* Print statistical information  */
    error=0;            /* Initialize error flag */


    for(i=0;i<64;i++)
    {
        pt[i]=0;
    }

    n=nnz;

 //   mkl_set_num_threads(mkl_parallel);


    h_Val=(MKL_Complex16 *)mkl_malloc(sizeof(MKL_Complex16)*(h_RowA[nnz]-h_RowA[0]),ALIGN);
    h_Row=(MKL_INT *)mkl_malloc(sizeof(MKL_INT)*(nnz+1), ALIGN);
    h_Col=(MKL_INT *)mkl_malloc(sizeof(MKL_INT)*(h_RowA[nnz]-h_RowA[0]), ALIGN);

    omp_memcpy_MKL_INT( mkl_parallel,h_Col,h_ColA,(h_RowA[nnz]-h_RowA[0]));
    omp_memcpy_dcomplex( mkl_parallel,h_Val,h_ValA,(h_RowA[nnz]-h_RowA[0]));
    
    

    memcpy(h_Row,h_RowA,sizeof(MKL_INT)*(nnz+1));
    
    
    //cblas_zcopy(nnz+1,h_ValA,1,h_Val,1);
    /*izero=0;
    fp = fopen("411col.txt", "w");
    for (MKL_INT i = 0; i < nnz; ++i)
    {
        MKL_INT temp_col;
        MKL_INT icol;
        MKL_INT jcol;
        dcomplex tempt_val;
        
        fprintf(fp,"%d\t",h_Row[i]);

        for(icol=h_Row[i];icol<=h_Row[i+1]-1;icol++)
        {
        	fprintf(fp,"%d\t",h_Col[icol-1]);
        	izero++;
        }
        fprintf(fp,"\n");
        
        fprintf(fp,"%d\t",h_Row[i]);

        for(icol=h_Row[i];icol<=h_Row[i+1]-1;icol++)
        {
        	fprintf(fp,"%e\t",h_Val[icol-1].real);
        }
        fprintf(fp,"\n");
     }
     fclose(fp);

		warn("MKL_INT=%d seizeocomplex16f=%d",sizeof(MKL_INT),sizeof(MKL_Complex16));*/
    phase = 11;
    //pt[34]=0;

//    PARDISO(pt,&maxfct,&mnum,&mtype,&phase,&n,h_Val,h_Row,h_Col,&idum,&nrhs,iparm,&msglvl,&ddum,&ddum,&error);
		PARDISO(pt,&maxfct,&mnum,&mtype,&phase,&n,h_Val,h_Row,h_Col,&idum,&nrhs,iparm,&msglvl,&ddum,&ddum,&error);


    if(error!=0)
    {
        warn("error=%d",error);
    }

    phase = 22;

    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
       &n, h_Val, h_Row, h_Col, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
       

    if(error!=0)
    {
        warn("phase=%d\terror=%d",phase,error);
    }


    phase = 33;

    iparm[11] = 0;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
           &n, h_Val, h_Row, h_Col, &idum, &nrhs, iparm, &msglvl, h_B, h_X, &error);

    if(error!=0)
    {
        warn("phase=%d\terror=%d",phase,error);
    }


    phase = -1;

    PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
       &n, &ddum, NULL, NULL, &idum, &nrhs,
       iparm, &msglvl, &ddum, &ddum, &error);

    if(error!=0)
    {
        warn("phase=%d\terror=%d",phase,error);
    }
    mkl_free(h_Val);
    mkl_free(h_Row);
    mkl_free(h_Col);
}

void FunctionSparseSpmm(int mkl_parallel,MKL_INT nRow,MKL_INT nCol,
            dcomplex **h_ValCoarse,MKL_INT **h_ColCoarse,MKL_INT **h_RowCoarse,
            dcomplex *h_ValFine,MKL_INT *h_ColFine,MKL_INT *h_RowFine,
            dcomplex *h_ValR,MKL_INT *h_ColR,MKL_INT *h_RowR,
            dcomplex *h_ValP,MKL_INT *h_ColP,MKL_INT *h_RowP)
{
    MKL_INT ii,i,j;
    sparse_index_base_t    indexing;
    MKL_INT  rows, cols;
    sparse_matrix_t csrA = NULL, csrR = NULL, csrC = NULL, csrP = NULL,csrCoaese = NULL;
    MKL_INT *pointerB_C = NULL,*pointerE_C = NULL,*columns_C = NULL;
    dcomplex *values_C;
    struct matrix_descr descr_type_gen;

    MKL_INT *h_row_C;
    dcomplex *h_value_C;
    MKL_INT *h_col_C;
    h_row_C=(MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (nRow + 1),ALIGN);

    indexing = SPARSE_INDEX_BASE_ONE;

    mkl_set_num_threads(mkl_parallel);

    mkl_sparse_z_create_csr( &csrA, SPARSE_INDEX_BASE_ONE, nCol, nCol, h_RowFine, h_RowFine+1, h_ColFine, h_ValFine );
    mkl_sparse_z_create_csr( &csrR, SPARSE_INDEX_BASE_ONE, nRow, nCol, h_RowR, h_RowR+1, h_ColR, h_ValR );
    mkl_sparse_z_create_csr( &csrP, SPARSE_INDEX_BASE_ONE, nCol, nRow, h_RowP, h_RowP+1, h_ColP, h_ValP );

    /* Compute C = A * B  */
    mkl_sparse_spmm( SPARSE_OPERATION_NON_TRANSPOSE, csrA, csrP, &csrC );
    mkl_sparse_spmm( SPARSE_OPERATION_NON_TRANSPOSE, csrR, csrC, &csrCoaese );


    mkl_sparse_z_export_csr( csrCoaese, &indexing, &rows, &cols, &pointerB_C, &pointerE_C, &columns_C, &values_C );

    h_row_C[0]=1;

    omp_memcpy_MKL_INT( mkl_parallel,&h_row_C[1],pointerE_C,rows);

    h_value_C = (dcomplex *)mkl_malloc(sizeof(dcomplex) * (h_row_C[rows]-h_row_C[0]),ALIGN);
    h_col_C = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (h_row_C[rows]-h_row_C[0]),ALIGN);



    
    omp_memcpy_dcomplex( mkl_parallel,h_value_C,values_C,(h_row_C[rows]-h_row_C[0]));
    omp_memcpy_MKL_INT( mkl_parallel,h_col_C,columns_C,(h_row_C[rows]-h_row_C[0]));


    *h_ColCoarse = h_col_C;
    *h_ValCoarse = h_value_C;
    *h_RowCoarse = h_row_C;


    mkl_sparse_destroy( csrA );
    mkl_sparse_destroy( csrR );
    mkl_sparse_destroy( csrC );
    mkl_sparse_destroy( csrP );
    mkl_sparse_destroy( csrCoaese );


}

void CoarseVector(int mkl_parallel,MKL_INT nRow,MKL_INT nCol,
    dcomplex *h_xc,dcomplex *h_xf,
    dcomplex *h_ValR,MKL_INT *h_RowR,MKL_INT *h_ColR
    )
{
    sparse_status_t ie_status;
    MKL_Complex16 alpha;
    MKL_Complex16 beta;
    struct matrix_descr descrA;
    sparse_matrix_t csrA = NULL;

    alpha.real = 1.0;
    alpha.imag = 0.0;

    beta.real = 0.0;
    beta.imag = 0.0;

    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

    mkl_set_num_threads(mkl_parallel);

    mkl_sparse_z_create_csr( &csrA, SPARSE_INDEX_BASE_ONE, nRow, nCol, h_RowR, h_RowR+1, h_ColR, h_ValR );

    ie_status = mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descrA, h_xf, beta, h_xc);

    if (ie_status != SPARSE_STATUS_SUCCESS)
    {
        printf(" Error in CoarseVector : %d\n", ie_status);
    }
    mkl_sparse_destroy( csrA );
}

void FineVector(int mkl_parallel,MKL_INT nRow,MKL_INT nCol,
    dcomplex *h_xf,dcomplex *h_xc,
    dcomplex *h_ValP,MKL_INT *h_RowP,MKL_INT *h_ColP
    )
{
    sparse_status_t ie_status;
    MKL_Complex16 alpha;
    MKL_Complex16 beta;
    sparse_matrix_t csrA = NULL;
    struct matrix_descr descrA;

    mkl_set_num_threads(mkl_parallel);

    alpha.real = 1.0;
    alpha.imag = 0.0;

    beta.real = 0.0;
    beta.imag = 0.0;

    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_z_create_csr( &csrA, SPARSE_INDEX_BASE_ONE, nRow, nCol, h_RowP, h_RowP+1, h_ColP, h_ValP );

    ie_status = mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descrA, h_xc, beta, h_xf);

    if (ie_status != SPARSE_STATUS_SUCCESS)
    {
        printf(" Error in FineVector : %d\n", ie_status);
    }
    mkl_sparse_destroy( csrA );
}


void MultiGridVcycle(int mkl_parallel,int ilevel,int Level,float tolerant,int gmres_out,int gmres_smoother,int m,
    int *NxIlevel,int *NzIlevel,int nxA,MKL_INT *RankIlevel,
    MKL_Complex16 **h_Val,MKL_INT **h_Row,MKL_INT **h_Col,
    MKL_Complex16 **h_ValR,MKL_INT **h_RowR,MKL_INT **h_ColR,
    MKL_Complex16 **h_ValP,MKL_INT **h_RowP,MKL_INT **h_ColP,
    MKL_Complex16 **h_ValSR,MKL_INT **h_RowSR,MKL_INT **h_ColSR,
    MKL_Complex16 *h_B,MKL_Complex16 *h_X,MKL_Complex16 *h_X0)
{
    int d_nx;
    int d_ny;

    int iter;
    int coarse;

    double norm1;

    MKL_Complex16 alpha;
    MKL_Complex16 beta;

    MKL_Complex16 *h_rFine;
    MKL_Complex16 *h_rCoarse;

    MKL_Complex16 *h_eFine;
    MKL_Complex16 *h_eCoarse;

    MKL_Complex16 *h_x0;

    float dxc;
    float dzc;
    float dxf;
    float dzf;

    int nxc;
    int nzc;
    int nxf;
    int nzf;

    int nnzc;
    int nnzf;

    dcomplex x0;

    int singularity = 0;
    double tol;
    const int reorder = 0;
    int i;

    int iiter;


    //gmres_out=1;


    mkl_set_num_threads(mkl_parallel);

    if( ilevel == Level-1 )
    {
    		/*warn("ilevel=%d Llevel=%d",ilevel,Level);
        solve_matrix(mkl_parallel,h_Val[ilevel],h_Row[ilevel],h_Col[ilevel],
                        RankIlevel[ilevel],h_B,h_X0);*/
                        
        gmres_cpu_dcomplex_restart( gmres_smoother, tolerant, gmres_out, m, mkl_parallel,
                                NxIlevel[ilevel], NzIlevel[ilevel], RankIlevel[ilevel],
                                h_Val[ilevel], h_Row[ilevel], h_Col[ilevel],
                                h_B, h_X0);

        memcpy(h_X,h_X0,sizeof(dcomplex)*RankIlevel[ilevel]);

        return 0;
    }

    h_rFine = (MKL_Complex16 *)mkl_malloc(sizeof(MKL_Complex16) * RankIlevel[ilevel], ALIGN);
    h_rCoarse = (MKL_Complex16 *)mkl_malloc(sizeof(MKL_Complex16) * RankIlevel[ilevel+1], ALIGN);
    h_eCoarse = (MKL_Complex16 *)mkl_malloc(sizeof(MKL_Complex16) * RankIlevel[ilevel+1], ALIGN);
    h_eFine = (MKL_Complex16 *)mkl_malloc(sizeof(MKL_Complex16) * RankIlevel[ilevel], ALIGN);
    h_x0 = (MKL_Complex16 *)mkl_malloc(sizeof(MKL_Complex16) * RankIlevel[ilevel+1], ALIGN);

    omp_memset_dcomplex( mkl_parallel,h_rFine,RankIlevel[ilevel]);
    omp_memset_dcomplex( mkl_parallel,h_eFine,RankIlevel[ilevel]);
    omp_memset_dcomplex( mkl_parallel,h_rCoarse,RankIlevel[ilevel+1]);
    omp_memset_dcomplex( mkl_parallel,h_eCoarse,RankIlevel[ilevel+1]);
    omp_memset_dcomplex( mkl_parallel,h_x0,RankIlevel[ilevel+1]);

     //warn("ilevel=%d",ilevel);
    gmres_cpu_dcomplex_restart( gmres_smoother, tolerant, gmres_out, m, mkl_parallel,
                                NxIlevel[ilevel], NzIlevel[ilevel], RankIlevel[ilevel],
                                h_Val[ilevel], h_Row[ilevel], h_Col[ilevel],
                                h_B, h_X0);


    // symmetric_Gauss_Seidel(RankIlevel[ilevel],
    // h_Val[ilevel], h_Row[ilevel], h_Col[ilevel],
    // h_B, h_X0);


    //warn("ilevel++=%d",ilevel);

    mkl_zcsrgemv("N",&RankIlevel[ilevel],h_Val[ilevel],h_Row[ilevel],h_Col[ilevel],h_X0,h_X);

    omp_memcpy_dcomplex(mkl_parallel,h_rFine,h_B,RankIlevel[ilevel]);

    alpha.real=-1.0;
    alpha.imag=0.0;

    cblas_zaxpy(RankIlevel[ilevel], &alpha, h_X, 1, h_rFine, 1);

    norm1 = cblas_dznrm2(RankIlevel[ilevel],h_rFine,1);


    // CoarseVector(mkl_parallel,RankIlevel[ilevel+1],RankIlevel[ilevel],
    //     h_rCoarse, h_rFine, h_ValSR[ilevel], h_RowSR[ilevel], h_ColSR[ilevel]);
		//warn("ilevel+=%d",ilevel);
    CoarseVector(mkl_parallel,RankIlevel[ilevel+1],RankIlevel[ilevel],
        h_rCoarse, h_rFine, h_ValR[ilevel], h_RowR[ilevel], h_ColR[ilevel]);

    MultiGridVcycle(mkl_parallel, ilevel+1, Level, tolerant, gmres_out, gmres_smoother, m,
                    NxIlevel, NzIlevel, nxA, RankIlevel,
                    h_Val, h_Row, h_Col,
                    h_ValR, h_RowR, h_ColR,
                    h_ValP, h_RowP, h_ColP,
                    h_ValSR, h_RowSR, h_ColSR,
                    h_rCoarse,h_eCoarse,h_x0);

    FineVector(mkl_parallel, RankIlevel[ilevel], RankIlevel[ilevel+1],
     h_eFine, h_eCoarse, h_ValP[ilevel], h_RowP[ilevel], h_ColP[ilevel]);

    alpha.real=1.0;
    alpha.imag=0.0;

    cblas_zaxpy(RankIlevel[ilevel], &alpha, h_eFine, 1, &h_X0[0], 1);

    gmres_cpu_dcomplex_restart( gmres_smoother, tolerant, gmres_out, m, mkl_parallel,
                                NxIlevel[ilevel], NzIlevel[ilevel], RankIlevel[ilevel],
                                h_Val[ilevel], h_Row[ilevel], h_Col[ilevel],
                                h_B, h_X0);

    // symmetric_Gauss_Seidel(RankIlevel[ilevel],
    // h_Val[ilevel], h_Row[ilevel], h_Col[ilevel],
    // h_B, h_X0);


    omp_memcpy_dcomplex(mkl_parallel,h_X,h_X0,RankIlevel[ilevel]);


    mkl_free(h_rFine);
    mkl_free(h_rCoarse);
    mkl_free(h_eCoarse);
    mkl_free(h_eFine);
    mkl_free(h_x0);

}



void Multigrid_Solver(int mkl_parallel,int nter_max,float tolerant,int gmres_out,int gmres_smoother,int m,
    int myid,int npros,MPI_Comm comm,
    int nxA,int Level,
    int *NxIlevel,int *NzIlevel,MKL_INT *RankIlevel,
    dcomplex **h_Val,MKL_INT **h_Row,MKL_INT **h_Col,
    dcomplex **h_ValR,MKL_INT **h_RowR,MKL_INT **h_ColR,
    dcomplex **h_ValP,MKL_INT **h_RowP,MKL_INT **h_ColP,
    dcomplex **h_ValSR,MKL_INT **h_RowSR,MKL_INT **h_ColSR,
    dcomplex *h_B,dcomplex *h_X,dcomplex *h_X0)
{
    int iter;
    double norm2;
    double norm;
    dcomplex *h_r;
    int i;

    double iStart, iElaps;

    dcomplex complex_scal;

    mkl_set_num_threads(mkl_parallel);

    // h_r = alloc1dcomplex(RankIlevel[0]);
    h_r = (dcomplex *)mkl_malloc(sizeof(dcomplex) * RankIlevel[0], ALIGN);

    // memset(h_r,  0, sizeof(dcomplex)*RankIlevel[0]);

    omp_memset_dcomplex( mkl_parallel,h_r,RankIlevel[0]);


    gmres_out = 0;

    for( iter = 0 ; iter < nter_max ; iter++ )
    {
        iStart = dsecnd();

        // if(iter==0)
        // {
        //     mkl_zcsrgemv("N",&RankIlevel[0],h_Val[0],h_Row[0],h_Col[0],h_X0,h_r);

        //     complex_scal.r=-1.0;
        //     complex_scal.i=0.0;
        //     cblas_zaxpy(RankIlevel[0],&complex_scal,h_B, 1, h_r, 1);
        //     norm2 = cblas_dznrm2(RankIlevel[0],h_r,1);

        //     norm=norm2;

        //     // warn("iter = %d \t norm2 = %f ",iter,norm2/norm);
        // }

        MultiGridVcycle( mkl_parallel, 0, Level, tolerant, gmres_out,gmres_smoother, m,
                        NxIlevel, NzIlevel, nxA, RankIlevel,
                        h_Val, h_Row, h_Col,
                        h_ValR, h_RowR, h_ColR,
                        h_ValP, h_RowP, h_ColP,
                        h_ValSR, h_RowSR, h_ColSR,
                        h_B, h_X, h_X0);

        omp_memcpy_dcomplex(mkl_parallel,h_X0,h_X,RankIlevel[0]);


        // mkl_zcsrgemv("N",&RankIlevel[0],h_Val[0],h_Row[0],h_Col[0],h_X0,h_r);

        // complex_scal.r=-1.0;
        // complex_scal.i=0.0;
        // cblas_zaxpy(RankIlevel[0],&complex_scal,h_B, 1, h_r, 1);
        // norm2 = cblas_dznrm2(RankIlevel[0],h_r,1);

        iElaps =dsecnd() - iStart;

        // warn("iter = %d \t norm2 = %f time=%lf",iter+1,norm2/norm,iElaps);

        // if(( (norm2/norm) < tolerant ) || ( iter >= nter_max - 1 ))
        // {
        //     break;
        // }
        if(( iter >= nter_max - 1 ))
        {
            break;
        }    
    }

    mkl_free(h_r);
}

void SparseMatrixTranspose(int mkl_parallel,MKL_INT nRow,MKL_INT nCol,
                           dcomplex *h_ValR,MKL_INT *h_RowR,MKL_INT *h_ColR,
                           dcomplex *h_ValP,MKL_INT *h_RowP,MKL_INT *h_ColP)
{
    MKL_INT *job;
    MKL_INT *temptRowR;
    MKL_INT *temptRowP;
    MKL_INT i;
    MKL_INT info;

    // job = alloc1int(6);
    // temptRowR = alloc1int(nCol+1);
    // temptRowP = alloc1int(nCol+1);

    job=(MKL_INT *)mkl_malloc(sizeof(MKL_INT) * 6, ALIGN);
    temptRowR=(MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (nCol+1), ALIGN);
    temptRowP=(MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (nCol+1), ALIGN);

    job[0] = 0;
    job[1] = 1;
    job[2] = 1;
    job[3] = 0;
    job[4] = 0;
    job[5] = 1;

    /*for(i=0;i<nCol+1;i++)
    {
        if(i<nRow+1)
        {
            temptRowR[i] = h_RowR[i];
        }
        else
        {
            temptRowR[i] = h_RowR[nRow];
        }
        
    }*/

    mkl_set_num_threads(mkl_parallel);

    memcpy(temptRowR,h_RowR,sizeof(MKL_INT)*(nRow+1));

    for(i=nRow+1;i<nCol+1;i++)
    {
        temptRowR[i] = h_RowR[nRow]; 
    }

    mkl_zcsrcsc(job,&nCol,h_ValR,h_ColR,temptRowR,h_ValP,h_ColP,temptRowP,&info);

    /*for(i=0;i<nCol+1;i++)
    {
        h_RowP[i] = temptRowP[i]; 
    }*/

    memcpy(h_RowP,temptRowP,sizeof(MKL_INT)*(nCol+1));


    mkl_free(job);
    mkl_free(temptRowR);
    mkl_free(temptRowP);

    

}

void matrix_sort(dcomplex *h_Val,MKL_INT *h_Row,MKL_INT *h_Col,MKL_INT nRow,int mkl_parallel)
{
    #pragma omp parallel for num_threads(mkl_parallel)
    for (MKL_INT i = 0; i < nRow; ++i)
    {
        MKL_INT temp_col;
        MKL_INT icol;
        MKL_INT jcol;
        dcomplex tempt_val;

        for(icol=h_Row[i];icol<=h_Row[i+1]-1;icol++)
        {
            for(jcol=icol+1;jcol<=h_Row[i+1]-1;jcol++)
            {
                if((h_Col[jcol-1]-h_Col[icol-1])<0)
                {
                    temp_col=h_Col[jcol-1];
                    h_Col[jcol-1]=h_Col[icol-1];
                    h_Col[icol-1]=temp_col;

                    tempt_val=h_Val[jcol-1];
                    h_Val[jcol-1]=h_Val[icol-1];
                    h_Val[icol-1]=tempt_val;

                }
            }
        }


    }
}




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
    dcomplex *h_b,dcomplex *h_x)
{

    dcomplex complex_scal;
    double scal;
    MKL_INT incx;
    MKL_INT incy;


    int iter;
    int j,k;
    int im;
    double error1;

    dcomplex *h_Ax;
    dcomplex *h_v;
    dcomplex *h_x0;
    dcomplex *h_z;
    dcomplex t;
    double norm2;
    double norm;
    double tolerant_real;
    dcomplex *re;
    double *res;

    dcomplex **h_hij;


    dcomplex *c;
    dcomplex *s;
    double gam;

    mkl_set_num_threads(mkl_parallel);


    re=(dcomplex *)mkl_malloc(sizeof(dcomplex) * (m+1), ALIGN);
    res=(double *)mkl_malloc(sizeof(double) * (nter_max+1), ALIGN);
    c=(dcomplex *)mkl_malloc(sizeof(dcomplex) * (m), ALIGN);;
    s=(dcomplex *)mkl_malloc(sizeof(dcomplex) * (m), ALIGN);;

    h_hij=alloc2dcomplex(m+1,m);
    h_v=(dcomplex *)mkl_malloc(sizeof(dcomplex) * ((m+1)*nzA), ALIGN);
    h_z=(dcomplex *)mkl_malloc(sizeof(dcomplex) * ((m+1)*nzA), ALIGN);
    h_x0=(dcomplex *)mkl_malloc(sizeof(dcomplex) * (nzA), ALIGN);


    memset(res,     0, sizeof(double)*(nter_max+1));


    incx=1;
    incy=1;

    iter=0;

    for(iter=0;iter<nter_max;)
    {
        memset(c,            0, sizeof(dcomplex)*(m));
        memset(s,            0, sizeof(dcomplex)*(m));
        memset(h_hij[0],     0, sizeof(dcomplex)*(m)*(m+1));
        memset(re,           0, sizeof(dcomplex)*(m+1));

        omp_memset_dcomplex( mkl_parallel,h_v, nzA*(m+1));
        omp_memset_dcomplex( mkl_parallel,h_z, nzA*(m+1));
        //y=A*x
        mkl_zcsrgemv("N",&nzA,h_CsrVal,h_CsrRowPtr,h_CsrColInd,h_x,h_v);

        scal=-1.0;
        cblas_zdscal(nzA,scal,&h_v[0],1);
        complex_scal.r=1.0;
        complex_scal.i=0.0;
        //y=a*x+y
        cblas_zaxpy(nzA,&complex_scal,h_b, incx, &h_v[0], incy);

        //norm2 = cblas_scnrm2(nzA,h_v,1);
        norm2 = cblas_dznrm2(nzA,h_v,1);

        switch(iter)
        {
            case 0:
                tolerant_real=tolerant*norm2;
                warn("tolerant=%.8f norm2=%lf tolerant_real=%lf",tolerant,norm2,tolerant_real);
                break;
        }

        // if(iter==0)
        // {
        //     tolerant_real=tolerant*norm2;
        //     warn("tolerant=%.8f norm2=%lf tolerant_real=%lf",tolerant,norm2,tolerant_real);
        // }

        if(((int)(norm2/tolerant_real)<1)||(iter>=nter_max))
        {
            break;
        }
        re[0].r=norm2;

        switch(fgmres_out)
        {
            case 1:
                warn("norm2 %f",norm2);
                break;
        }

        
        res[iter]=norm2;
        
        //norm2_res[iter]=res[iter]/res[0];
        norm2_res[iter]=log10(res[iter]/res[0]);

        scal=1.0/norm2;

        cblas_zdscal(nzA,scal,&h_v[0],1);

        im=0;

        while((im<m)&&(iter<nter_max))
        {
            //FGMRES Z=M^-1*V (im+1)=omega
            //Z=M^-1*&h_v[(im*nzA)]

            switch(precond)
            {
                case 1:
                    memset(h_x0,  0, sizeof(dcomplex)*nzA);
                    Multigrid_Solver( mkl_parallel, nter_precond, tolerant, gmres_out, gmres_smoother, m_g,
                          myid, npros, comm,
                          nxA, Level,
                          NxIlevel, NzIlevel, RankIlevel,
                          h_ValPrecond, h_RowPrecond, h_ColPrecond,
                          h_ValR, h_RowR, h_ColR,
                          h_ValP, h_RowP, h_ColP,
                          h_ValSR, h_RowSR, h_ColSR,
                          &h_v[(im*nzA)], &h_z[(im*nzA)], h_x0);

                    mkl_zcsrgemv("N",&nzA,h_CsrVal,h_CsrRowPtr,h_CsrColInd,
                        &h_z[(im*nzA)],&h_v[((im+1)*nzA)]);
                    break;

                default :
                    mkl_zcsrgemv("N",&nzA,h_CsrVal,h_CsrRowPtr,h_CsrColInd,
                        &h_v[(im*nzA)],&h_v[((im+1)*nzA)]);
                    break;
            }
    

            for(j=0;j<=im;j++)
            {
                cblas_zdotc_sub(nzA,&h_v[j*nzA],1,&h_v[(im+1)*nzA],1,&complex_scal);
                h_hij[im][j]=complex_scal;

                complex_scal.r=-complex_scal.r;
                complex_scal.i=-complex_scal.i;

                cblas_zaxpy(nzA,&complex_scal, &h_v[j*nzA], incx, &h_v[(im+1)*nzA], incy);

            }
            norm2 = cblas_dznrm2(nzA,&h_v[(im+1)*nzA],1);

            h_hij[im][im+1].r=norm2;

            if(sqrt(norm2*norm2)>1e-6)
            {
                scal=1.0/norm2;
                cblas_zdscal(nzA,scal,&h_v[(im+1)*nzA],1);
            }
            else
            {
                break;
            }

            
            switch(im)
            {
                case 0:
                    break;

                default :
                    for(k=1;k<=im;k++)
                    {
                        complex_scal=h_hij[im][k-1];
                    
                        h_hij[im][k-1]=dcadd(dcmul(dconjg(c[k-1]),complex_scal),
                                         dcmul(s[k-1],h_hij[im][k]));

                        h_hij[im][k]=dcsub(dcmul(c[k-1],h_hij[im][k]),
                                       dcmul(s[k-1],complex_scal));
                    }
                    break;
            }

            gam=sqrt(h_hij[im][im].r*h_hij[im][im].r+h_hij[im][im].i*h_hij[im][im].i
                +h_hij[im][im+1].r*h_hij[im][im+1].r+h_hij[im][im+1].i*h_hij[im][im+1].i);

            c[im].r=h_hij[im][im].r/gam;
            c[im].i=h_hij[im][im].i/gam;

            s[im].r=h_hij[im][im+1].r/gam;
            s[im].i=h_hij[im][im+1].i/gam;

            re[im+1]=dcsub(dcmplx(0.0,0.0),dcmul(s[im],re[im]));
            re[im]=dcmul(dconjg(c[im]),re[im]);

            h_hij[im][im]=dcadd(dcmul(dconjg(c[im]),h_hij[im][im]),
                                dcmul(s[im],h_hij[im][im+1]));

            res[iter+1]=sqrt(re[im+1].r*re[im+1].r+re[im+1].i*re[im+1].i);

            norm2_res[iter+1]=log10(res[iter+1]/res[0]);

            if(((int)(res[iter+1]/tolerant_real)<1))
            {
                break;
            }

            //if(out_put==1)

            switch(fgmres_out)
            {
                case 1:
                    warn("fgmres cpu iter %d res %f",iter,res[iter]);
                    break;
            }

            iter++;
            im++;



        }

        if(im>=2)
        {
            re[im-1]=dcdiv(re[im-1],h_hij[im-1][im-1]);
            for(k=im-2;k>=0;k--)
            {
                complex_scal=re[k];
                for(j=k+1;j<=im-1;j++)
                {
                    complex_scal=dcsub(complex_scal,dcmul(h_hij[j][k],re[j]));
                }
                re[k]=dcdiv(complex_scal,h_hij[k][k]);
            }

            for(j=0;j<=(im-1);j++)
            {
                incx=1;
                incy=1;

                switch(precond)
                {
                    case 1:
                        cblas_zaxpy(nzA,&re[j], &h_z[j*nzA],incx,&h_x[0], incy);
                        break;

                    default :
                        cblas_zaxpy(nzA,&re[j], &h_v[j*nzA],incx,&h_x[0], incy);
                        break;
                }

                // if(precond==1)
                // {
                //     cblas_zaxpy(nzA,&re[j], &h_z[j*nzA],incx,&h_x[0], incy);

                // }
                // else
                // {
                //     cblas_zaxpy(nzA,&re[j], &h_v[j*nzA],incx,&h_x[0], incy);
                // }
                
            }
        }
        else
        {
            break;
        }

    }


    mkl_free(re);
    mkl_free(res);

    mkl_free(c);
    mkl_free(s);

    mkl_free(h_v);
    mkl_free(h_z);
    mkl_free(h_x0);
    free2dcomplex(h_hij);

    return(iter);

}


void BiCGStab_Z_MKL_Precond(int mkl_parallel,int nter_max,double tolerant,MKL_INT nnz,
    dcomplex *h_Val,MKL_INT *h_Row,MKL_INT *h_Col,
    int precond,int nter_gmg,
    int gmres_out,int fgmres_out,int gmres_smoother,int m,int myid,int npros,MPI_Comm comm,int nxA,int Level,
    int *NxIlevel,int *NzIlevel,MKL_INT *RankIlevel,
    dcomplex **h_ValGMG,MKL_INT **h_RowGMG,MKL_INT **h_ColGMG,
    dcomplex **h_ValR,MKL_INT **h_RowR,MKL_INT **h_ColR,
    dcomplex **h_ValP,MKL_INT **h_RowP,MKL_INT **h_ColP,
    dcomplex **h_ValSR,MKL_INT **h_RowSR,MKL_INT **h_ColSR,
    dcomplex *h_b,dcomplex *h_x,float *res)
{
    dcomplex *h_r;
    dcomplex *h_Ax;
    dcomplex *h_rtld;
    dcomplex *tmp_arry;
    dcomplex *h_p;
    dcomplex *h_v;
    dcomplex *h_phat;
    dcomplex *h_shat;
    dcomplex *h_t;
    dcomplex *h_s;
    dcomplex *h_x0;

    dcomplex *h_Kt;
    dcomplex *h_Ks;


    dcomplex tempt_val1;
    dcomplex tempt_val2;

    dcomplex complex_scal;
    dcomplex alpha;
    dcomplex omega;
    dcomplex beta;
    dcomplex rho;
    dcomplex rho_1;


    int iter;
    int flag;
    int i;

    double bnorm2;
    double norm2;

    mkl_set_num_threads(mkl_parallel);

    bnorm2 = cblas_dznrm2(nnz,h_b,1);


    if(bnorm2<1e-6)
    {
        bnorm2=1.0;
    }

    h_r=alloc1dcomplex(nnz);
    h_Ax=alloc1dcomplex(nnz);
    h_rtld=alloc1dcomplex(nnz);
    tmp_arry=alloc1dcomplex(nnz);
    h_p=alloc1dcomplex(nnz);
    h_v=alloc1dcomplex(nnz);
    h_phat=alloc1dcomplex(nnz);
    h_shat=alloc1dcomplex(nnz);
    h_t=alloc1dcomplex(nnz);
    h_s=alloc1dcomplex(nnz);
    h_x0=alloc1dcomplex(nnz);

    h_Kt=alloc1dcomplex(nnz);
    h_Ks=alloc1dcomplex(nnz);

    omp_memset_dcomplex(mkl_parallel,h_r,nnz);
    omp_memset_dcomplex(mkl_parallel,h_Ax,nnz);
    omp_memset_dcomplex(mkl_parallel,h_rtld,nnz);
    omp_memset_dcomplex(mkl_parallel,tmp_arry,nnz);
    omp_memset_dcomplex(mkl_parallel,h_p,nnz);
    omp_memset_dcomplex(mkl_parallel,h_v,nnz);
    omp_memset_dcomplex(mkl_parallel,h_phat,nnz);
    omp_memset_dcomplex(mkl_parallel,h_shat,nnz);
    omp_memset_dcomplex(mkl_parallel,h_t,nnz);
    omp_memset_dcomplex(mkl_parallel,h_s,nnz);
    omp_memset_dcomplex(mkl_parallel,h_x0,nnz);
    omp_memset_dcomplex(mkl_parallel,h_Kt,nnz);
    omp_memset_dcomplex(mkl_parallel,h_Ks,nnz);


    mkl_zcsrgemv("N",&nnz,h_Val,h_Row,h_Col,h_x,h_Ax);

    complex_scal.r=-1.0;
    complex_scal.i=0.0;

    omp_memcpy_dcomplex( mkl_parallel, h_r, h_b, nnz);

    cblas_zaxpy(nnz,&complex_scal,h_Ax, 1, h_r, 1);

    norm2=cblas_dznrm2(nnz,h_r,1);
    res[0]=norm2/bnorm2;


    switch(fgmres_out)
    {
        case 1:warn("bnorm2=%f",bnorm2);break;
    }

    

    if(res[0]<tolerant)
    {
        return;
    }

    omega.r=1.0;
    omega.i=0.0;
    
    omp_memcpy_dcomplex( mkl_parallel, h_rtld, h_r, nnz);

    rho_1.r=1.0;
    rho_1.i=0.0;

    omega.r=1.0;
    omega.i=0.0;
    alpha.r=1.0;
    alpha.i=0.0;
    // omp_memset_dcomplex(mkl_parallel,h_v,nnz);
    // omp_memset_dcomplex(mkl_parallel,h_p,nnz);



    for(iter=0;iter<nter_max;iter++)
    {
         cblas_zdotc_sub(nnz,h_rtld,1,h_r,1,&rho);

        //cblas_zdotc_sub(nnz,h_r,1,h_rtld,1,&rho);

        if(sqrt(rho.r*rho.r+rho.i*rho.i)<1e-12)
        {
            warn("first rho=0!!");
            break;
        }

        // switch((int)(sqrt(rho.r*rho.r+rho.i*rho.i)/1e-12))
        // {
        //     case 0:
        //         warn("first rho=0!!");
        //         break;
        // }

        switch(iter)
        {
            case 0:
                omp_memcpy_dcomplex( mkl_parallel, h_p,h_r, nnz);
                break;

            default :
                beta=dcmul(dcdiv(rho,rho_1),dcdiv(alpha,omega));
                complex_scal.r=-1.0*omega.r;
                complex_scal.i=-1.0*omega.i;
                omp_memcpy_dcomplex( mkl_parallel, tmp_arry, h_p, nnz);
                cblas_zaxpy(nnz,&complex_scal,h_v, 1, tmp_arry, 1);
                omp_memcpy_dcomplex( mkl_parallel, h_p, h_r, nnz);

                cblas_zaxpy(nnz,&beta,tmp_arry, 1, h_p, 1);
                break;
        }


        switch(precond)
        {
            case 1:
                omp_memset_dcomplex( mkl_parallel,h_x0, nnz);
                Multigrid_Solver( mkl_parallel, nter_gmg, tolerant, gmres_out, gmres_smoother, m,
                          myid, npros, comm,
                          nxA, Level,
                          NxIlevel, NzIlevel, RankIlevel,
                          h_ValGMG, h_RowGMG, h_ColGMG,
                          h_ValR, h_RowR, h_ColR,
                          h_ValP, h_RowP, h_ColP,
                          h_ValSR, h_RowSR, h_ColSR,
                          h_p, h_phat, h_x0);
                break;

            default :
                omp_memcpy_dcomplex( mkl_parallel, h_phat,h_p, nnz);
                break;
        }

        //v = A*p_hat;
        mkl_zcsrgemv("N",&nnz,h_Val,h_Row,h_Col,h_phat,h_v);

        //alpha = rho / ( r_tld'*v );
         cblas_zdotc_sub(nnz,h_rtld,1,h_v,1,&tempt_val1);
        // cblas_zdotc_sub(nnz,h_v,1,h_rtld,1,&tempt_val1);
        alpha=dcdiv(rho,tempt_val1);

        //s = r - alpha*v;
        complex_scal.r=-1.0*alpha.r;
        complex_scal.i=-1.0*alpha.i;

        omp_memcpy_dcomplex( mkl_parallel, h_s, h_r, nnz);
        cblas_zaxpy(nnz,&complex_scal,h_v, 1, h_s, 1);

        // early convergence check
        norm2=cblas_dznrm2(nnz,h_s,1);
        if(norm2<=tolerant)
        {
            cblas_zaxpy(nnz,&alpha,h_phat, 1, h_x, 1);

            res[iter+1]=norm2/bnorm2;

            break;
        }

        switch(precond)
        {
            case 1:
                omp_memset_dcomplex( mkl_parallel,h_x0, nnz);

                Multigrid_Solver( mkl_parallel, nter_gmg, tolerant, gmres_out, gmres_smoother, m,
                          myid, npros, comm,
                          nxA, Level,
                          NxIlevel, NzIlevel, RankIlevel,
                          h_ValGMG, h_RowGMG, h_ColGMG,
                          h_ValR, h_RowR, h_ColR,
                          h_ValP, h_RowP, h_ColP,
                          h_ValSR, h_RowSR, h_ColSR,
                          h_s, h_shat, h_x0);
                break;

            default :
                omp_memcpy_dcomplex( mkl_parallel, h_shat,h_s, nnz);
                break;
        }

        //t = A*s_hat;
        mkl_zcsrgemv("N",&nnz,h_Val,h_Row,h_Col,h_shat,h_t);

        switch(precond)
        {
            case 1:
                omp_memset_dcomplex( mkl_parallel,h_x0, nnz);
                Multigrid_Solver( mkl_parallel, nter_gmg, tolerant, gmres_out,gmres_smoother, m,
                          myid, npros, comm,
                          nxA, Level,
                          NxIlevel, NzIlevel, RankIlevel,
                          h_ValGMG, h_RowGMG, h_ColGMG,
                          h_ValR, h_RowR, h_ColR,
                          h_ValP, h_RowP, h_ColP,
                          h_ValSR, h_RowSR, h_ColSR,
                          h_t, h_Kt, h_x0);

                omp_memcpy_dcomplex( mkl_parallel, h_Ks,h_shat, nnz);

                break;

            default :
                omp_memcpy_dcomplex( mkl_parallel, h_Ks,h_s, nnz);
                omp_memcpy_dcomplex( mkl_parallel, h_Kt,h_t, nnz);
                break;
        }

        //omega = ( t'*s) / ( t'*t );
        cblas_zdotc_sub(nnz,h_Kt,1,h_Ks,1,&tempt_val1);
        cblas_zdotc_sub(nnz,h_Kt,1,h_Kt,1,&tempt_val2);

        // cblas_zdotc_sub(nnz,h_Ks,1,h_Kt,1,&tempt_val1);
        // cblas_zdotc_sub(nnz,h_Kt,1,h_Kt,1,&tempt_val2);
        omega=dcdiv(tempt_val1,tempt_val2);

        //x = x + alpha*p_hat + omega*s_hat;              update approximation
        cblas_zaxpy(nnz,&alpha,h_phat, 1, h_x, 1);
        cblas_zaxpy(nnz,&omega,h_shat, 1, h_x, 1);

        //r = s - omega*t; 
        // memcpy(h_r,h_s,sizeof(dcomplex)*nnz);
        omp_memcpy_dcomplex( mkl_parallel, h_r, h_s, nnz);
        complex_scal.r=-1.0*omega.r;
        complex_scal.i=-1.0*omega.i;
        cblas_zaxpy(nnz,&complex_scal,h_t, 1, h_r, 1);

        //error(iter) = norm( r ) / bnrm2;                 check convergence
        norm2=cblas_dznrm2(nnz,h_r,1);
        res[iter+1]=norm2/bnorm2;

        switch(fgmres_out)
        {
            case 1:
                warn("res[%d]=%f",iter,res[iter]);
                break;
        }

        

        if(res[iter+1]<=tolerant)
        {
            warn("res enough!!");
            break;
        }

        if(sqrt(rho.r*rho.r+rho.i*rho.i)<1e-12)
        {

            warn("second rho=0!!");
            break;
        }

        rho_1=rho;
    }

    if(res[iter]<=tolerant)
    {
        flag=0;
    }
    else if(iter==nter_max-1)
    {
        flag=-1;
    }


    free1dcomplex(h_r);
    free1dcomplex(h_Ax);
    free1dcomplex(h_rtld);
    free1dcomplex(tmp_arry);
    free1dcomplex(h_p);
    free1dcomplex(h_v);
    free1dcomplex(h_phat);
    free1dcomplex(h_shat);
    free1dcomplex(h_t);
    free1dcomplex(h_s);
    free1dcomplex(h_x0);

    free1dcomplex(h_Ks);
    free1dcomplex(h_Kt);

}



