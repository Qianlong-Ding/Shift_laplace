#include "fre_solver_mkl.h"
#include "omp_string_subfunction.h"
#include "par.h"
#include "su.h"
#include <mpi.h>
#define ALIGN 64

void Gauss_Seidel(int nRow,int nCol,
    dcomplex *h_Val,int *h_Row,int *h_Col,
    dcomplex *h_b,dcomplex *h_x)
{
    int i;
    int icol;
    int irow;
    dcomplex Aii;
    dcomplex tempt;
    for(i=0;i<nRow;i++)
    {
        tempt.r = 0.0;
        tempt.i = 0.0;
        for(icol=h_Row[i];icol<h_Row[i+1];icol++)
        {
            if(h_Col[icol-1]==i+1)
            {
                Aii=h_Val[icol-1];
                continue;
            }
            tempt=dcadd(tempt,dcmul(h_Val[icol-1],h_x[h_Col[icol-1]-1]));
        }
        h_x[i] = dcdiv(dcsub(h_b[i],tempt),Aii);
        //h_x[i].r=1.0;
        //h_x[i].i=0.0;
    }
}

void Gauss_Seidel_BASE_ZERO(int nRow,int nCol,
    dcomplex *h_Val,int *h_Row,int *h_Col,
    dcomplex *h_b,dcomplex *h_x)
{
    int i;
    int icol;
    int irow;
    dcomplex Aii;
    dcomplex tempt;
    for(i=0;i<nRow;i++)
    {
        tempt.r = 0.0;
        tempt.i = 0.0;
        for(icol=h_Row[i];icol<h_Row[i+1];icol++)
        {
            if(h_Col[icol]==i)
            {
                Aii=h_Val[icol];
                continue;
            }
            tempt=dcadd(tempt,dcmul(h_Val[icol],h_x[h_Col[icol]]));
        }
        h_x[i] = dcdiv(dcsub(h_b[i],tempt),Aii);
        //h_x[i].r=1.0;
        //h_x[i].i=0.0;
    }
}

void Gauss_Seidel_BASE_ONE(int nRow,int nCol,
    dcomplex *h_Val,int *h_Row,int *h_Col,
    dcomplex *h_b,dcomplex *h_x)
{
    int i;
    int icol;
    int irow;
    dcomplex Aii;
    dcomplex tempt;
    for(i=0;i<nRow;i++)
    {
        tempt.r = 0.0;
        tempt.i = 0.0;
        for(icol=h_Row[i];icol<h_Row[i+1];icol++)
        {
            if(h_Col[icol-1]==i+1)
            {
                Aii=h_Val[icol-1];
                continue;
            }
            tempt=dcadd(tempt,dcmul(h_Val[icol-1],h_x[h_Col[icol-1]-1]));
        }
        h_x[i] = dcdiv(dcsub(h_b[i],tempt),Aii);
        //h_x[i].r=1.0;
        //h_x[i].i=0.0;
    }
}

void symmetric_Gauss_Seidel(MKL_INT nRow,
    dcomplex *h_Val,MKL_INT *h_Row,MKL_INT *h_Col,
    dcomplex *h_b,dcomplex *h_x)
{
    // mkl_sparse_z_symgs

    MKL_INT i;
    MKL_INT icol;
    MKL_INT irow;
    dcomplex Aii;
    dcomplex tempt;

    for(i=0;i<nRow;++i)
    {
        tempt.r = 0.0;
        tempt.i = 0.0;
        for(icol=h_Row[i];icol<h_Row[i+1];icol++)
        {
            if(h_Col[icol-1]==i+1)
            {
                Aii=h_Val[icol-1];
                continue;
            }
            tempt=dcadd(tempt,dcmul(h_Val[icol-1],h_x[h_Col[icol-1]-1]));
        }
        h_x[i] = dcdiv(dcsub(h_b[i],tempt),Aii);
    }


    for(i=nRow-1;i>=0;--i)
    {
        tempt.r = 0.0;
        tempt.i = 0.0;
        for(icol=h_Row[i];icol<h_Row[i+1];icol++)
        {
            if(h_Col[icol-1]==i+1)
            {
                Aii=h_Val[icol-1];
                continue;
            }
            tempt=dcadd(tempt,dcmul(h_Val[icol-1],h_x[h_Col[icol-1]-1]));
        }
        h_x[i] = dcdiv(dcsub(h_b[i],tempt),Aii);
    }
}

void gmres_cpu_dcomplex_restart(
    int nter_max,float tolerant,int out_put,int m,int mkl_parallel,
    int nx,int nz,MKL_INT nzA,
    dcomplex *h_CsrVal,MKL_INT *h_CsrRowPtr,MKL_INT *h_CsrColInd,
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
    dcomplex t;
    double norm2;
    dcomplex *re;
    double *res;

    dcomplex **h_hij;


    dcomplex *c;
    dcomplex *s;
    double gam;



    // re=alloc1dcomplex(m+1);
    // res=alloc1double(nter_max+1);

    // c=alloc1dcomplex(m);
    // s=alloc1dcomplex(m);

    // h_hij=alloc2dcomplex(m+1,m);
    // h_v=alloc1dcomplex(NINT((m+1)*nzA));

    mkl_set_num_threads(mkl_parallel);


    re=(dcomplex *)mkl_malloc(sizeof(dcomplex) * (m+1), ALIGN);
    res=(double *)mkl_malloc(sizeof(double) * (nter_max+1), ALIGN);

    c=(dcomplex *)mkl_malloc(sizeof(dcomplex) * (m), ALIGN);;
    s=(dcomplex *)mkl_malloc(sizeof(dcomplex) * (m), ALIGN);;

    h_hij=alloc2dcomplex(m+1,m);
    h_v=(dcomplex *)mkl_malloc(sizeof(dcomplex) * ((m+1)*nzA), ALIGN);


    memset(res,     0, sizeof(double)*(nter_max+1));

    



    incx=1;
    incy=1;

    iter=0;

    for(iter=0;iter<nter_max;)
    {
        //warn("nzA=%d",nzA);
        //warn("m=%d",m);
        //memset(h_v,          0, sizeof(dcomplex)*nzA*(m+1));
        memset(c,            0, sizeof(dcomplex)*(m));
        memset(s,            0, sizeof(dcomplex)*(m));
        memset(h_hij[0],     0, sizeof(dcomplex)*(m)*(m+1));
        memset(re,           0, sizeof(dcomplex)*(m+1));

        omp_memset_dcomplex( mkl_parallel,h_v, nzA*(m+1));
        //y=A*x
        //mkl_ccsrgemv("N",&nzA,h_CsrVal,h_CsrRowPtr,h_CsrColInd,h_x,h_v);
        mkl_zcsrgemv("N",&nzA,h_CsrVal,h_CsrRowPtr,h_CsrColInd,h_x,h_v);

        //y=alpha*Ax+beta*y
        //mkl_ccsrmv("N",nzA,nzA,alpha,,h_CsrVal,h_CsrColInd,&h_CsrRowPtr[0]);

        //y=alpha*Ax+beta*y
        //mkl_ccoomm("N",&nzA,&1,&nzA,alpha,)

        scal=-1.0;
        //cblas_csscal(nzA,scal,&h_v[0],1);
        cblas_zdscal(nzA,scal,&h_v[0],1);
        complex_scal.r=1.0;
        complex_scal.i=0.0;
        //y=a*x+y
        //cblas_caxpy(nzA,&complex_scal,h_b, incx, &h_v[0], incy);
        cblas_zaxpy(nzA,&complex_scal,h_b, incx, &h_v[0], incy);

        //norm2 = cblas_scnrm2(nzA,h_v,1);
        norm2 = cblas_dznrm2(nzA,h_v,1);

        if(iter==0)
        {
            tolerant=tolerant*norm2;
        }

        if((norm2<tolerant)||(iter>=nter_max))
        {
            break;
        }
        re[0].r=norm2;
        res[iter]=norm2;
        // warn("norm2=%f",norm2);
        scal=1.0/norm2;

        cblas_zdscal(nzA,scal,&h_v[0],1);

        im=0;

        while((im<m)&&(iter<nter_max))
        {
            mkl_zcsrgemv("N",&nzA,h_CsrVal,h_CsrRowPtr,h_CsrColInd,
                        &h_v[(im*nzA)],&h_v[((im+1)*nzA)]);

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
                //warn("norm->0 ,%f",norm2);
                break;
            }

            

            if(im>=1)
            {
                for(k=1;k<=im;k++)
                {
                    complex_scal=h_hij[im][k-1];
                    
                    h_hij[im][k-1]=dcadd(dcmul(dconjg(c[k-1]),complex_scal),
                                         dcmul(s[k-1],h_hij[im][k]));

                    h_hij[im][k]=dcsub(dcmul(c[k-1],h_hij[im][k]),
                                       dcmul(s[k-1],complex_scal));
                }
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

            if(out_put==1)
            {
                warn("cpu iter %d res %f",iter,res[iter]);
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

            /*for(k=im-1;k>=0;k--)
            {
                h_hij[k][k+1].r=0.0;
                h_hij[k][k+1].i=0.0;
            }

            cblas_ctrsv(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,
                im,h_hij[0],im,re,1);*/

            for(j=0;j<=(im-1);j++)
            {
                incx=1;
                incy=1;

                cblas_zaxpy(nzA,&re[j], &h_v[j*nzA],incx,&h_x[0], incy);
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
    free2dcomplex(h_hij);

}

// void solve_matrix(int mkl_parallel,MKL_Complex16 *h_ValA,MKL_INT *h_RowA,MKL_INT *h_ColA,
//     MKL_INT nnz,MKL_Complex16 *h_B,MKL_Complex16 *h_X)
// {
//     static MKL_INT n;
//     static MKL_Complex16 *h_Val=NULL;
//     static MKL_INT *h_Row=NULL;
//     static MKL_INT *h_Col=NULL;

//     static MKL_INT mtype = 13;       /* Real complex unsymmetric matrix */
//     static MKL_INT nrhs = 1;     /* Number of right hand sides. */
//     static void *pt[64];
//     static MKL_INT iparm[64];
//     static MKL_INT maxfct, mnum, phase, error, msglvl;

//     /* Auxiliary variables. */
//     static MKL_INT i, j;

//     static MKL_Complex16 ddum;       /* Double dummy */
//     static MKL_INT idum;         /* Integer dummy. */

//     for (i = 0; i < 64; i++)
//     {
//         iparm[i] = 0;
//     }
//     iparm[0] = 1;         /* No solver default */
//     iparm[1] = 2;         /* Fill-in reordering from METIS */
//     iparm[3] = 0;         /* No iterative-direct algorithm */
//     iparm[4] = 0;         /* No user fill-in reducing permutation */
//     iparm[5] = 0;         /* Write solution into x */
//     iparm[6] = 0;         /* Not in use */
//     iparm[7] = 2;         /* Max numbers of iterative refinement steps */
//     iparm[8] = 0;         /* Not in use */
//     iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
//     iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
//     iparm[11] = 0;        /* Conjugate transposed/transpose solve */
//     iparm[12] = 1;        /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
//     iparm[13] = 0;        /* Output: Number of perturbed pivots */
//     iparm[14] = 0;        /* Not in use */
//     iparm[15] = 0;        /* Not in use */
//     iparm[16] = 0;        /* Not in use */
//     iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
//     iparm[18] = -1;       /* Output: Mflops for LU factorization */
//     iparm[19] = 0;        /* Output: Numbers of CG Iterations */

//     maxfct = 1;           /* Maximum number of numerical factorizations.  */
//     mnum = 1;         /* Which factorization to use. */

//     msglvl = 0;           /* Print statistical information  */
//     error = 0;            /* Initialize error flag */


//     for(i = 0;i<64;i++)
//     {
//         pt[i]=0;
//     }

//     n=nnz;

//     mkl_set_num_threads(mkl_parallel);


//     h_Val=(MKL_Complex16 *)mkl_malloc(sizeof(MKL_Complex16)*(h_RowA[nnz]-h_RowA[0]),ALIGN);
//     h_Row=(MKL_INT *)mkl_malloc(sizeof(MKL_INT)*(nnz+1), ALIGN);
//     h_Col=(MKL_INT *)mkl_malloc(sizeof(MKL_INT)*(h_RowA[nnz]-h_RowA[0]), ALIGN);

//     omp_memcpy_MKL_INT( mkl_parallel,h_Col,h_ColA,(h_RowA[nnz]-h_RowA[0]));
//     omp_memcpy_dcomplex( mkl_parallel,h_Val,h_ValA,(h_RowA[nnz]-h_RowA[0]));

//     memcpy(h_Row,h_RowA,sizeof(MKL_INT)*(nnz+1));


//     phase = 11;

//     PARDISO(pt,&maxfct,&mnum,&mtype,&phase,&n,h_Val,h_Row,h_Col,&idum,&nrhs,iparm,&msglvl,&ddum,&ddum,&error);


//     if(error!=0)
//     {
//         warn("error=%d",error);
//     }

//     phase = 22;

//     PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
//        &n, h_Val, h_Row, h_Col, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);

//     if(error!=0)
//     {
//         warn("phase=%d\terror=%d",phase,error);
//     }


//     phase = 33;

//     iparm[11] = 0;
//     PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
//            &n, h_Val, h_Row, h_Col, &idum, &nrhs, iparm, &msglvl, h_B, h_X, &error);

//     if(error!=0)
//     {
//         warn("phase=%d\terror=%d",phase,error);
//     }


//     phase = -1;

//     PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
//        &n, &ddum, NULL, NULL, &idum, &nrhs,
//        iparm, &msglvl, &ddum, &ddum, &error);

//     if(error!=0)
//     {
//         warn("phase=%d\terror=%d",phase,error);
//     }
//     mkl_free(h_Val);
//     mkl_free(h_Row);
//     mkl_free(h_Col);
// }




void QMR(int mkl_parallel,int precond,int nter_max,float tolerant,int gmres_out,int m,
    int myid,int npros,MPI_Comm comm,
    int nxA,MKL_INT nzA,int ilevel,int Level,
    int *NxIlevel,int *NzIlevel,MKL_INT *RankIlevel,
    dcomplex *h_CsrVal,MKL_INT *h_CsrRowPtr,MKL_INT *h_CsrColInd,
    dcomplex **h_ValPrecond,MKL_INT **h_RowPrecond,MKL_INT **h_ColPrecond,
    dcomplex **h_ValR,MKL_INT **h_RowR,MKL_INT **h_ColR,
    dcomplex **h_ValP,MKL_INT **h_RowP,MKL_INT **h_ColP,
    dcomplex **h_ValSR,MKL_INT **h_RowSR,MKL_INT **h_ColSR,
    dcomplex *h_b,dcomplex *h_x)
{


	//bnorm2=norm(b);


}

void QMR_Precond(int mkl_parallel,int precond,int nter_max,float tolerant,int gmres_out,int m,
    int myid,int npros,MPI_Comm comm,
    int nxA,MKL_INT nzA,int ilevel,int Level,
    int *NxIlevel,int *NzIlevel,MKL_INT *RankIlevel,
    dcomplex *h_CsrVal,MKL_INT *h_CsrRowPtr,MKL_INT *h_CsrColInd,
    dcomplex **h_ValPrecond,MKL_INT **h_RowPrecond,MKL_INT **h_ColPrecond,
    dcomplex **h_ValR,MKL_INT **h_RowR,MKL_INT **h_ColR,
    dcomplex **h_ValP,MKL_INT **h_RowP,MKL_INT **h_ColP,
    dcomplex **h_ValSR,MKL_INT **h_RowSR,MKL_INT **h_ColSR,
    dcomplex *h_b,dcomplex *h_x)
{


}






