#include <stdio.h>
#include <stdlib.h>
#include <cusparse_v2.h>
#include <cuda.h>
#include <cusparse.h>
#include <common.h>

#include "time.h"
#include "par.h"
#include "su.h"
#include "segy.h"
#include <cuda_runtime.h>
#include "cufft.h"
#include <mpi.h>
#include <assert.h>
#include <ctype.h>
#include "cusolverSp.h"

#include "helper_cuda.h"
#include "helper_cusolver.h"

int main(int argc, char **argv)
{
    int ix;
    int iz;
    int nx;
    int nz;
    int no_zero;


    cuFloatComplex *h_A;
    cuFloatComplex *h_CsrValA;
    cuFloatComplex *h_b;
    cuFloatComplex *h_x;

    cuFloatComplex *d_A;
    cuFloatComplex *d_csrVal;
    cuFloatComplex *d_b;
    cuFloatComplex *d_x;

    int *h_CsrColIndA;
    int *h_CsrRowPtrA;

    int *d_csrColInd;
    int *d_csrRowPtr;

    cuFloatComplex alpha;
    cuFloatComplex beta;

    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;

    cusparseMatDescr_t descr_M = 0;
    cusparseMatDescr_t descr_L = 0;
    cusparseMatDescr_t descr_U = 0;
    csrilu02Info_t info_M;
    csrsv2Info_t  info_L;
    csrsv2Info_t info_U;
    int pBufferSize_M;
    int pBufferSize_L;
    int pBufferSize_U;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    //const double alpha = 1.;
    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_U  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseStatus_t status;
    int m,nnz;

    cuFloatComplex *d_z;
    cuFloatComplex *d_y;

    cusparseCreate(&handle);

    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    nx=nz=4;

    cudaMalloc((cuFloatComplex **)&d_z,      sizeof(cuFloatComplex)*nx);
    cudaMalloc((cuFloatComplex **)&d_y,      sizeof(cuFloatComplex)*nx);


    h_A = (cuFloatComplex *)malloc(sizeof(cuFloatComplex) * (nx)*(nx));
    h_CsrValA= (cuFloatComplex *)malloc(sizeof(cuFloatComplex) * (nx)*(nx));

    h_b= (cuFloatComplex *)malloc(sizeof(cuFloatComplex) * (nx));
    h_x= (cuFloatComplex *)malloc(sizeof(cuFloatComplex) * (nx));

    h_CsrColIndA= (int *)malloc(sizeof(int) * (nx)*(nx));
    h_CsrRowPtrA= (int *)malloc(sizeof(int) * (nx+1));

    memset(h_A     ,         0, sizeof(cuFloatComplex)*(nx)*(nx));
    memset(h_CsrValA,        0, sizeof(cuFloatComplex)*(nx)*(nx));
    memset(h_b     ,         0, sizeof(cuFloatComplex)*(nx));
    memset(h_x     ,         0, sizeof(cuFloatComplex)*(nx));

    memset(h_CsrColIndA ,        0, sizeof(int)*(nx)*(nx));
    memset(h_CsrRowPtrA ,        0, sizeof(int)*(nx+1));

    cudaMalloc((cuFloatComplex **)&d_csrVal,      sizeof(cuFloatComplex)*nx*nx);
    cudaMalloc((cuFloatComplex **)&d_b,            sizeof(cuFloatComplex)*nx);
    cudaMalloc((cuFloatComplex **)&d_x,            sizeof(cuFloatComplex)*nx);
    cudaMalloc((int **)&d_csrColInd,              sizeof(int)*nx*(nx));
    cudaMalloc((int **)&d_csrRowPtr,              sizeof(int)*(nx+1));


    /*cudaMemset(d_CsrValA,            0, sizeof(cuComplex)*nx);
    cudaMemset(d_b,            0, sizeof(cuComplex)*nx);
    cudaMemset(d_x,            0, sizeof(cuComplex)*nx);
    cudaMemset(d_x,            0, sizeof(cuComplex)*nx);
    cudaMemset(d_x,            0, sizeof(cuComplex)*nx);*/

    //cudaMemset(d_CsrValA,            0, sizeof(cuComplex)*nx*nx);

    ix=0;
    iz=0;
    h_A[ix*nx+iz].x=2.0;h_A[ix*nx+iz].y=2.0; ix=1;iz=0; h_A[ix*nx+iz].x=2.0;h_A[ix*nx+iz].y=2.0;

    ix=0;
    iz=1;
    h_A[ix*nx+iz].x=3.0;h_A[ix*nx+iz].y=3.0; ix=1;iz=1; h_A[ix*nx+iz].x=2.0;h_A[ix*nx+iz].y=2.0; ix=2;iz=1; h_A[ix*nx+iz].x=4.0;h_A[ix*nx+iz].y=4.0;

    ix=1;
    iz=2;
    h_A[ix*nx+iz].x=5.0;h_A[ix*nx+iz].y=5.0; ix=2;iz=2; h_A[ix*nx+iz].x=1.0;h_A[ix*nx+iz].y=1.0; ix=3;iz=2; h_A[ix*nx+iz].x=3.0;h_A[ix*nx+iz].y=3.0;


    ix=2;
    iz=3;
    h_A[ix*nx+iz].x=2.0;h_A[ix*nx+iz].y=2.0; ix=3;iz=3; h_A[ix*nx+iz].x=3.0;h_A[ix*nx+iz].y=3.0;

    no_zero=0;

    for(iz=0;iz<nz;iz++)
    {
        for(ix=0;ix<nx;ix++)
        {
            if((h_A[ix*nx+iz].x*h_A[ix*nx+iz].x)>1e-6)
            {
                h_CsrValA[no_zero].x=h_A[ix*nx+iz].x;
                h_CsrValA[no_zero].y=h_A[ix*nx+iz].y;
                printf("h_CsrValA[no_zero].x=%f\n",h_CsrValA[no_zero].x);
                h_CsrColIndA[no_zero]=ix;
                no_zero++;
            }
            
        }
        h_CsrRowPtrA[iz+1]=no_zero;
        h_b[iz].x=1.0;
        h_x[iz].x=1.0;
    }

    printf("no_zero=%d\n",no_zero);



    cudaMemcpy(d_csrVal,h_CsrValA ,sizeof(cuFloatComplex)*nx*nx,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b ,sizeof(cuFloatComplex)*nx,cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,h_x ,sizeof(cuFloatComplex)*nx,cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd,h_CsrColIndA ,sizeof(int)*nx*(nx),cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrRowPtr,h_CsrRowPtrA ,sizeof(int)*(nx+1),cudaMemcpyHostToDevice);

    cudaMemset(d_b,            0, sizeof(cuComplex)*nx);
    alpha.x=1.0;
    alpha.y=0.0;
    beta.x=0.0;
    beta.y=0.0;
    //y = alpha * op(A) * x  + beta * y
    //CUSPARSE_OPERATION_TRANSPOSE
    //CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
    //CUSPARSE_OPERATION_NON_TRANSPOSE
    cusparseCcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,nz,nx,no_zero,&alpha,
        descr,d_csrVal,d_csrRowPtr,d_csrColInd,
            d_x,&beta,d_b);
    /*cusparseCcsrmv(handle,CUSPARSE_OPERATION_TRANSPOSE,nz,nx,no_zero,&alpha,
        descr,d_CsrValA,d_CsrRowPtrA,d_CsrColIndA,
            d_x,&beta,d_b);*/

    (cudaMemcpy(h_b,d_b,sizeof(cuFloatComplex)*nx,cudaMemcpyDeviceToHost));

    for(ix=0;ix<nx;ix++)
    {
        printf("d_b[%d]=%f+%fi\n",ix,h_b[ix].x,h_b[ix].y);
    }

    m=nx;
    nnz=no_zero;

    checkCudaErrors(cudaMemcpy(h_CsrValA,d_csrVal,sizeof(cuFloatComplex)*nx*nx,cudaMemcpyDeviceToHost));
for(ix=0;ix<no_zero;ix++)
    {
        //printf("ilu [%d]=%f\n",ix,h_CsrValA[ix].x);
    }

checkCudaErrors(cusparseCreateMatDescr(&descr_M));
cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

cusparseCreateMatDescr(&descr_L);
cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

cusparseCreateMatDescr(&descr_U);
cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

cusparseCreateCsrilu02Info(&info_M);
cusparseCreateCsrsv2Info(&info_L);
cusparseCreateCsrsv2Info(&info_U);

cusparseCcsrilu02_bufferSize(handle, m, nnz,
    descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_M, &pBufferSize_M);

cusparseCcsrsv2_bufferSize(handle, trans_L, m, nnz,
    descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_L, &pBufferSize_L);

cusparseCcsrsv2_bufferSize(handle, trans_U, m, nnz,
    descr_U, d_csrVal, d_csrRowPtr, d_csrColInd, info_U, &pBufferSize_U);

pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));

cudaMalloc((void**)&pBuffer, pBufferSize);

cusparseCcsrilu02_analysis(handle, m, nnz, descr_M,
    d_csrVal, d_csrRowPtr, d_csrColInd, info_M,
    policy_M, pBuffer);

status = cusparseXcsrilu02_zeroPivot(handle, info_M, &structural_zero);
if (CUSPARSE_STATUS_ZERO_PIVOT == status){
   printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
}

cusparseCcsrsv2_analysis(handle, trans_L, m, nnz, descr_L,
    d_csrVal, d_csrRowPtr, d_csrColInd,
    info_L, policy_L, pBuffer);
cusparseCcsrsv2_analysis(handle, trans_U, m, nnz, descr_U,
    d_csrVal, d_csrRowPtr, d_csrColInd,
    info_U, policy_U, pBuffer);

// step 5: M = L * U
cusparseCcsrilu02(handle, m, nnz, descr_M,
    d_csrVal, d_csrRowPtr, d_csrColInd, info_M, policy_M, pBuffer);

status = cusparseXcsrilu02_zeroPivot(handle, info_M, &numerical_zero);
if (CUSPARSE_STATUS_ZERO_PIVOT == status){
   printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
}


checkCudaErrors(cudaMemcpy(h_CsrValA,d_csrVal,sizeof(cuFloatComplex)*nx*nx,cudaMemcpyDeviceToHost));

    for(ix=0;ix<no_zero;ix++)
    {
        printf("ilu [%d]=%f+%fi\n",ix,h_CsrValA[ix].x,h_CsrValA[ix].y);
    }


// step 6: solve L*z = x


/*checkCudaErrors(cusparseCcsrsv2_solve(handle, trans_L, m, nnz, &alpha, descr_L,
   d_csrVal, d_csrRowPtr, d_csrColInd, info_L,
   d_b, d_z, policy_L, pBuffer));

memset( h_x,        0, sizeof(cuComplex)*(nx));
checkCudaErrors(cudaMemcpy(h_x,d_z,sizeof(cuFloatComplex)*nx,cudaMemcpyDeviceToHost));

    for(ix=0;ix<nx;ix++) 
    {
        printf("ilu d_z[%d]=%f+%fi\n",ix,h_x[ix].x,h_x[ix].y);
    }
// step 7: solve U*y = z

checkCudaErrors(cusparseCcsrsv2_solve(handle, trans_U, m, nnz, &alpha, descr_U,
    d_csrVal, d_csrRowPtr, d_csrColInd, info_U,
    d_z, d_y, policy_U, pBuffer));

memset( h_x,        0, sizeof(cuComplex)*(nx));
checkCudaErrors(cudaMemcpy(h_x,d_y,sizeof(cuFloatComplex)*nx,cudaMemcpyDeviceToHost));

    for(ix=0;ix<nx;ix++)
    {
        printf("ilu d_y[%d]=%f+%fi\n",ix,h_x[ix].x,h_x[ix].y);
    }*/




// step 6: free resources
cudaFree(pBuffer);
cusparseDestroyMatDescr(descr_M);
cusparseDestroyMatDescr(descr_L);
cusparseDestroyMatDescr(descr_U);
cusparseDestroyCsrilu02Info(info_M);
cusparseDestroyCsrsv2Info(info_L);
cusparseDestroyCsrsv2Info(info_U);





    free(h_A);
    free(h_CsrValA);
    free(h_b);
    free(h_x);
    free(h_CsrColIndA);
    free(h_CsrRowPtrA);

    cudaFree(d_csrVal);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_csrColInd);
    cudaFree(d_csrRowPtr);
    cudaFree(d_y);
    cudaFree(d_z);

    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);





}