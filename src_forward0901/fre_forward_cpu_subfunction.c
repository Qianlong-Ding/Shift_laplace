#include "fre_forward_cpu_subfunction.h"
#include "fre_compute_sparse_matrix3d_mkl.h"
#include "fre_gmg_solver_mkl.h"
#include "par.h"
#include "su.h"
#include "segy.h"
#include "time.h"
// #include <omp.h>
#include <assert.h>


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mkl.h"

// #include <mkl_cluster_sparse_solver.h>
#define LOOKFAC 2 
#ifdef MKL_ILP64
#define INT_PRINT_FORMAT "%lld"
#else
#define INT_PRINT_FORMAT "%d"
#endif
segy verttr;
segy horiztr;


#define ALIGN 64

int ComputeSR1d(int *OffsetSRx,int *OffsetSRy,int *OffsetSRz,float *SR,int srmaxnum)
{
    int ix;
    int iy;
    int iz;
    int Snine;
    float x;
    float y;
    float z;

    float r;
    float rflag;

    float G;
    float Sigmma;
    float Sigmma2;

    char fnm[BUFSIZ];
    FILE *otfp=NULL;

    int add;


    memset(OffsetSRx, 0, sizeof(int)*(srmaxnum*2+1)*(srmaxnum*2+1)*(srmaxnum*2+1));
    memset(OffsetSRy, 0, sizeof(int)*(srmaxnum*2+1)*(srmaxnum*2+1)*(srmaxnum*2+1));
    memset(OffsetSRz, 0, sizeof(int)*(srmaxnum*2+1)*(srmaxnum*2+1)*(srmaxnum*2+1));

    memset(SR,        0, sizeof(float)*(srmaxnum*2+1)*(srmaxnum*2+1)*(srmaxnum*2+1));

    rflag=(float)srmaxnum*srmaxnum;
    Snine=0;

    //Sigmma=5.0/3.0;
    Sigmma=srmaxnum/3.0;
    //Sigmma=1.5;

    Sigmma2=Sigmma*Sigmma;

    for(ix=-srmaxnum;ix<=srmaxnum;ix++)
    {
        for(iy=-srmaxnum;iy<=srmaxnum;iy++)
        {
            for(iz=-srmaxnum;iz<=srmaxnum;iz++)
            {
                x=(float)ix;
                y=(float)iy;
                z=(float)iz;

                r=(x*x+y*y+z*z);

                if((r-rflag)<1e-6)
                {
                    OffsetSRx[Snine]=ix;
                    OffsetSRy[Snine]=iy;
                    OffsetSRz[Snine]=iz;

                    SR[Snine]=1.0/(2.0*PI*Sigmma2)*exp(-r/(2.0*Sigmma2));

                    Snine++;
                }
            }
        }
    }

    return(Snine);

}

void IDFTt(int mkl_parallel,int nt,complex *S1,float *x,int it)
{
    int j;
    float a,b;
    complex *S;
    S= alloc1complex(nt);
    memset(S, 0, (nt)*sizeof(complex));

    #pragma omp parallel for num_threads(mkl_parallel)
    for(int i=0;i<(nt/2);i++)
    {
        S[i].r=S1[i].r;
        S[i].i=S1[i].i;
    }

    #pragma omp parallel for num_threads(mkl_parallel)
    for(int i=0;i<NINT(nt/2);i++)
    {
        S[NINT(nt/2+i)].r=S1[NINT(nt/2-i)].r;
        S[NINT(nt/2+i)].i=-S1[NINT(nt/2-i)].i;
    }
    a=0;
    b=0;
    for(j=0;j<nt;j++)
    {
        a+=S[j].r*cos(2.0*PI/nt*j*it)-S[j].i*sin(2.0*PI/nt*j*it);
        b+=S[j].i*cos(2.0*PI/nt*j*it)+S[j].r*sin(2.0*PI/nt*j*it);
    }
    x[0]=a/nt;
    free1complex(S);
}


void IDFTt_thread(int nt,complex *S1,float *x,int it)
{
    int j;
    float a,b;
    complex *S;
    S= alloc1complex(nt);
    memset(S, 0, (nt)*sizeof(complex));

    for(int i=0;i<(nt/2);i++)
    {
        S[i].r=S1[i].r;
        S[i].i=S1[i].i;
    }

    for(int i=0;i<NINT(nt/2);i++)
    {
        S[NINT(nt/2+i)].r=S1[NINT(nt/2-i)].r;
        S[NINT(nt/2+i)].i=-S1[NINT(nt/2-i)].i;
    }
    a=0;
    b=0;
    for(j=0;j<nt;j++)
    {
        a+=S[j].r*cos(2.0*PI/nt*j*it)-S[j].i*sin(2.0*PI/nt*j*it);
        b+=S[j].i*cos(2.0*PI/nt*j*it)+S[j].r*sin(2.0*PI/nt*j*it);
    }
    x[0]=a/nt;
    free1complex(S);
}


float ricker (float t, float fpeak)
{
    float x,xx;

    x = PI*fpeak*t;
    xx = x*x;

    return exp(-xx)*(1.0-2.0*xx);
}


void IFFT_ft(int nt,complex *S,float *sif)
{
    register float *rtf;
    int i;
    /* Allocate fft arrays */
    rtf= alloc1float(nt);
    /* Main loop over traces */
/* Load trace into rt (zero-padded) */
    memset((void *) rtf,0,FSIZE*nt);

    /* IFFT (F->T) */
    pfacr(1, nt, S, rtf);
    for (i = 0; i < nt; ++i)
    {
        sif[i]=rtf[i]/nt;
    }
    free1float(rtf);
}

void FFT_tf(int nt,int nfreq,float *s, complex *S)
{
    register float *rt;    /* real trace               */
    register complex *ct;  /* complex transformed trace        */
    register float *rtf;
    int iff,it,i;
    ct= alloc1complex(nfreq);
    memset((void *) ct,0,FSIZE*nfreq);
    /* FFT (T->F) */
    pfarc(-1, nt, s, ct);
    /* output f-x domain shot  data */
    for(iff=0;iff<nfreq;++iff)
    {
        S[iff]=ct[iff];
       
    }
    free1complex(ct);
}

float Norm_2(float **A,int nx,int nz)
{
    float emax;
    int i,j,z;
    float **B;
    float *resout;
    int lwork;
    float *work;
    int info;
    info=0;
    B=alloc2float(nz,nz);
    resout=alloc1float(nz);
    lwork=3*nz-1;
    work=alloc1float(lwork);
    memset((void *) B[0], 0, sizeof(float)*nz*nz);
    for(j=0;j<nz;j++)
    {
        for(i=j;i<nz;i++)
        {
            for(z=0;z<nx;z++)
            {
                B[i][j]+=A[z][j]*A[z][i];
            }
        }
    }
    ssyev_( "N", "U", &nz, B[0],&nz, resout, work, &lwork,&info);
    emax=resout[0];
    for(i=0;i<nz;i++)
    {
        if(emax<resout[i])
        {
            emax=resout[i];
        }
    }
    /*emax=0;
    for(i=0;i<nx;i++)
    {
        for(j=0;j<nz;j++)
        {
            emax+=A[i][j]*A[i][j];
        }
    }*/
    emax=sqrt(emax);
    free2float(B);
    free1float(resout);
    free1float(work);
    return(emax);
}


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
    char *frslicepath,FILE* logfp)
{
    int iter;
    float error1;
    int ixA;
    double iStart, iElaps;
    int ix,iz;
    int d_nx;
    int d_ny;
    int is;

    char fnm[BUFSIZ];
    FILE *otfp=NULL;
    FILE *optfp=NULL;
    float *h_initial;

    int ilevel;
    int irow;
    int icol;
    int i;
    int izero;

    dcomplex *h_x0;

    float *res;
    FILE* fp;


    omega=2.0*PI*ifreq*df;
    
    warn("omega=%f",omega);

    // h_x0 = alloc1dcomplex(RankIlevel[0]);
    // res = alloc1float(RankIlevel[0]);

    h_x0 = (dcomplex *)mkl_malloc(sizeof(dcomplex) * RankIlevel[0], ALIGN);
    res =(float *)mkl_malloc(sizeof(float) * nter_max, ALIGN);

    memset(h_x0,  0, sizeof(dcomplex)*RankIlevel[0]);

    SetPML3D( nx, ny, nz, dx, dy, dz,
              vel,
              dampx, dampy, dampz,
              dampxdiff, dampydiff, dampzdiff,
              R, fpeak, omega,
              alphax, alphay, alphaz,
              alphax_diff, alphay_diff, alphaz_diff,
              alpha_max,
              deno_ex2, deno_ey2, deno_ez2,
              deno_ex3, deno_ey3, deno_ez3,
              pml_thick);

    Compute_matrix3D_sparse(shift_laplace_option, nx, ny, nz, dx, dy, dz, nxA, nzA,
                             pml_thick,
                             dampx, dampy, dampz,
                             dampxdiff, dampydiff, dampzdiff,
                             alphax, alphay, alphaz,
                             alphax_diff, alphay_diff, alphaz_diff,
                             deno_ex2, deno_ey2, deno_ez2,
                             deno_ex3, deno_ey3, deno_ez3,
                             dux, duy, duz,
                             duxx, duyy, duzz,
                             A, vel, omega, omegav_cofficient, coefficient,
                             h_Val[0], h_Row[0], h_Col[0],
                             nppw,
                             h_ValA, h_RowA, h_ColA,
                             no_zero_number);

    warn("no_zero_number="INT_PRINT_FORMAT"",no_zero_number[0]);

    ComputeMatrixR3d( Level, NxIlevel, NyIlevel, NzIlevel, RankIlevel,
     h_ValR, h_RowR, h_ColR);

    ComputeMatrixSR3d( Level, NxIlevel, NyIlevel, NzIlevel, RankIlevel,
     h_ValSR, h_RowSR, h_ColSR,
     OffsetSRx, OffsetSRy, OffsetSRz, SR, srnum);


    matrix_sort(h_ValA,h_RowA,h_ColA,RankIlevel[0],mkl_parallel);

    for(ilevel = 0;ilevel < Level-1;ilevel++)
    {
        warn("compute sort ilevel=%d",ilevel);
        matrix_sort(h_ValR[ilevel],h_RowR[ilevel],h_ColR[ilevel],RankIlevel[ilevel+1],mkl_parallel);
        matrix_sort(h_ValSR[ilevel],h_RowSR[ilevel],h_ColSR[ilevel],RankIlevel[ilevel+1],mkl_parallel);
    }



    for(ilevel = 0;ilevel < Level-1;ilevel++)
    {
        warn("compute Transpose ilevel=%d",ilevel);
        SparseMatrixTranspose(mkl_parallel,RankIlevel[ilevel+1],RankIlevel[ilevel],
                                h_ValR[ilevel], h_RowR[ilevel], h_ColR[ilevel],
                                h_ValP[ilevel], h_RowP[ilevel], h_ColP[ilevel]);
    }

    for(ilevel = 0;ilevel < Level-1;ilevel++)
    {

        warn("compute spmm ilevel=%d",ilevel);
        FunctionSparseSpmm(mkl_parallel,RankIlevel[ilevel+1],RankIlevel[ilevel],
            &h_Val[ilevel+1],&h_Col[ilevel+1],&h_Row[ilevel+1],
             h_Val[ilevel],h_Col[ilevel],h_Row[ilevel],
             h_ValR[ilevel],h_ColR[ilevel],h_RowR[ilevel],
             h_ValP[ilevel],h_ColP[ilevel],h_RowP[ilevel]);
    }
    
    /*izero=0;
    fp = fopen("43col.txt", "w");
    for (MKL_INT i = 0; i < RankIlevel[4]; ++i)
    {
        MKL_INT temp_col;
        MKL_INT icol;
        MKL_INT jcol;
        dcomplex tempt_val;

        for(icol=h_Row[4][i];icol<=h_Row[4][i+1]-1;icol++)
        {
        	fprintf(fp,"%d\t",h_Col[4][icol-1]);
        	izero++;
        }
        fprintf(fp,"\n");
     }
     fclose(fp);*/

    
    for(ilevel = 0;ilevel < Level;ilevel++)
    {
    		warn("compute sort ilevel=%d",ilevel);
        matrix_sort(h_Val[ilevel],h_Row[ilevel],h_Col[ilevel],RankIlevel[ilevel],mkl_parallel);
    }
    
    //for(ilevel = 0;ilevel < Level;ilevel++)
    {
    		ilevel=Level-1;
    		warn("compute sort ilevel=%d",ilevel);
        matrix_sort(h_Val[ilevel],h_Row[ilevel],h_Col[ilevel],RankIlevel[ilevel],mkl_parallel);
    }
    
    /*izero=0;
    fp = fopen("4col.txt", "w");
    for (MKL_INT i = 0; i < RankIlevel[4]; ++i)
    {
        MKL_INT temp_col;
        MKL_INT icol;
        MKL_INT jcol;
        dcomplex tempt_val;

        fprintf(fp,"%d\t",h_Row[4][i]);

        for(icol=h_Row[4][i];icol<=h_Row[4][i+1]-1;icol++)
        {
        	fprintf(fp,"%d\t",h_Col[4][icol-1]);
        	izero++;
        }
        fprintf(fp,"\n");
     }
     fclose(fp);*/
    
    
		warn("izero=%dh_Row[%d][8112]=%d",izero,ilevel,h_Row[4][RankIlevel[4]]);
		warn("h_Col[%d][0]=%d",ilevel,h_Col[4][0]);
    //for(is=0;is<ns;is++)
    {
        is=0;
        memset(h_b,          0, sizeof(dcomplex)*RankIlevel[0]);
        memset(h_x,          0, sizeof(dcomplex)*RankIlevel[0]);

        h_b[ixs[is]*ny*nz+iys[is]*nz+izs[is]].r=-source_fre[ifreq].r;
        h_b[ixs[is]*ny*nz+iys[is]*nz+izs[is]].i=-source_fre[ifreq].i;
        warn("ix=%d\tiy=%d\tiz=%d\t",ixs[is],iys[is],izs[is]);

        warn("index="INT_PRINT_FORMAT"",ixs[is]*ny*nz+iys[is]*nz+izs[is]);

        iStart = dsecnd();

        switch(solve_option)
        {
            case 2:
                warn("solve option BICGSTAB");
                BiCGStab_Z_MKL_Precond( mkl_parallel, nter_max, tolerant, RankIlevel[0],
                                        h_ValA, h_RowA, h_ColA,
                                        precond, nter_precond,
                                        gmres_out, fgmres_out, gmres_smoother, m_g, myid, npros, comm, nxA, Level,
                                        NxIlevel, NzIlevel, RankIlevel,
                                        h_Val, h_Row, h_Col,
                                        h_ValR, h_RowR, h_ColR,
                                        h_ValP, h_RowP, h_ColP,
                                        h_ValSR, h_RowSR, h_ColSR,
                                         h_b, h_x, res);
                break;

            default :
                warn("solve option fgmres");
                nter_real=fgmres( mkl_parallel,
                        precond, nter_precond, gmres_smoother, fgmres_out,
                        nter_max, norm2_res, tolerant, gmres_out, m_fg, m_g,
                        myid, npros, comm,
                        nxA, RankIlevel[0], Level,
                        NxIlevel, NzIlevel, RankIlevel,
                        h_ValA, h_RowA, h_ColA,
                        h_Val, h_Row, h_Col,
                        h_ValR, h_RowR, h_ColR,
                        h_ValP, h_RowP, h_ColP,
                        h_ValSR, h_RowSR, h_ColSR,
                        h_b, h_x);
                break;
        }

        

    //             fgmres( mkl_parallel,
    //  precond, nter_precond, gmres_smoother, fgmres_out,
    //  nter_max, tolerant, gmres_out, m,
    // myid, npros, comm,
    //  nxA, RankIlevel[0], Level,
    //  NxIlevel, NzIlevel, RankIlevel,
    //  h_ValA, h_RowA, h_ColA,
    //  h_Val, h_Row, h_Col,
    //  h_ValR, h_RowR, h_ColR,
    //  h_ValP, h_RowP, h_ColP,
    //  h_ValSR, h_RowSR, h_ColSR,
    //  h_b, h_x);



        // Multigrid_Solver( mkl_parallel, nter_max, tolerant, gmres_out, m,
        //                   myid, npros, comm,
        //                   nxA, Level,
        //                   NxIlevel, NzIlevel, RankIlevel,
        //                   h_Val, h_Row, h_Col,
        //                   h_ValR, h_RowR, h_ColR,
        //                   h_ValP, h_RowP, h_ColP,
        //                   h_ValSR, h_RowSR, h_ColSR,
        //                   h_b, h_x, h_x0);


        // solve_matrix(mkl_parallel,h_Val[0],h_Row[0],h_Col[0],
        //                 RankIlevel[0],h_b,h_x);



        iElaps =dsecnd() - iStart;
        warn("time=%lf\n",iElaps);

        if(myid==0)
        {
            fprintf(logfp,"ifreq=%d\tnter_real=%d\t iElaps=%lf\n",ifreq,nter_real,iElaps);
        }


    }


    mkl_free(h_x0);
    mkl_free(res);

}


/* Subrutine for output shotgahters */
void Output_shotgather3d(int mkl_parallel,int nx,int ny,int nz,int nt,int nfreq,float dt,int nfft,float ***shot,
    int is,float *xs,float *ys,float *zs,int *ixs,int *iys,int *izs,
    float fx,float fy,float dx,float dy,float dz,float hrz,
    float tdelay,
    char *frslicepath,char *shotgatherpath)
{
    int i,j,ifreq,ix,iy,iz,it,k;
    int ir;
    char fnm[BUFSIZ];
    FILE *otfp=NULL;
    char fnm1[BUFSIZ];
    FILE *hseisfp=NULL;  /* pointer to output horiz rec line file  */
    FILE *vseisfp=NULL;
    char fnm2[BUFSIZ];
    FILE *otfp2=NULL;
    
    
    float *t;
    long tracl=0;       /* trace number within a line */
    long tracr=0;       /* trace number within a reel */
    complex v;
    complex **V;
    int ihrz;
    V=alloc2complex(nx*ny,nfft);
    
    

    ihrz=NINT(hrz/dz);

    memset(V[0], 0, sizeof(complex)*nx*ny*nfft);

    #pragma omp parallel for num_threads(mkl_parallel)
    for(int ifreq=2;ifreq<nfft;ifreq++)
    {
        //warn("ifreq=%d",ifreq);
        char fnm[BUFSIZ];
        FILE *otfp=NULL;
        sprintf(fnm,"%s/frequency_slice_%d.bin",frslicepath,ifreq);
        otfp=fopen(fnm,"r");
        int ir=0;
        for(int ix=0;ix<nx;ix++)
        {
            for(int iy=0;iy<ny;iy++)
            {
                fseek(otfp,(ix*ny*nz+iy*nz+ihrz+1)*sizeof(complex),0);
                fread(&V[ifreq][ir],sizeof(complex),1,otfp);
                ir++;
            }
        }
        fclose(otfp);
    }
    warn("finish reading");
    #pragma omp parallel for num_threads(mkl_parallel)
    for(int ix=0;ix<nx;ix++)
    {
        int ir=0;
        complex *a;
        float *b;
        a=alloc1complex(nfreq);
        b=alloc1float(nt);

        for(int iy=0;iy<ny;iy++)
        {
            ir=ix*ny+iy;
            memset(a, 0, (nfreq)*sizeof(complex));
            for(int ifreq=1;ifreq<nfft;ifreq++)
            {
                a[ifreq]=V[ifreq][ir];
            }
            memset(b, 0, (nt)*sizeof(float));
            IFFT_ft(nt,a,b);
            for(int j=0;j<nt;j++)
            {
                shot[ix][iy][j]=b[j];
            }
        }
        free1complex(a);
        free1float(b);
    }

    sprintf(fnm1,"%s/hs_shot%d.out",shotgatherpath,is);
    hseisfp=fopen(fnm1,"w");

    horiztr.fldr = is;
    horiztr.sx = xs[is]*1000;
    horiztr.sy = ys[is]*1000;
    horiztr.sdepth = zs[is]*1000;
    horiztr.trid = 1;
    horiztr.ns = nt ;
    horiztr.dt = 1000000 * dt ;
    //horiztr.d2 = dx*mr ;
    horiztr.delrt = -1000.0*tdelay;

    tracl = tracr = 0;
    for (iy=0 ; iy < ny ; ++iy)
    {
        for (ix=0, tracl=0 ; ix < nx ; ++ix) 
        {
            ++tracl;
            ++tracr;

            horiztr.gx=(fx + ix*dx)*1000;
            horiztr.gy=(fy + iy*dy)*1000;

            horiztr.gelev = hrz*1000;

            horiztr.tracl = (int) tracl;
            horiztr.tracr = (int) tracr;

            for (it = 0 ; it < nt ; ++it)
            {
                horiztr.data[it] = shot[ix][iy][it];
            }
            fputtr(hseisfp , &horiztr);
        }
    }

    fclose(hseisfp);

    
}

/* Subrutine for output snapshot */

void Output_snapshot3d(int mkl_parallel,int nx,int ny,int nz,int nt,int nfreq,int it,float dt,int nfft,int is,
    char *frslicepath,char *slicepath)
{
    int i,j,k,ifreq;
    complex *fre_slice[nfft];

    float **time_slicex2d;
    
    
    char fnm1[BUFSIZ];
    FILE *otfp1=NULL;
    float ***pt;
    
    complex v;
    float t;

    complex *a;
    a=alloc1complex(nfreq);
    
    pt=alloc3float(nz,ny,nx);

    time_slicex2d=alloc2float(nz,nx);

    for(ifreq=0;ifreq<nfft;ifreq++)
    {
        fre_slice[ifreq]=(complex *)mkl_malloc(sizeof(complex)*nx*ny*nz, ALIGN);
        omp_memset_complex(mkl_parallel,fre_slice[ifreq],nx*ny*nz);
    }


    #pragma omp parallel for num_threads(mkl_parallel)
    for(int itemp=5;itemp<nfft;itemp++)
    {
        char fnm[BUFSIZ];
        FILE *otfp=NULL;

        sprintf(fnm,"%s/frequency_slice_%d.bin",frslicepath,itemp);
        //warn("%s",fnm);
        otfp=fopen(fnm,"r");
        fread(fre_slice[itemp],sizeof(complex),nx*ny*nz,otfp);
        fclose(otfp);
    }


    // for(i=0;i<nx;i++)
    // {
    //     for(j=0;j<ny;j++)
    //     {
    //         for(k=0;k<nz;k++)
    //         {
    //             memset(a, 0, (nfreq)*sizeof(complex));
    //             for(ifreq=1;ifreq<nfft;ifreq++)
    //             {
    //                 a[ifreq]=fre_slice[ifreq][(i*nz*ny+j*nz+k)];
    //             }
    //             IDFTt(mkl_parallel,nt,a,&pt[i][j][k],it);

    //         }
    //     }
    // }
    warn("nx=%d ny=%d nz=%d nt=%d nfft=%d",nx,ny,nz,nt,nfft);

    #pragma omp parallel for num_threads(mkl_parallel)
    for(int ix=0;ix<nx;ix++)
    {
        complex *tmpa;
        tmpa=alloc1complex(nfreq);
        for(int iy=0;iy<ny;iy++)
        {
            for(int iz=0;iz<nz;iz++)
            {
                memset(tmpa, 0, (nfreq)*sizeof(complex));
                for(int ifreq=0;ifreq<nfft;ifreq++)
                {
                    tmpa[ifreq]=fre_slice[ifreq][(ix*nz*ny+iy*nz+iz)];
                }
                IDFTt_thread(nt,tmpa,&pt[ix][iy][iz],it);
            }
        }
        free1complex(tmpa);
    }
    
    t=it*dt;

    warn("t=%f",t);
    sprintf(fnm1,"%s/snapshot_t%f.out",slicepath,t);
    otfp1=fopen(fnm1,"w");
    fwrite(pt[0][0],sizeof(float),nx*ny*nz,otfp1);
    fclose(otfp1);

    j=100;
    #pragma omp parallel for num_threads(mkl_parallel)
    for(int i=0;i<nx;i++)
    {
        for(k=0;k<nz;k++)
        {
            time_slicex2d[i][k]=pt[i][j][k];
        }
    }

    sprintf(fnm1,"%s/snapshot2d_t%f.out",slicepath,t);
    otfp1=fopen(fnm1,"w");
    fwrite(time_slicex2d[0],sizeof(float),nx*nz,otfp1);
    fclose(otfp1);

    free1complex(a);
    free3float(pt);

    free2float(time_slicex2d);
    

    for(ifreq=0;ifreq<nfft;ifreq++)
    {
        mkl_free(fre_slice[ifreq]);
        
    }
}
































