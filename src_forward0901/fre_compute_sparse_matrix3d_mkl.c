#include "fre_compute_sparse_matrix3d_mkl.h"
#include "omp_string_subfunction.h"
#include "par.h"
#include "su.h"
#include "cwp.h"
#include "segy.h"
#include "time.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
        int pml_thick)
{
    int i,j,k;
    float pmldisx,pmldisy,pmldisz,disx,disy,disz;

    complex ex;
    complex ey;
    complex ez;

    complex exx;
    complex eyy;
    complex ezz;

    complex exxx;
    complex eyyy;
    complex ezzz;

    for(i=0;i<pml_thick;i++)
    {
        for(j=0;j<ny;j++)
        {
            for(k=0;k<nz;k++)
            {
                disx = (pml_thick-i)*dx;
                dampx[i][j][k] = -(3.0*log(R)*vel[i][j][k]*disx*disx)/
                                (2*pow(pml_thick*dx,3.0));
                dampxdiff[i][j][k] = (3.0*log(R)*vel[i][j][k]*disx)
                                /pow( pml_thick*dx,3.0);
                /*cpml*/
                alphax[i][j][k]=alpha_max*(1.0-disx/(pml_thick*dx));
                alphax_diff[i][j][k]=alpha_max/(pml_thick*dx);

                /*ex*/
                ex=cadd(cdiv(cmplx(dampx[i][j][k],0.0),cmplx(alphax[i][j][k],omega)),cmplx(1.0,0.0));
                exx=cipow(ex,2);
                exxx=cipow(ex,3);
                deno_ex2[i][j][k]=cdiv(cmplx(1.0,0.0),exx);
                deno_ex3[i][j][k]=cdiv(cmplx(1.0,0.0),exxx);
            }
        }
    }
    for(i=nx-pml_thick;i<nx;i++)
    {
        for(j=0;j<ny;j++)
        {
            for(k=0;k<nz;k++)
            {
                disx=(i+1-nx+pml_thick)*dx;
                dampx[i][j][k] = -(3.0*log(R)*vel[i][j][k]*disx*disx)/
                            (2*pow(pml_thick*dx,3.0));
                dampxdiff[i][j][k]=-(3.0*log(R)*vel[i][j][k]*disx)
                            /pow(pml_thick*dx,3.0);

                /*cpml*/
                alphax[i][j][k]=alpha_max*(1.0-disx/(pml_thick*dx));
                alphax_diff[i][j][k]=-alpha_max/(pml_thick*dx);

                /*ex*/
                ex=cadd(cdiv(cmplx(dampx[i][j][k],0.0),cmplx(alphax[i][j][k],omega)),cmplx(1.0,0.0));
                exx=cipow(ex,2);
                exxx=cipow(ex,3);
                deno_ex2[i][j][k]=cdiv(cmplx(1.0,0.0),exx);
                deno_ex3[i][j][k]=cdiv(cmplx(1.0,0.0),exxx);
            }
        }
    }

    for(j=0;j<pml_thick;j++)
    {
        for(i=0;i<nx;i++)
        {
            for(k=0;k<nz;k++)
            {
                disy = (pml_thick-j)*dy;
                dampy[i][j][k] = -(3.0*log(R)*vel[i][j][k]*disy*disy)/
                                (2*pow(pml_thick*dy,3.0));
                dampydiff[i][j][k] = (3.0*log(R)*vel[i][j][k]*disy)
                                /pow( pml_thick*dy,3.0);

                /*cpml*/
                alphay[i][j][k]=alpha_max*(1.0-disy/(pml_thick*dy));
                alphay_diff[i][j][k]=alpha_max/(pml_thick*dy);

                /*ex*/
                ey=cadd(cdiv(cmplx(dampy[i][j][k],0.0),cmplx(alphay[i][j][k],omega)),cmplx(1.0,0.0));
                eyy=cipow(ey,2);
                eyyy=cipow(ey,3);
                deno_ey2[i][j][k]=cdiv(cmplx(1.0,0.0),eyy);
                deno_ey3[i][j][k]=cdiv(cmplx(1.0,0.0),eyyy);
            }
        }
    }

    for(j=ny-pml_thick;j<ny;j++)
    {
        for(i=0;i<nx;i++)
        {
            for(k=0;k<nz;k++)
            {
                disy = (j+1-ny+pml_thick)*dy;
                dampy[i][j][k] = -(3.0*log(R)*vel[i][j][k]*disy*disy)/
                                (2*pow(pml_thick*dy,3.0));
                dampydiff[i][j][k] = -(3.0*log(R)*vel[i][j][k]*disy)
                                /pow( pml_thick*dy,3.0);
                /*cpml*/
                alphay[i][j][k]=alpha_max*(1.0-disy/(pml_thick*dy));
                alphay_diff[i][j][k]=-alpha_max/(pml_thick*dy);

                /*ex*/
                ey=cadd(cdiv(cmplx(dampy[i][j][k],0.0),cmplx(alphay[i][j][k],omega)),cmplx(1.0,0.0));
                eyy=cipow(ey,2);
                eyyy=cipow(ey,3);
                deno_ey2[i][j][k]=cdiv(cmplx(1.0,0.0),eyy);
                deno_ey3[i][j][k]=cdiv(cmplx(1.0,0.0),eyyy);
            }
        }
    }

    for(k=0;k<pml_thick;k++)
    {
        for(i=0;i<nx;i++)
        {
            for(j=0;j<ny;j++)
            {
                disz = (pml_thick-k)*dz;
                dampz[i][j][k] = -(3.0*log(R)*vel[i][j][k]*disz*disz)/
                                (2*pow(pml_thick*dz,3.0));
                dampzdiff[i][j][k] = (3.0*log(R)*vel[i][j][k]*disz)
                                /pow( pml_thick*dz,3.0);

                /*cpml*/
                alphaz[i][j][k]=alpha_max*(1.0-disz/(pml_thick*dz));
                alphaz_diff[i][j][k]=alpha_max/(pml_thick*dz);

                /*ex*/
                ez=cadd(cdiv(cmplx(dampz[i][j][k],0.0),cmplx(alphaz[i][j][k],omega)),cmplx(1.0,0.0));
                ezz=cipow(ez,2);
                ezzz=cipow(ez,3);
                deno_ez2[i][j][k]=cdiv(cmplx(1.0,0.0),ezz);
                deno_ez3[i][j][k]=cdiv(cmplx(1.0,0.0),ezzz);
            }
        }
    }

    for(k=nz-pml_thick;k<nz;k++)
    {
        for(i=0;i<nx;i++)
        {
            for(j=0;j<ny;j++)
            {
                disz = (k+1-nz+pml_thick)*dz;
                dampz[i][j][k] = -(3.0*log(R)*vel[i][j][k]*disz*disz)/
                                (2*pow(pml_thick*dz,3.0));
                dampzdiff[i][j][k] =-(3.0*log(R)*vel[i][j][k]*disz)
                                /pow(pml_thick*dz,3.0);

                /*cpml*/
                alphaz[i][j][k]=alpha_max*(1.0-disz/(pml_thick*dz));
                alphaz_diff[i][j][k]=-alpha_max/(pml_thick*dz);

                /*ex*/
                ez=cadd(cdiv(cmplx(dampz[i][j][k],0.0),cmplx(alphaz[i][j][k],omega)),cmplx(1.0,0.0));
                ezz=cipow(ez,2);
                ezzz=cipow(ez,3);
                deno_ez2[i][j][k]=cdiv(cmplx(1.0,0.0),ezz);
                deno_ez3[i][j][k]=cdiv(cmplx(1.0,0.0),ezzz);
            }
        }
    }
       
}


void stencil_six_order3D(int *nine_Ax,int *nine_Ay,int *nine_Az,int ix,int iy,int iz,int nx,int ny,int nz)
{
    // memset(nine_Ax,  0, sizeof(int)*19);
    {
        nine_Ax[0]=NINT(ix-3);
        nine_Ax[1]=NINT(ix-2);
        nine_Ax[2]=NINT(ix-1);
        nine_Ax[3]=NINT(ix+1);
        nine_Ax[4]=NINT(ix+2);
        nine_Ax[5]=NINT(ix+3);
        nine_Ax[6]=NINT(ix);
        nine_Ax[7]=NINT(ix);
        nine_Ax[8]=NINT(ix);
        nine_Ax[9]=NINT(ix);
        nine_Ax[10]=NINT(ix);
        nine_Ax[11]=NINT(ix);
        nine_Ax[12]=NINT(ix);
        nine_Ax[13]=NINT(ix);
        nine_Ax[14]=NINT(ix);
        nine_Ax[15]=NINT(ix);
        nine_Ax[16]=NINT(ix);
        nine_Ax[17]=NINT(ix);
        nine_Ax[18]=NINT(ix);
    }

    // memset(nine_Ay,  0, sizeof(int)*19);
    {
        nine_Ay[0]=NINT(iy);
        nine_Ay[1]=NINT(iy);
        nine_Ay[2]=NINT(iy);
        nine_Ay[3]=NINT(iy);
        nine_Ay[4]=NINT(iy);
        nine_Ay[5]=NINT(iy);
        nine_Ay[6]=NINT(iy-3);
        nine_Ay[7]=NINT(iy-2);
        nine_Ay[8]=NINT(iy-1);
        nine_Ay[9]=NINT(iy+1);
        nine_Ay[10]=NINT(iy+2);
        nine_Ay[11]=NINT(iy+3);
        nine_Ay[12]=NINT(iy);
        nine_Ay[13]=NINT(iy);
        nine_Ay[14]=NINT(iy);
        nine_Ay[15]=NINT(iy);
        nine_Ay[16]=NINT(iy);
        nine_Ay[17]=NINT(iy);
        nine_Ay[18]=NINT(iy);
    }

    //(ix+0)*ny*nz + (iy-1)*nz + (iz-1);
    // memset(nine_Az,  0, sizeof(int)*19);
    {
        nine_Az[0]=NINT(iz);
        nine_Az[1]=NINT(iz);
        nine_Az[2]=NINT(iz);
        nine_Az[3]=NINT(iz);
        nine_Az[4]=NINT(iz);
        nine_Az[5]=NINT(iz);
        nine_Az[6]=NINT(iz);
        nine_Az[7]=NINT(iz);
        nine_Az[8]=NINT(iz);
        nine_Az[9]=NINT(iz);
        nine_Az[10]=NINT(iz);
        nine_Az[11]=NINT(iz);
        nine_Az[12]=NINT(iz-3);
        nine_Az[13]=NINT(iz-2);
        nine_Az[14]=NINT(iz-1);
        nine_Az[15]=NINT(iz+1);
        nine_Az[16]=NINT(iz+2);
        nine_Az[17]=NINT(iz+3);
        nine_Az[18]=NINT(iz);
    }
}



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
    float a,float b,float c,float d,float e,float *omegav_cofficient)
{
    float deno2_ex;
    float deno2_ey;
    float deno2_ez;
    float deno3_ex;
    float deno3_ey;
    float deno3_ez;

    complex dpx;
    complex dpy;
    complex dpz;

    complex dpAx;
    complex dpAy;
    complex dpAz;

    complex dp2x;
    complex dp2y;
    complex dp2z;

    complex dpAxx;
    complex dpAyy;
    complex dpAzz;

    int index;


    dpx=cmul(deno_ex3[ix][iy][iz],
            csub(cdiv(cmplx(alphax_diff[ix][iy][iz]*dampx[ix][iy][iz],0.0),cipow(cmplx(alphax[ix][iy][iz],omega),2)),
                 cdiv(cmplx(dampxdiff[ix][iy][iz],0.0),cmplx(alphax[ix][iy][iz],omega))));

    dpy=cmul(deno_ey3[ix][iy][iz],
            csub(cdiv(cmplx(alphay_diff[ix][iy][iz]*dampy[ix][iy][iz],0.0),cipow(cmplx(alphay[ix][iy][iz],omega),2)),
                 cdiv(cmplx(dampydiff[ix][iy][iz],0.0),cmplx(alphay[ix][iy][iz],omega))));

    dpz=cmul(deno_ez3[ix][iy][iz],
            csub(cdiv(cmplx(alphaz_diff[ix][iy][iz]*dampz[ix][iy][iz],0.0),cipow(cmplx(alphaz[ix][iy][iz],omega),2)),
                 cdiv(cmplx(dampzdiff[ix][iy][iz],0.0),cmplx(alphaz[ix][iy][iz],omega))));

    dpAx=cmul(dpx,A[ix][iy][iz]);
    dpAy=cmul(dpy,A[ix][iy][iz]);
    dpAz=cmul(dpz,A[ix][iy][iz]);
    dpAxx=cmul(deno_ex2[ix][iy][iz],A[ix][iy][iz]);
    dpAyy=cmul(deno_ey2[ix][iy][iz],A[ix][iy][iz]);
    dpAzz=cmul(deno_ez2[ix][iy][iz],A[ix][iy][iz]);

    // memset(coefficient, 0, sizeof(complex)*(19));

    for(index=0;index<19;index++)
    {
        coefficient[index].r=dpAx.r*dux[index]+dpAy.r*duy[index]+dpAz.r*duz[index]
                            +dpAxx.r*duxx[index]+dpAyy.r*duyy[index]+dpAzz.r*duzz[index]
                            +omegav_cofficient[index];

        coefficient[index].i=dpAx.i*dux[index]+dpAy.i*duy[index]+dpAz.i*duz[index]
                            +dpAxx.i*duxx[index]+dpAyy.i*duyy[index]+dpAzz.i*duzz[index];
    }

}


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
    float a,float b,float c,float d,float e,float *omegav_cofficient)
{
    float deno2_ex;
    float deno2_ey;
    float deno2_ez;
    float deno3_ex;
    float deno3_ey;
    float deno3_ez;

    // float a;

    complex dpx;
    complex dpy;
    complex dpz;

    complex dpAx;
    complex dpAy;
    complex dpAz;

    complex dp2x;
    complex dp2y;
    complex dp2z;

    complex dpAxx;
    complex dpAyy;
    complex dpAzz;

    int index;

    complex shift_laplacian;


    dpx=cmul(deno_ex3[ix][iy][iz],
            csub(cdiv(cmplx(alphax_diff[ix][iy][iz]*dampx[ix][iy][iz],0.0),cipow(cmplx(alphax[ix][iy][iz],omega),2)),
                 cdiv(cmplx(dampxdiff[ix][iy][iz],0.0),cmplx(alphax[ix][iy][iz],omega))));

    dpy=cmul(deno_ey3[ix][iy][iz],
            csub(cdiv(cmplx(alphay_diff[ix][iy][iz]*dampy[ix][iy][iz],0.0),cipow(cmplx(alphay[ix][iy][iz],omega),2)),
                 cdiv(cmplx(dampydiff[ix][iy][iz],0.0),cmplx(alphay[ix][iy][iz],omega))));

    dpz=cmul(deno_ez3[ix][iy][iz],
            csub(cdiv(cmplx(alphaz_diff[ix][iy][iz]*dampz[ix][iy][iz],0.0),cipow(cmplx(alphaz[ix][iy][iz],omega),2)),
                 cdiv(cmplx(dampzdiff[ix][iy][iz],0.0),cmplx(alphaz[ix][iy][iz],omega))));

    dpAx=cmul(dpx,A[ix][iy][iz]);
    dpAy=cmul(dpy,A[ix][iy][iz]);
    dpAz=cmul(dpz,A[ix][iy][iz]);
    dpAxx=cmul(deno_ex2[ix][iy][iz],A[ix][iy][iz]);
    dpAyy=cmul(deno_ey2[ix][iy][iz],A[ix][iy][iz]);
    dpAzz=cmul(deno_ez2[ix][iy][iz],A[ix][iy][iz]);

    // shift_laplacian=cmul(cmplx(1.0,0.5),cmplx(1.0,0.5));

    // beta2=2.0*3.1415926*a/dx*aqrt(3.1415926*3.1415926/(dx*dx)*a*a+omegav_cofficient[index]*omegav_cofficient[index])*
    // 1.0/(omegav_cofficient[index]*omegav_cofficient[index]);

    //shift_laplacian=cmplx(1.0,-0.05);
    shift_laplacian=cmplx(1.0,-0.025);
    //shift_laplacian=cmplx(1.0,-0.025/dx);//12-0.1
     //0.00158
     // shift_laplacian=cmplx(1.0,-1.0/88.25);
    // shift_laplacian=cmplx(1.0,-1.0/1122.25);

    // shift_laplacian=cmplx(1.0,-1.0/(16.0+(nppw[ix][iy][iz]-4.0)*(nppw[ix][iy][iz]-4.0)));

    // dpAx=dpx;
    // dpAy=dpy;
    // dpAz=dpz;
    // dpAxx=deno_ex2[ix][iy][iz];
    // dpAyy=deno_ey2[ix][iy][iz];
    // dpAzz=deno_ez2[ix][iy][iz];

    // memset(coefficient, 0, sizeof(complex)*(19));

    for(index=0;index<19;index++)
    {
        coefficient[index].r=dpAx.r*dux[index]+dpAy.r*duy[index]+dpAz.r*duz[index]
                            +dpAxx.r*duxx[index]+dpAyy.r*duyy[index]+dpAzz.r*duzz[index]
                            +omegav_cofficient[index];

        // coefficient[index].i=dpAx.i*dux[index]+dpAy.i*duy[index]+dpAz.i*duz[index]
        //                     +dpAxx.i*duxx[index]+dpAyy.i*duyy[index]+dpAzz.i*duzz[index]
        //                     +sqrt(omegav_cofficient[index])*shift_laplacian.i;

        coefficient[index].i=dpAx.i*dux[index]+dpAy.i*duy[index]+dpAz.i*duz[index]
                            +dpAxx.i*duxx[index]+dpAyy.i*duyy[index]+dpAzz.i*duzz[index]
                            +(omegav_cofficient[index])*shift_laplacian.i;
    }

}


void stencil_coefficient_six_order3D_shiftlaplace_new(int shift_laplace_option,complex *coefficient,
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
    float a,float b,float c,float d,float e,float *omegav_cofficient)
{
    float deno2_ex;
    float deno2_ey;
    float deno2_ez;
    float deno3_ex;
    float deno3_ey;
    float deno3_ez;

    // float a;

    float alpha1;
    float alpha2;

    float k;
    float damp;

    complex dpx;
    complex dpy;
    complex dpz;

    complex dpAx;
    complex dpAy;
    complex dpAz;

    complex dp2x;
    complex dp2y;
    complex dp2z;

    complex dpAxx;
    complex dpAyy;
    complex dpAzz;

    int index;

    complex shift_laplacian;


    dpx=cmul(deno_ex3[ix][iy][iz],
            csub(cdiv(cmplx(alphax_diff[ix][iy][iz]*dampx[ix][iy][iz],0.0),cipow(cmplx(alphax[ix][iy][iz],omega),2)),
                 cdiv(cmplx(dampxdiff[ix][iy][iz],0.0),cmplx(alphax[ix][iy][iz],omega))));

    dpy=cmul(deno_ey3[ix][iy][iz],
            csub(cdiv(cmplx(alphay_diff[ix][iy][iz]*dampy[ix][iy][iz],0.0),cipow(cmplx(alphay[ix][iy][iz],omega),2)),
                 cdiv(cmplx(dampydiff[ix][iy][iz],0.0),cmplx(alphay[ix][iy][iz],omega))));

    dpz=cmul(deno_ez3[ix][iy][iz],
            csub(cdiv(cmplx(alphaz_diff[ix][iy][iz]*dampz[ix][iy][iz],0.0),cipow(cmplx(alphaz[ix][iy][iz],omega),2)),
                 cdiv(cmplx(dampzdiff[ix][iy][iz],0.0),cmplx(alphaz[ix][iy][iz],omega))));

    dpAx=cmul(dpx,A[ix][iy][iz]);
    dpAy=cmul(dpy,A[ix][iy][iz]);
    dpAz=cmul(dpz,A[ix][iy][iz]);
    dpAxx=cmul(deno_ex2[ix][iy][iz],A[ix][iy][iz]);
    dpAyy=cmul(deno_ey2[ix][iy][iz],A[ix][iy][iz]);
    dpAzz=cmul(deno_ez2[ix][iy][iz],A[ix][iy][iz]);

    // shift_laplacian=cmul(cmplx(1.0,0.5),cmplx(1.0,0.5));

    // beta2=2.0*3.1415926*a/dx*aqrt(3.1415926*3.1415926/(dx*dx)*a*a+omegav_cofficient[index]*omegav_cofficient[index])*
    // 1.0/(omegav_cofficient[index]*omegav_cofficient[index]);
    

    // damp=0.008;
    // alpha1=1.0-1.0*nppw[ix][iy][iz]*nppw[ix][iy][iz]/(4.0)*damp*damp;
    // alpha2=damp*nppw[ix][iy][iz];

    // damp=0.004;
    // alpha1=1.0-1.0*nppw[ix][iy][iz]*nppw[ix][iy][iz]/(4.0)*damp*damp;
    // alpha2=damp*nppw[ix][iy][iz];
    damp=0.004;
    alpha1=1.0-1.0*nppw[ix][iy][iz]*nppw[ix][iy][iz]/(4.0)*damp*damp;
    alpha2=damp*nppw[ix][iy][iz];


    // damp=0.004;
    // alpha1=1.0-1.0*(nppw[ix][iy][iz]-120)*(nppw[ix][iy][iz]-120)/(4.0)*damp*damp;
    // alpha2=damp*(nppw[ix][iy][iz]-120);

    

    switch(shift_laplace_option)
    {
        case 1:
        alpha1=1.0;
        alpha2=0.5;
    }
    // alpha1=1.0;
    // alpha2=0.5;

    // alpha1=1.0;
    // alpha2=2.0*PI/dx*damp*sqrt(PI*PI/(dx*dx)*damp*damp+omegav_cofficient[18])*1.0/omegav_cofficient[18];

    //shift_laplacian=cmplx(1.0,-0.05);
    shift_laplacian=cmplx(alpha1,-1.0*alpha2);
    //shift_laplacian=cmplx(1.0,-0.025/dx);//12-0.1
     //0.00158
     // shift_laplacian=cmplx(1.0,-1.0/88.25);
    // shift_laplacian=cmplx(1.0,-1.0/1122.25);

    // shift_laplacian=cmplx(1.0,-1.0/(16.0+(nppw[ix][iy][iz]-4.0)*(nppw[ix][iy][iz]-4.0)));

    // dpAx=dpx;
    // dpAy=dpy;
    // dpAz=dpz;
    // dpAxx=deno_ex2[ix][iy][iz];
    // dpAyy=deno_ey2[ix][iy][iz];
    // dpAzz=deno_ez2[ix][iy][iz];

    // memset(coefficient, 0, sizeof(complex)*(19));

    for(index=0;index<19;index++)
    {
        coefficient[index].r=dpAx.r*dux[index]+dpAy.r*duy[index]+dpAz.r*duz[index]
                            +dpAxx.r*duxx[index]+dpAyy.r*duyy[index]+dpAzz.r*duzz[index]
                            +omegav_cofficient[index]*shift_laplacian.r;

        // coefficient[index].i=dpAx.i*dux[index]+dpAy.i*duy[index]+dpAz.i*duz[index]
        //                     +dpAxx.i*duxx[index]+dpAyy.i*duyy[index]+dpAzz.i*duzz[index]
        //                     +sqrt(omegav_cofficient[index])*shift_laplacian.i;

        coefficient[index].i=dpAx.i*dux[index]+dpAy.i*duy[index]+dpAz.i*duz[index]
                            +dpAxx.i*duxx[index]+dpAyy.i*duyy[index]+dpAzz.i*duzz[index]
                            +(omegav_cofficient[index])*shift_laplacian.i;
    }

}

void compute_du_six_order3D(int nx,int ny,int nz,
                            float dx,float dy,float dz,
                            int ix,int iy,int iz,
                            float omega,float ***vel,
                            float *dux,float *duy,float *duz,
                            float *duxx,float *duyy,float *duzz,
                            float a,float b,float c,float d,float e,
                            float *omegav_cofficient)
{
    float omega_v;
    float a1;
    float a2;
    float a3;

    float b1;
    float b2;
    float b3;

    a1=3.0/2.0;
    a2=-3.0/20.0;
    a3=1.0/90.0;

    b1=0.75;
    b2=-3.0/20.0;
    b3=1.0/60.0;


    omega_v=omega*omega/(vel[ix][iy][iz]*vel[ix][iy][iz]);

    dux[0]=-b3/dx;
    dux[1]=-b2/dx;
    dux[2]=-b1/dx;
    dux[3]=b1/dx;
    dux[4]=b2/dx;
    dux[5]=b3/dx;
    dux[6]=0.0;
    dux[7]=0.0;
    dux[8]=0.0;
    dux[9]=0.0;
    dux[10]=0.0;
    dux[11]=0.0;
    dux[12]=0.0;
    dux[13]=0.0;
    dux[14]=0.0;
    dux[15]=0.0;
    dux[16]=0.0;
    dux[17]=0.0;
    dux[18]=0.0;

    duy[0]=0.0;
    duy[1]=0.0;
    duy[2]=0.0;
    duy[3]=0.0;
    duy[4]=0.0;
    duy[5]=0.0;
    duy[6]=-b3/dy;
    duy[7]=-b2/dy;
    duy[8]=-b1/dy;
    duy[9]=b1/dy;
    duy[10]=b2/dy;
    duy[11]=b3/dy;
    duy[12]=0.0;
    duy[13]=0.0;
    duy[14]=0.0;
    duy[15]=0.0;
    duy[16]=0.0;
    duy[17]=0.0;
    duy[18]=0.0;

    duz[0]=0.0;
    duz[1]=0.0;
    duz[2]=0.0;
    duz[3]=0.0;
    duz[4]=0.0;
    duz[5]=0.0;
    duz[6]=0.0;
    duz[7]=0.0;
    duz[8]=0.0;
    duz[9]=0.0;
    duz[10]=0.0;
    duz[11]=0.0;
    duz[12]=-b3/dz;
    duz[13]=-b2/dz;
    duz[14]=-b1/dz;
    duz[15]=b1/dz;
    duz[16]=b2/dz;
    duz[17]=b3/dz;
    duz[18]=0.0;
    
    
    duxx[0]=a3/(dx*dx);
    duxx[1]=a2/(dx*dx);
    duxx[2]=a1/(dx*dx);
    duxx[3]=a1/(dx*dx);
    duxx[4]=a2/(dx*dx);
    duxx[5]=a3/(dx*dx);
    duxx[6]=0.0;
    duxx[7]=0.0;
    duxx[8]=0.0;
    duxx[9]=0.0;
    duxx[10]=0.0;
    duxx[11]=0.0;
    duxx[12]=0.0;
    duxx[13]=0.0;
    duxx[14]=0.0;
    duxx[15]=0.0;
    duxx[16]=0.0;
    duxx[17]=0.0;
    duxx[18]=-2.0*a1/(dx*dx)
    					-2.0*a2/(dx*dx)
    					-2.0*a3/(dx*dx);

    duyy[0]=0.0;
    duyy[1]=0.0;
    duyy[2]=0.0;
    duyy[3]=0.0;
    duyy[4]=0.0;
    duyy[5]=0.0;
    duyy[6]=a3/(dy*dy);
    duyy[7]=a2/(dy*dy);
    duyy[8]=a1/(dy*dy);
    duyy[9]=a1/(dy*dy);
    duyy[10]=a2/(dy*dy);
    duyy[11]=a3/(dy*dy);
    duyy[12]=0.0;
    duyy[13]=0.0;
    duyy[14]=0.0;
    duyy[15]=0.0;
    duyy[16]=0.0;
    duyy[17]=0.0;
    duyy[18]=-2.0*a1/(dy*dy)
    					-2.0*a2/(dy*dy)
    					-2.0*a3/(dy*dy);

    duzz[0]=0.0;
    duzz[1]=0.0;
    duzz[2]=0.0;
    duzz[3]=0.0;
    duzz[4]=0.0;
    duzz[5]=0.0;
    duzz[6]=0.0;
    duzz[7]=0.0;
    duzz[8]=0.0;
    duzz[9]=0.0;
    duzz[10]=0.0;
    duzz[11]=0.0;
    duzz[12]=a3/(dz*dz);
    duzz[13]=a2/(dz*dz);
    duzz[14]=a1/(dz*dz);
    duzz[15]=a1/(dz*dz);
    duzz[16]=a2/(dz*dz);
    duzz[17]=a3/(dz*dz);
    duzz[18]=-2.0*a1/(dz*dz)
    					-2.0*a2/(dz*dz)
    					-2.0*a3/(dz*dz);


    omegav_cofficient[0]=0.0;
    omegav_cofficient[1]=0.0;
    omegav_cofficient[2]=0.0;
    omegav_cofficient[3]=0.0;
    omegav_cofficient[4]=0.0;
    omegav_cofficient[5]=0.0;
    omegav_cofficient[6]=0.0;
    omegav_cofficient[7]=0.0;
    omegav_cofficient[8]=0.0;
    omegav_cofficient[9]=0.0;
    omegav_cofficient[10]=0.0;
    omegav_cofficient[11]=0.0;
    omegav_cofficient[12]=0.0;
    omegav_cofficient[13]=0.0;
    omegav_cofficient[14]=0.0;
    omegav_cofficient[15]=0.0;
    omegav_cofficient[16]=0.0;
    omegav_cofficient[17]=0.0;
    omegav_cofficient[18]=omega_v;
    //omegav_cofficient[18]=0.0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
}





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
    MKL_INT *no_zero_number)
{
    int ix,iy,iz;
    int ixx,iyy,izz;
    int ixA;
    float a;
    float b;
    float c;
    float d;
    float e;
    int *six_Ax;
    int *six_Ay;
    int *six_Az;
    complex **A_m;
    complex *A_row;

    char fnm[BUFSIZ];
    FILE *otfp=NULL;

    char fnm1[BUFSIZ];
    FILE *otfp1=NULL;

    six_Ax=alloc1int(nxA);
    six_Ay=alloc1int(nxA);
    six_Az=alloc1int(nxA);

    a=1.0;
    b=0.0;
    c=0.6248;
    d=0.09381;
    e=-0.00001;

    no_zero_number[0]=0;
    h_CsrRowPtrA[0]=1;

    for(ix=0;ix<nx;ix++)
    {
        for(iy=0;iy<ny;iy++)
        {
            for(iz=0;iz<nz;iz++)
            {
                stencil_six_order3D(six_Ax, six_Ay, six_Az, ix, iy, iz, nx, ny, nz);

                compute_du_six_order3D( nx, ny, nz,
                                        dx, dy, dz,
                                        ix, iy, iz,
                                        omega, vel,
                                        dux, duy, duz,
                                        duxx, duyy, duzz,
                                        a, b, c, d, e,
                                        omegav_cofficient);

                // stencil_coefficient_six_order3D_shiftlaplace( coefficient,
                //                                 dampx, dampy, dampz,
                //                                 dampxdiff, dampydiff, dampzdiff,
                //                                 ix, iy, iz, dx,
                //                                 alphax, alphay, alphaz,
                //                                 alphax_diff, alphay_diff, alphaz_diff,
                //                                 deno_ex2, deno_ey2, deno_ez2,
                //                                 deno_ex3, deno_ey3, deno_ez3,
                //                                 A, omega, nppw,
                //                                 dux, duy, duz,
                //                                 duxx, duyy, duzz,
                //                                 a, b, c, d, e, omegav_cofficient);

                stencil_coefficient_six_order3D_shiftlaplace_new(shift_laplace_option, coefficient,
                                                dampx, dampy, dampz,
                                                dampxdiff, dampydiff, dampzdiff,
                                                ix, iy, iz, dx,
                                                alphax, alphay, alphaz,
                                                alphax_diff, alphay_diff, alphaz_diff,
                                                deno_ex2, deno_ey2, deno_ez2,
                                                deno_ex3, deno_ey3, deno_ez3,
                                                A, omega, nppw,
                                                dux, duy, duz,
                                                duxx, duyy, duzz,
                                                a, b, c, d, e, omegav_cofficient);

                for(ixA=0;ixA<nxA;ixA++)
                {
                    if((six_Ax[ixA]>=0)&&
                        (six_Ax[ixA]<nx)&&
                        (six_Ay[ixA]>=0)&&
                        (six_Ay[ixA]<ny)&&
                        (six_Az[ixA]>=0)&&
                        (six_Az[ixA]<nz))
                    {
                        h_CsrValA[no_zero_number[0]].r=(double)coefficient[ixA].r;
                        h_CsrValA[no_zero_number[0]].i=(double)coefficient[ixA].i;
                        h_CsrColIndA[no_zero_number[0]]=((six_Ax[ixA])*ny*nz+six_Ay[ixA]*nz+six_Az[ixA]+1);
                        no_zero_number[0]++;

                    }
                
                }
                h_CsrRowPtrA[((ix)*ny*nz+iy*nz+iz)+1]=no_zero_number[0]+1;
            }
        }
        
    }


    no_zero_number[0]=0;
    h_RowA[0]=1;

    for(ix=0;ix<nx;ix++)
    {
        for(iy=0;iy<ny;iy++)
        {
            for(iz=0;iz<nz;iz++)
            {
                stencil_six_order3D(six_Ax, six_Ay, six_Az, ix, iy, iz, nx, ny, nz);

                compute_du_six_order3D( nx, ny, nz,
                                        dx, dy, dz,
                                        ix, iy, iz,
                                        omega, vel,
                                        dux, duy, duz,
                                        duxx, duyy, duzz,
                                        a, b, c, d, e,
                                        omegav_cofficient);

                stencil_coefficient_six_order3D( coefficient,
                                                dampx, dampy, dampz,
                                                dampxdiff, dampydiff, dampzdiff,
                                                ix, iy, iz,
                                                alphax, alphay, alphaz,
                                                alphax_diff, alphay_diff, alphaz_diff,
                                                deno_ex2, deno_ey2, deno_ez2,
                                                deno_ex3, deno_ey3, deno_ez3,
                                                A, omega,
                                                dux, duy, duz,
                                                duxx, duyy, duzz,
                                                a, b, c, d, e, omegav_cofficient);

                for(ixA=0;ixA<nxA;ixA++)
                {
                    if((six_Ax[ixA]>=0)&&
                        (six_Ax[ixA]<nx)&&
                        (six_Ay[ixA]>=0)&&
                        (six_Ay[ixA]<ny)&&
                        (six_Az[ixA]>=0)&&
                        (six_Az[ixA]<nz))
                    {
                        h_ValA[no_zero_number[0]].r=(double)coefficient[ixA].r;
                        h_ValA[no_zero_number[0]].i=(double)coefficient[ixA].i;
                        h_ColA[no_zero_number[0]]=((six_Ax[ixA])*ny*nz+six_Ay[ixA]*nz+six_Az[ixA]+1);
                        no_zero_number[0]++;

                    }
                
                }
                h_RowA[((ix)*ny*nz+iy*nz+iz)+1]=no_zero_number[0]+1;
            }
        }
        
    }



    free1int(six_Ax);
    free1int(six_Ay);
    free1int(six_Az);

}


void StencilR3d(float *R)
{
    int add;
    
    add=0;
    R[add+0]=1.0/64.0;
    R[add+1]=1.0/32.0;
    R[add+2]=1.0/64.0;

    R[add+3]=1.0/32.0;
    R[add+4]=1.0/16.0;
    R[add+5]=1.0/32.0;

    R[add+6]=1.0/64.0;
    R[add+7]=1.0/32.0;
    R[add+8]=1.0/64.0;


    add=9;
    R[add+0]=1.0/32.0;
    R[add+1]=1.0/16.0;
    R[add+2]=1.0/32.0;

    R[add+3]=1.0/16.0;
    R[add+4]=1.0/8.0;
    R[add+5]=1.0/16.0;

    R[add+6]=1.0/32.0;
    R[add+7]=1.0/16.0;
    R[add+8]=1.0/32.0;


    add=18;
    R[add+0]=1.0/64.0;
    R[add+1]=1.0/32.0;
    R[add+2]=1.0/64.0;

    R[add+3]=1.0/32.0;
    R[add+4]=1.0/16.0;
    R[add+5]=1.0/32.0;

    R[add+6]=1.0/64.0;
    R[add+7]=1.0/32.0;
    R[add+8]=1.0/64.0;


}
void StencilOffsetR3d(int *OffsetRx,int *OffsetRy,int *OffsetRz)
{
    int ix;
    int iy;
    int iz;
    int ixyz;

    ixyz=0;
    for(ix=-1;ix<=1;ix++)
    {
        for(iy=-1;iy<=1;iy++)
        {
            for(iz=-1;iz<=1;iz++)
            {
                OffsetRx[ixyz]=ix;
                OffsetRy[ixyz]=iy;
                OffsetRz[ixyz]=iz;
                ixyz++;
            }
        }
    }
}


void StencilindexR3d(int *Rx,int *Ry, int *Rz,int number,
    int ix,int iy,int iz,
    int *OffsetRx,int *OffsetRy,int *OffsetRz)
{
    int ixyz;

    for(ixyz=0;ixyz<number;ixyz++)
    {
        Rx[ixyz]=ix+OffsetRx[ixyz];
        Ry[ixyz]=iy+OffsetRy[ixyz];
        Rz[ixyz]=iz+OffsetRz[ixyz];
    }
    
}

void ComputeMatrixR3d(int Level,int *NxIlevel,int *NyIlevel,int *NzIlevel,MKL_INT *RankIlevel,
    dcomplex **h_CsrValR,MKL_INT **h_CsrRowR,MKL_INT **h_CsrColR)
{
    int ilevel;
    int nxc;
    int nzc;
    int nxf;
    int nzf;

    MKL_INT NoZeroNum;
    int ix,iy,iz;
    int ixx,iyy,izz;
    int ixA;

    int *Rx;
    int *Ry;
    int *Rz;
    int *OffsetRx;
    int *OffsetRy;
    int *OffsetRz;

    float *R;

    char fnm[BUFSIZ];
    FILE *otfp=NULL;

    char fnm1[BUFSIZ];
    FILE *otfp1=NULL;

    

    Rx=alloc1int(27);
    Ry=alloc1int(27);
    Rz=alloc1int(27);

    OffsetRx=alloc1int(27);
    OffsetRy=alloc1int(27);
    OffsetRz=alloc1int(27);

    R=alloc1float(27);


    StencilOffsetR3d(OffsetRx,OffsetRy,OffsetRz);
    StencilR3d(R);

    for(ilevel=0;ilevel<Level-1;ilevel++)
    {
        NoZeroNum = 0;
        h_CsrRowR[ilevel][0] = 1;

        for(ix=0;ix<NxIlevel[ilevel+1];ix++)
        {
            for(iy=0;iy<NyIlevel[ilevel+1];iy++)
            {
                for(iz=0;iz<NzIlevel[ilevel+1];iz++)
                {
                    ixx = 2 * ix ;
                    iyy = 2 * iy ;
                    izz = 2 * iz ;

                    StencilindexR3d(Rx, Ry, Rz, 27,
                                    ixx, iyy, izz,
                                    OffsetRx,OffsetRy,OffsetRz);

                    for(ixA=0;ixA<27;ixA++)
                    {
                        if((Rx[ixA]>=0)&&
                            (Rx[ixA]<NxIlevel[ilevel])&&
                            (Ry[ixA]>=0)&&
                            (Ry[ixA]<NyIlevel[ilevel])&&
                            (Rz[ixA]>=0)&&
                            (Rz[ixA]<NzIlevel[ilevel]))
                        {
                            h_CsrValR[ilevel][NoZeroNum].r = (double) R[ixA];
                            //h_CsrValR[ilevel][NoZeroNum].r = 1.0;
                            h_CsrValR[ilevel][NoZeroNum].i = 0.0; 
                            h_CsrColR[ilevel][NoZeroNum] = (Rx[ixA]*NyIlevel[ilevel]*NzIlevel[ilevel]
                                +Ry[ixA]*NzIlevel[ilevel]+Rz[ixA])+1;

                            NoZeroNum++;
                        }

                    }

                    h_CsrRowR[ilevel][((ix)*NyIlevel[ilevel+1]*NzIlevel[ilevel+1]
                        +iy*NzIlevel[ilevel+1]+iz)+1] = NoZeroNum+1;

                }
            }
        }
    }

    free1int(Rx);
    free1int(Ry);
    free1int(Rz);

    free1int(OffsetRx);
    free1int(OffsetRy);
    free1int(OffsetRz);

    free1float(R);

}




void ComputeMatrixSR3d(
    int Level,int *NxIlevel,int *NyIlevel,int *NzIlevel,MKL_INT *RankIlevel,
    dcomplex **h_CsrValR,MKL_INT **h_CsrRowR,MKL_INT **h_CsrColR,
    int *OffsetSRx,int *OffsetSRy,int *OffsetSRz,float *SR,int srnum)
{
    int ilevel;
    int nxc;
    int nzc;
    int nxf;
    int nzf;

    int NoZeroNum;
    int ix,iy,iz;
    int ixx,iyy,izz;
    int ixA;

    char fnm[BUFSIZ];
    FILE *otfp=NULL;

    char fnm1[BUFSIZ];
    FILE *otfp1=NULL;

    int *SRx;
    int *SRy;
    int *SRz;

    SRx=alloc1int(srnum);
    SRy=alloc1int(srnum);
    SRz=alloc1int(srnum);

    for(ilevel=0;ilevel<Level-1;ilevel++)
    {
        NoZeroNum = 0;
        h_CsrRowR[ilevel][0] = 1;

        for(ix=0;ix<NxIlevel[ilevel+1];ix++)
        {
            for(iy=0;iy<NyIlevel[ilevel+1];iy++)
            {
                for(iz=0;iz<NzIlevel[ilevel+1];iz++)
                {
                    ixx = 2 * ix ;
                    iyy = 2 * iy ;
                    izz = 2 * iz ;

                    StencilindexR3d( SRx, SRy, SRz, srnum,
                                    ixx, iyy, izz,
                                    OffsetSRx, OffsetSRy, OffsetSRz);
                    for(ixA=0;ixA<srnum;ixA++)
                    {
                        if((SRx[ixA]>=0)&&
                            (SRx[ixA]<NxIlevel[ilevel])&&
                            (SRy[ixA]>=0)&&
                            (SRy[ixA]<NyIlevel[ilevel])&&
                            (SRz[ixA]>=0)&&
                            (SRz[ixA]<NzIlevel[ilevel]))
                        {
                            h_CsrValR[ilevel][NoZeroNum].r = (double) SR[ixA];
                            h_CsrValR[ilevel][NoZeroNum].i = 0.0; 
                            h_CsrColR[ilevel][NoZeroNum] = (SRx[ixA]*NyIlevel[ilevel]*NzIlevel[ilevel]
                                +SRy[ixA]*NzIlevel[ilevel]+SRz[ixA])+1;

                            NoZeroNum++;
                        }

                    }

                    h_CsrRowR[ilevel][((ix)*NyIlevel[ilevel+1]*NzIlevel[ilevel+1]
                        +iy*NzIlevel[ilevel+1]+iz)+1] = NoZeroNum+1;

                }
            }
        }
    }

    free1int(SRx);
    free1int(SRy);
    free1int(SRz);
}

















