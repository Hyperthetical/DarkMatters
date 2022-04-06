#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>
#include <stdio.h>
#include <omp.h>
#include <iostream>

using namespace std;

double simps_3_8(int const n,std::vector<double> &x,std::vector<double> &y){
    double w,result,h;
    h = (log10(x[n-1]) - log10(x[0]))/(n-1);
    result = 0.0;
    for (int i = 0;i<n;i++){
        if ((i == 0) || (i == n-1)){
            w = 1.0;
        }
        else if (i%3 == 0){
            w = 2.0;
        }
        else{
            w = 3.0;
        }
        result += h*w*x[i]*y[i]*log(10.0)*3/8.0;
    }
    return result;
}

double simps(int const n,std::vector<double> &x,std::vector<double> &y){
    double w,result,h;
    h = (log10(x[n-1]) - log10(x[0]))/(n-1);
    result = 0.0;
    for (int i = 0;i<n;i++){
        if ((i == 0) || (i == n-1)){
            w = 1.0;
        }
        else if (i%2 == 0){
            w = 2.0;
        }
        else{
            w = 4.0;
        }
        result += h*w*x[i]*y[i]*log(10.0)/3.0;
    }
    return result;
}

double loss_function(double E, double b_av,double ne_av,double z,int ISRF){
    double me = 0.511e-3;
    double eloss_tot,eloss_ic,eloss_sync,eloss_coul,eloss_brem;
    double eloss_ic_0 = 0.25e-16*pow(1+z,4);
    if (ISRF == 1){
        eloss_ic_0 = 6.08e-16 + 0.25e-16*pow(1+z,4);
    }
    eloss_brem = 4.7e-16*ne_av*E*me;
    eloss_coul = 6.13e-16*ne_av*(1+log(E/ne_av)/75.0);
    eloss_sync = 0.0254e-16*pow(E*me,2)*pow(b_av,2);
    eloss_ic = eloss_ic_0*pow(E*me,2);
    if (eloss_brem < 0.0){
        eloss_brem = 0.0;
    }
    //printf("%le %le %le %le \n",E*me,0.0254e-16*pow(E*me,2)*pow(b_av,2),eloss_brem,eloss_ic*pow(E*me,2));
    eloss_tot =  eloss_ic + eloss_sync + eloss_coul + eloss_brem;
    return eloss_tot/me; //make it units of gamma s^-1 as integrals over E are all unitless too
}

double green_integrand(double rpr,double rn,double dv,double rhosq){
    return rpr/rn*(exp(-pow(rpr-rn,2)/(4.0*dv))-exp(-pow(rpr+rn,2)/(4.0*dv)))*rhosq;
}

double Green(int const ngr,std::vector<double> &r_set_gr,double r,std::vector<double> &ratioRhosq,double dv,int diff){
    double rh = r_set_gr[ngr-1];
    double k1,rn;
    double G;
    std::vector<double> r_int(ngr); //store integration points
    std::vector<double> int_G(ngr);
    int const images  = 51; //image charges for green function solution
    int p,i;
    std::vector<double> image_set(images);

    for (i = 0;i<images;i++){
        image_set[i] = -(images-1)*0.5 + i; 
    }
    if(diff == 0 || dv == 0){
        G = 1.0;
    }
    else{
        G = 0.0;
        k1 = 1.0/sqrt(4*M_PI*dv); //coefficient
        for (p=0;p<images;p++){
            rn = pow(-1.0,p)*r + 2.0*rh*p;
            for (i=0;i<ngr;i++){
                int_G[i] = green_integrand(r_set_gr[i],rn,dv,ratioRhosq[i]);
            }
            //G += pow(-1.0,p)*simps_3_8((log10(r_set_gr[ngr-1])-log10(r_set_gr[0]))/((ngr-1)/3.0-1),(ngr-1)/3,r_set_gr,int_G)*k1;
            G += pow(-1.0,p)*simps(ngr,r_set_gr,int_G)*k1;
        }
    }
    return G;
}

std::vector<double> equilibrium_p(int const k,std::vector<double> &E_set,std::vector<double> &Q_set,int const n,int const ngr,std::vector<double> &r_set,std::vector<double> &r_set_gr,std::vector<double> &rhosq,std::vector<double> &rhosq_gr,std::vector<double> &b_set,std::vector<double> &n_set,double z,double mchi,double lc,double delta,int diff,double b_av,double ne_av,double d0,int ISRF,double mode_exp,int num_threads){
    /*k is length of E_set and Q_set
    ngr is length of r_set_gr and rhosq_gr
    n is length of all other arrays
    E_set is an array of E/me, weights/coeffs compensate for me factors (GeV/GeV)
    Q_set is the electron generation function from chi-chi annihilation
    n is the number of spherical integration shells
    z is redshift
    mchi is mass of neutralino in GeV
    rhos is the value rho_s/rho_crit: central density over critical halo density
    rh is the assumed halo radius in Mpc
    rcore is the core radius in Mpc
    lc is the turbulent length scale for B in kpc
    delta is the power-law slope for the B field turbulence spectrum 5/3 Kolmogorov, 2 is Bohm
    diff is a flag, 0 -> no diffusion, 1 -> diffusion */

    double me = 0.511e-3; //GeV - electron mass
    double nwimp0 = pow(1.458e-33,0.5*mode_exp)*pow(1.0/mchi,mode_exp)/mode_exp;  //non-thermal wimp density (cm^-3) (central)
    std::vector<double> loss(k);
    std::vector<double> electrons(k*n);   //electron equilibrium distribution
    std::vector<vector<double>> ratioRhosq(n,std::vector<double>(ngr));

    for (int i=0;i<n;i++){ 
        rhosq[i] = nwimp0*rhosq[i];
    }
    for (int i=0;i<ngr;i++){ 
        rhosq_gr[i] = nwimp0*rhosq_gr[i];
    }
    for (int i=0;i<n;i++){ 
        for (int j=0;j<ngr;j++){ 
            ratioRhosq[i][j] = rhosq_gr[j]/rhosq[i];
        }
    }
    for (int i=0;i<k;i++){
        loss[i] = loss_function(E_set[i],b_av,ne_av,z,ISRF);  //energy loss (E)
        //printf("%le \n",E_set[i]);
    }
    //printf("%le \n",d0);
    //printf("%le \n",mode_exp);
    //printf("%le \n",d0);
    omp_set_num_threads(num_threads);
    int steps_done = 0;
    #pragma omp parallel for 
    for (int i=0;i<k;i++){ //loop over energies
        double E = E_set[i];
        for (int j=0;j<n;j++){   //loop of r
            double r = r_set[j];
            std:vector<double> int_E(k,0.0);
            for (int l=0;l<k-1;l++){   //loop over energies barring last one, compare E and E2
                double E2 = E_set[l];
                double dv = 0.0;
                if (E2 < E){ 
                    //integral runs over e values bigger than target energy E2
                    int_E[l] = 0.0;
                }
                else{
                    if (diff == 1){
                        std::vector<double> int_v1(k,0.0);
                        std::vector<double> int_v2(k,0.0);
                        for (int i2=0;i2<k-1;i2++){
                            double E3 = E_set[i2];
                            if (E3 < E){
                                int_v1[i2] = 0.0;  //integral runs over E3 from lower lim E (v prime)
                            }
                            else{
                                int_v1[i2] = pow(E3*me,2.0-delta)*pow(b_av,delta-2)*pow(lc,delta-1)/loss[i2];
                            }
                            if (E3 < E2){
                                int_v2[i2] = 0.0;  //integral runs over E3 from lower lim E2 (v)
                            }
                            else{
                                int_v2[i2] = pow(E3*me,2.0-delta)*pow(b_av,delta-2)*pow(lc,delta-1)/loss[i2];
                            }
                        }
                        double v1 = simps_3_8(k,E_set,int_v1);
                        double v2 = simps_3_8(k,E_set,int_v2);
                        dv = v1-v2;   //spacial diffusion gradient
                        dv *= d0/pow(3.09e24,2.0);//*exp(r/35e-6); //Mpc^2
                        //printf("%le \n",dv);
                    }
                    else{
                        dv = 0.0;
                    }
                    double Gf = Green(ngr,r_set_gr,r,ratioRhosq[j],dv,diff);  //diffusion integrand
                    int_E[l] = Q_set[l]*Gf;  //diffusion integrand
                }
            }
            //#pragma omp atomic
            //electrons[n*i +j] = 2.0*simps_3_8((log10(E_set[k-1])-log10(E_set[0]))/((k-1)/3.0-1),(k-1)/3,E_set,int_E)/loss[i]*rhosq[j]; //the 2 is for electrons and positrons
            electrons[n*i +j] = 2.0*simps_3_8(k,E_set,int_E)/loss[i]*rhosq[j]; //the 2 is for electrons and positrons
            #pragma omp atomic
            steps_done++;
            #pragma omp critical
            printf("\rProgress: %d%%", int((steps_done+1)*1.0/(n*k)*100.0));
            fflush(stdout);
        }
    }    
    printf("\n");
    return electrons;              
}

/*input file structure?
try a file with rows of arrays
line 0 is k,n,n_gr
line 1 is r
line 2 is r_gr
line 3 is e_set
line 4 is q_set
line 5 is rhosq
line 6 is rhosq_gr
line 7 is b_set
line 8 is ne_set
line 9 is z,mchi,lc,delta,ne_av,b_av
line 10 is diff, ISRF, d0, mode_exp
*/
int main(int argc, char *argv[]){
    FILE *fptr;
    int num,num_threads;
    int lines = 4;
    int entries_per_line = 1;
    int k,n,n_gr,diff,ISRF;
    double z,mchi,lc,delta,ne_av,b_av,d0,mode_exp;

    if (argc > 1){
        fptr = fopen(argv[1],"r");

        if(fptr == NULL)
        {
            printf("Error! No input file found at: ");
            printf(argv[1]);   
            exit(1);             
        }
        //read line 0 then assign
        fscanf(fptr,"%d %d %d",&k,&n,&n_gr);
        //printf("%d %d %d \n",k,n,n_gr);
        std::vector<double> r_set(n);std::vector<double> r_set_gr(n_gr);std::vector<double> e_set(k);std::vector<double> q_set(k);std::vector<double> rhosq(n);std::vector<double> rhosq_gr(n_gr);std::vector<double> b_set(n);std::vector<double> ne_set(n);
        for (int i = 0;i<n;i++){
            fscanf(fptr,"%lf", &r_set[i]);
        }
        //printf("%le %le\n",r_set[0],r_set[n-1]);
        for (int i = 0;i<n_gr;i++){
            fscanf(fptr,"%lf", &r_set_gr[i]);
        }
        for (int i = 0;i<k;i++){
            fscanf(fptr,"%lf", &e_set[i]);
        }
        //printf("%le %le\n",e_set[0],e_set[k-1]);
        for (int i = 0;i<k;i++){
            fscanf(fptr,"%le", &q_set[i]);
        }
        //printf("%le %le\n",q_set[0],q_set[k-1]);
        for (int i = 0;i<n;i++){
            fscanf(fptr,"%lf", &rhosq[i]);
        }
        for (int i = 0;i<n_gr;i++){
            fscanf(fptr,"%lf", &rhosq_gr[i]);
        }
        for (int i = 0;i<n;i++){
            fscanf(fptr,"%lf", &b_set[i]);
        }
        for (int i = 0;i<n;i++){
            fscanf(fptr,"%lf", &ne_set[i]);
        }
        //printf("%le %le \n",ne_set[0],ne_set[n-1]);
        fscanf(fptr,"%lf %lf %lf %lf %lf %lf",&z,&mchi,&lc,&delta,&b_av,&ne_av);
        fscanf(fptr,"%d %d %lf %lf %d",&diff,&ISRF,&d0,&mode_exp,&num_threads);
        fclose(fptr);
        //printf("%le %le %le %le %le %le \n",z,mchi,lc,delta,ne_av,b_av);
        //printf("%le \n",d0);
        std::vector<double> electrons = equilibrium_p(k,e_set,q_set,n,n_gr,r_set,r_set_gr,rhosq,rhosq_gr,b_set,ne_set,z,mchi,lc,delta,diff,b_av,ne_av,d0,ISRF,mode_exp,num_threads);
        fptr = fopen(argv[2],"w");
        for (int i = 0;i<k;i++){
            for (int j = 0;j<n;j++){
                fprintf(fptr,"%lf ",electrons[n*i + j]);
            }
        }
        fclose(fptr);
    }
    return 0;
}
/*
bn::ndarray electron_wrapper(int k,double E_set[],double Q_set[],int n,int ngr,double r_set[],double r_set_gr[],double rhosq[],double rhosq_gr[],double b_set[],double n_set[],double z,double mchi,double lc,double delta,int diff,double b_av,double ne_av){
    std::vector<double> v;
    Py_BEGIN_ALLOW_THREADS
    v = equilibrium_p(k,E_set,Q_set,n,ngr,r_set,r_set_gr,rhosq,rhosq_gr,b_set,n_set,z,mchi,lc,delta,diff,b_av,ne_av);
    Py_END_ALLOW_THREADS
    Py_intptr_t shape[1] = { v.size() };
    bn::ndarray result = bn::zeros(1,shape,bn::dtype::get_builtin<double>());
    std::copy(v.begin(),v.end(),reinterpret_cast<double*>(result.get_data())); 
    return result;
}*/

/*BOOST_PYTHON_MODULE(electrons_mod){
    bn::initialize();
    bp::def("equilibrium_p",&equilibrium_p);
}*/
