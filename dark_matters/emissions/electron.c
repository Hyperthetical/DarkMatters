#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>
#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>

using namespace std;

double boole_rule(int const n,std::vector<double> &x,std::vector<double> &y){
    double  w,result,h;
    h = (log10(x[n-1]) - log10(x[0]))/(n-1);
    result = 0.0;
    for (int i=0;i<n;i++){
        if ((i == 0) || (i == n-1)){
            w = 7.0;
        }
        else if (i%2 != 0){
            w = 32.0;
        }
        else{
            if (i%4 == 0){
                w =14.0;
            }
            else{
                w = 12.0;
            }
        }
        //printf("x %le y %le\n",x[i],y[i]);
        result += 2*h/45.0*w*x[i]*y[i]*log(10.0);
    }
    return result;
}

double boole_rule_linear(int const n,std::vector<double> &x,std::vector<double> &y){
    double  w,result,h;
    h = (x[n-1] - x[0])/(n-1);
    result = 0.0;
    for (int i=0;i<n;i++){
        if ((i == 0) || (i == n-1)){
            w = 7.0;
        }
        else if (i%2 != 0){
            w = 32.0;
        }
        else{
            if (i%4 == 0){
                w =14.0;
            }
            else{
                w = 12.0;
            }
        }
        //printf("x %le y %le\n",x[i],y[i]);
        result += 2*h/45.0*w*y[i];
    }
    return result;
}

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

double loss_function(double E, double b_av,double ne_av,double z,double uPh){
    double me = 0.511e-3;
    double eloss_tot,eloss_ic,eloss_sync,eloss_coul,eloss_brem;
    double eloss_ic_0 = 0.76e-16*uPh + 0.25e-16*pow(1+z,4);
    eloss_brem = 4.7e-16*ne_av*E*me;
    eloss_coul = 6.13e-16*ne_av*(1+log(E/ne_av)/75.0);
    eloss_sync = 0.0254e-16*pow(E*me,2)*pow(b_av,2);
    eloss_ic = eloss_ic_0*pow(E*me,2);
    if (eloss_brem < 0.0){
        eloss_brem = 0.0;
    }
    //printf("%le %le %le %le \n",E*me,0.0254e-16*pow(E*me,2)*pow(b_av,2),eloss_brem,eloss_ic*pow(E*me,2));
    eloss_tot =  eloss_ic + eloss_sync + eloss_coul + eloss_brem;
    return eloss_tot/me; //make it units of gamma s^-1 as emissivity integrals over E are all unitless too
}

double green_integrand(double rpr,double rn,double dv,double rhosq){
    return rpr/rn*(exp(-pow(rpr-rn,2)/(4.0*dv))-exp(-pow(rpr+rn,2)/(4.0*dv)))*rhosq;
}

double green_function(int const ngr,double r,double rhosq_r,double r_data_min,double rh,double dv,int diff,int num_images,gsl_spline const * rho_spline){
    double k1,rn,r_central;
    double G;
    std::vector<double> r_int(ngr); //store integration points
    std::vector<double> int_G(ngr);
    int p,i;
    if(diff == 0 || dv == 0){
        G = 1.0;
    }
    else{
        G = 0.0;
        k1 = 1.0/sqrt(4*M_PI*dv); //coefficient
        for (p=-num_images;p<num_images+1;p++){
            if (p == 0){
                rn = r;
            }
            else{
                rn = pow(-1.0,p)*r + 2.0*rh*p;
            }
            if (abs(rn) > rh){
                r_central = rh;
            }
            else if (abs(rn) < r_data_min){
                r_central = r_data_min;
            }
            else{
                r_central = abs(rn);
            }
            double rmin = r_central - 10*sqrt(dv);
            if (rmin < r_data_min) rmin = r_data_min;
            double rmax = r_central + 10*sqrt(dv);
            if (rmax > rh) rmax = rh;
            double dx = (rmax-rmin)/(ngr-1);
            gsl_interp_accel *acc_rho = gsl_interp_accel_alloc ();
            for (i=0;i<ngr;i++){
                r_int[i] = rmin + i*dx;
                if (r_int[i] > rh) r_int[i] = rh;
                //printf("%le\n",r_int[i]);
                int_G[i] = green_integrand(r_int[i],rn,dv,gsl_spline_eval(rho_spline,r_int[i],acc_rho)/rhosq_r);
                //printf("%d %le %le %le %le\n",p,dv,rn,r_int[i],int_G[i]);
            }
            G += pow(-1.0,p)*boole_rule_linear(ngr,r_int,int_G)*k1;
            //printf("%le\n",G);
        }
    }
    return G;
}

double Green(int const ngr,std::vector<double> &r_set_gr,double r,std::vector<double> &ratioRhosq,double dv,int diff,int num_images){
    double rh = r_set_gr[ngr-1];
    double k1,rn;
    double G;
    std::vector<double> r_int(ngr); //store integration points
    std::vector<double> int_G(ngr);
    int p,i;
    if(diff == 0 || dv == 0){
        G = 1.0;
    }
    else{
        G = 0.0;
        k1 = 1.0/sqrt(4*M_PI*dv); //coefficient
        for (p=-num_images;p<num_images+1;p++){
            rn = pow(-1.0,p)*r + 2.0*rh*p;
            if (p == 0) rn = r;
            for (i=0;i<ngr;i++){
                int_G[i] = green_integrand(r_set_gr[i],rn,dv,ratioRhosq[i]);
                printf("%d %le %le %le %le\n",p,dv,rn,r_set_gr[i],int_G[i]);
            }
            G += pow(-1.0,p)*boole_rule(ngr,r_set_gr,int_G)*k1;
        }
    }
    return G;
}

double dvFunc(double E,void * params){
    double me = 0.511e-3;
    std::vector<double> diffusionParams = *(std::vector<double> *)params;
    double b_av = diffusionParams[0];
    double ne_av = diffusionParams[1];
    double delta = diffusionParams[2];
    double z = diffusionParams[3];
    double uPh = diffusionParams[4];
    return pow(E*me,2.0-delta)/loss_function(E,b_av,ne_av,z,uPh);
}

std::vector<double> equilibrium_p2(int const k,int const kp,std::vector<double> &E_set,std::vector<double> &Q_set,int const n,int const ngr,std::vector<double> &r_set,std::vector<double> &rho_dm_set,std::vector<double> &b_set,std::vector<double> &n_set,double z,double mchi,double delta,int diff,double b_av,double ne_av,double d0,double uPh,double mode_exp,int num_threads,int num_images){
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
    delta is the power-law slope for the B field turbulence spectrum 5/3 Kolmogorov, 2 is Bohm
    diff is a flag, 0 -> no diffusion, 1 -> diffusion */

    double me = 0.511e-3; //GeV - electron mass
    double nwimp0 = pow(1.458e-33,0.5*mode_exp)*pow(1.0/mchi,mode_exp)/mode_exp;  //non-thermal wimp density (cm^-3) (central)
    std::vector<double> electrons(k*n);   //electron equilibrium distribution
    double v_set[k];
    double e_array[k];
    double q_array[k];
    double r_array[n];
    double rho_sq_array[n];
    std::vector<double> dvParams (5);
    double vResult, vError;
    dvParams[0] = b_av;
    dvParams[1] = ne_av;
    dvParams[2] = delta;
    dvParams[3] = z;
    dvParams[4] = uPh;
    for (int i=0;i<k;i++){
        if ((i == k-1) || (diff == 0)){
            v_set[i] = 0.0;
        }
        else{
            gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
            gsl_function F;
            F.function = &dvFunc;
            F.params = &dvParams;
            gsl_integration_qags (&F, E_set[i], E_set[k-1], 0, 1e-8,1000,w, &vResult, &vError);
            v_set[i] = vResult*d0/pow(3.086e24,2);
            gsl_integration_workspace_free (w);
        }
        q_array[i] = Q_set[i];
        e_array[i] = E_set[i];
    }
    for (int i=0;i<n;i++){
        r_array[i] = r_set[i];
        rho_sq_array[i] = pow(rho_dm_set[i],mode_exp)*nwimp0;
    }
    gsl_spline * rho_spline = gsl_spline_alloc(gsl_interp_steffen,n);
    gsl_spline_init(rho_spline,r_array,rho_sq_array,n);
    gsl_spline * v_spline = gsl_spline_alloc (gsl_interp_steffen, k);
    gsl_spline_init (v_spline, e_array, v_set, k);
    gsl_spline *q_spline = gsl_spline_alloc (gsl_interp_steffen, k);
    gsl_spline_init (q_spline, e_array, q_array, k);
    omp_set_num_threads(num_threads);
    int steps_done = 0;
    #pragma omp parallel for 
    for (int i=0;i<k;i++){ //loop over energies
        double dxp = log10(E_set[k-1]/E_set[i])/(kp-1);
        gsl_interp_accel *acc_v = gsl_interp_accel_alloc ();
        gsl_interp_accel *acc_q = gsl_interp_accel_alloc ();
        for (int j=0;j<n;j++){   //loop of r
            std::vector<double> int_E(kp,0.0);
            std::vector<double> e_prime(kp,0.0);
            for (int l=0;l<kp;l++){   //loop over primed energies
                e_prime[l] = E_set[i]*pow(10,dxp*l);
                double Gf = 1.0;
                if (e_prime[l] > E_set[k-1]) e_prime[l] = E_set[k-1];
                if ((diff == 1) && (e_prime[l] != E_set[i])){
                    double dv = v_set[i] - gsl_spline_eval(v_spline,e_prime[l],acc_v);
                    Gf = green_function(ngr,r_set[j],rho_sq_array[j],r_set[0],r_set[n-1],dv,diff,num_images,rho_spline);
                }
                int_E[l] = gsl_spline_eval(q_spline,e_prime[l],acc_q)*Gf;  //diffusion integrand
            }
            electrons[n*i +j] = boole_rule(kp,e_prime,int_E)/loss_function(E_set[i],b_set[j],n_set[j],z,uPh)*rho_sq_array[j]; 
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

std::vector<double> equilibrium_p(int const k,int const kp,std::vector<double> &E_set,std::vector<double> &Q_set,int const n,int const ngr,std::vector<double> &r_set,std::vector<double> &r_set_gr,std::vector<double> &rhosq,std::vector<double> &rhosq_gr,std::vector<double> &b_set,std::vector<double> &n_set,double z,double mchi,double delta,int diff,double b_av,double ne_av,double d0,double uPh,double mode_exp,int num_threads,int num_images){
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
    delta is the power-law slope for the B field turbulence spectrum 5/3 Kolmogorov, 2 is Bohm
    diff is a flag, 0 -> no diffusion, 1 -> diffusion */

    double me = 0.511e-3; //GeV - electron mass
    double nwimp0 = pow(1.458e-33,0.5*mode_exp)*pow(1.0/mchi,mode_exp)/mode_exp;  //non-thermal wimp density (cm^-3) (conversion factor)
    std::vector<double> electrons(k*n);   //electron equilibrium distribution
    std::vector<vector<double>> ratioRhosq(n,std::vector<double>(ngr));
    double v_set[k];
    double e_array[k];
    double q_array[k];
    double r_array[n];
    double rho_array[n];
    std::vector<double> dvParams (6);
    double vResult, vError;
    dvParams[0] = b_av;
    dvParams[1] = ne_av;
    dvParams[2] = delta;
    dvParams[3] = z;
    dvParams[4] = uPh;
    for (int i=0;i<k;i++){
        if ((i == k-1) || (diff == 0)){
            v_set[i] = 0.0;
        }
        else{
            gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
            gsl_function F;
            F.function = &dvFunc;
            F.params = &dvParams;
            gsl_integration_qags (&F, E_set[i], E_set[k-1], 0, 1e-8,1000,w, &vResult, &vError);
            v_set[i] = vResult*d0/pow(3.086e24,2);
            gsl_integration_workspace_free (w);
        }
        q_array[i] = Q_set[i];
        e_array[i] = E_set[i];
    }
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
    for (int i=0;i<n;i++){
        r_array[i] = r_set[i];
        rho_array[i] = rhosq[i]*nwimp0;
    }
    gsl_spline * v_spline = gsl_spline_alloc (gsl_interp_steffen, k);
    gsl_spline_init (v_spline, e_array, v_set, k);
    gsl_spline *q_spline = gsl_spline_alloc (gsl_interp_steffen, k);
    gsl_spline_init (q_spline, e_array, q_array, k);
    omp_set_num_threads(num_threads);
    int steps_done = 0;
    #pragma omp parallel for 
    for (int i=30;i<k;i++){ //loop over energies
        double dxp = log10(E_set[k-1]/E_set[i])/(kp-1);
        gsl_interp_accel *acc_v = gsl_interp_accel_alloc ();
        gsl_interp_accel *acc_q = gsl_interp_accel_alloc ();
        for (int j=n-7;j<n;j++){   //loop of r
            std::vector<double> int_E(kp,0.0);
            std::vector<double> e_prime(kp,0.0);
            for (int l=0;l<kp;l++){   //loop over primed energies
                e_prime[l] = E_set[i]*pow(10,dxp*l);
                double Gf = 1.0;
                if (e_prime[l] > E_set[k-1]) e_prime[l] = E_set[k-1];
                if ((diff == 1) && (e_prime[l] != E_set[i])){
                    double dv = v_set[i] - gsl_spline_eval(v_spline,e_prime[l],acc_v);
                    Gf = Green(ngr,r_set_gr,r_set[j],ratioRhosq[j],dv,diff,num_images);
                    //printf("%le %le %le %le %le\n",E_set[i],r_set[j],e_prime[l],dv,Gf);
                }
                int_E[l] = gsl_spline_eval(q_spline,e_prime[l],acc_q)*Gf;  //diffusion integrand
            }
            electrons[n*i +j] = boole_rule(kp,e_prime,int_E)/loss_function(E_set[i],b_set[j],n_set[j],z,uPh)*rhosq[j]; 
            printf("Int %le %le %le\n",E_set[i],r_set[j],electrons[i*n+j]);
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

int main(int argc, char *argv[]){
    FILE *fptr;
    int num,num_threads,num_images;
    int k,kp,n,n_gr,diff;
    double z,mchi,delta,ne_av,b_av,d0,mode_exp,uPh;

    if (argc > 1){
        fptr = fopen(argv[1],"r");

        if(fptr == NULL)
        {
            printf("Error! No input file found at: ");
            printf(argv[1]);   
            exit(1);             
        }
        fscanf(fptr,"%d %d %d %d",&k,&kp,&n,&n_gr);
        std::vector<double> r_set(n);std::vector<double> e_set(k);std::vector<double> q_set(k);std::vector<double> rho_dm(n);std::vector<double> b_set(n);std::vector<double> ne_set(n);
        for (int i = 0;i<n;i++){
            fscanf(fptr,"%lf", &r_set[i]);
        }
        for (int i = 0;i<k;i++){
            fscanf(fptr,"%lf", &e_set[i]);
        }
        for (int i = 0;i<k;i++){
            fscanf(fptr,"%le", &q_set[i]);
        }
        for (int i = 0;i<n;i++){
            fscanf(fptr,"%lf", &rho_dm[i]);
        }
        for (int i = 0;i<n;i++){
            fscanf(fptr,"%lf", &b_set[i]);
        }
        for (int i = 0;i<n;i++){
            fscanf(fptr,"%lf", &ne_set[i]);
        }
        fscanf(fptr,"%lf %lf %lf %lf %lf",&z,&mchi,&delta,&b_av,&ne_av);
        fscanf(fptr,"%d %lf %lf %lf %d %d",&diff,&uPh,&d0,&mode_exp,&num_threads,&num_images);
        fclose(fptr);
        std::vector<double> electrons = equilibrium_p2(k,kp,e_set,q_set,n,n_gr,r_set,rho_dm,b_set,ne_set,z,mchi,delta,diff,b_av,ne_av,d0,uPh,mode_exp,num_threads,num_images);
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