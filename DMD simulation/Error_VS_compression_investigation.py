# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:37:22 2020

@author: yluea
"""

import numpy as np;
import matplotlib.pyplot as plt;
import matplotlib.image as mpimg;
#from scipy.fftpack import dct, idct;        
#from skimage import data, color;
#from skimage.transform import rescale, resize, downscale_local_mean;
from PMT_compressed_sensing_simu import Compressed_sensing_PMT as CSP;

#import fitz;
#import os;
#import datetime;


#import YZ_save_results;
from YZ_save_results import YZ_save_results as SR;



r = 100;
c = 100;

#cycles = [1000]
error = [];

a = CSP();
a.Load_img("ECE.jpg",RGB = True, size = (r,c), anti_aliasing = False);  


def Investigate_Compression_Impact(ratio = False, mode = 'Basis'):
    error = [];
    if not np.all(ratio):
        ratio = np.array(range(10))+1;
    cycles = ratio**2;
    cycles = cycles/np.max(cycles);
    
    YZ_SR = SR();
    
    YZ_SR.Pdf_init();
    
    YZ_SR.Make_preface();
    
    for i in cycles:
        
        condition  = "Ratio = " + str(i) ;
        print(condition);
        
        n_measurements = int(i*r*c);
        if n_measurements < 1:
            n_measurements = 1;
            
        a.Make_DMD_basis(n_measurements, scan_mode= mode);
        measure = a.PMT_measure_simu(100,n_measurements, Poisson = True);        
        a.PMT_reconstruct(r,c,measure, user_evaluation = False, learning_rate = 2e-3,
                          regularization = 1e-5, plot_step= 5000, max_epoch = 100000);
        
        YZ_SR.Load_img_mtx(a.img_reconstructed, condition);
        YZ_SR.Plot_water_mark();
        
        E = np.mean((a.img_reconstructed - a.img)**2)
        error.append(E);
        print("The Mean Square Error of reconstruction is", E);
        
    plt.figure();
    plt.plot(cycles, np.array(error))
    plt.xlabel("n_measurements / pixels");
    plt.ylabel("Mean Square Error");
    plt.savefig("Loss.jpg");
    plt.show();
    
    YZ_SR.Pdf_write();
    
    
def Investigate_Intensity_Impact(log_photons = False):
    error = [];
    if not np.all(log_photons):
        log_photons = np.array(range(1,5));
    photons = np.power(10, log_photons.astype(float));
    
    YZ_SR = SR();
    
    YZ_SR.Pdf_init();
    
    YZ_SR.Make_preface();
    
    for i in photons:
        condition  = "Photons/(pixel*mask) = " + str(i);
        print(condition);
        n_measurements = int(1*r*c);
        ## it is fully reconstruction, so no regularization
        measure = a.PMT_measure_simu(i,n_measurements, Poisson = True, Gaussian = False);        
        a.PMT_reconstruct(r,c,measure, user_evaluation = False, learning_rate = 2e-3,regularization = 0.0, plot_step= 5000, max_epoch = 100000);
        
        YZ_SR.Load_img_mtx(a.img_reconstructed, condition);
        YZ_SR.Plot_water_mark();
        
        E = np.mean((a.img_reconstructed - a.img)**2)
        error.append(E);
        print("The Mean Square Error of reconstruction is", E);
        
    
        
    plt.figure();
    plt.plot(photons, np.array(error))
    plt.xlabel("photons/pixel/measurement");
    plt.ylabel("Mean Square Error");
    plt.savefig("Loss.jpg");
    plt.show();
    
    YZ_SR.Pdf_write();
    
    
def Compression_with_same_light_level(total_photons = 1e8, ratio = False, mode = 'Random'):
    #r = 50;
    #c = 50;
    
    #a.Load_img("ECE.jpg",RGB = True, size = (r,c), anti_aliasing = False);  
    
    
    error = [];
    if not np.all(ratio):
        ratio = np.array(range(10))+1;
    cycles = ratio**2;
    cycles = cycles/np.max(cycles);
    
    YZ_SR = SR();  
    
    
    YZ_SR.Pdf_init();
    
    YZ_SR.Make_preface();
    
    for i in cycles:
        n_measurements = int(i*r*c);
        if n_measurements < 1:
            n_measurements = 1;
            
        photon_num = total_photons / n_measurements / (r*c) * 2; ## roughly half chips open;
        
        condition  = "Ratio = " + str(i) ;
        print(condition);
        #print("Ratio",i)
        
        a.Make_DMD_basis(n_measurements, scan_mode= mode);
        measure = a.PMT_measure_simu(photon_num,n_measurements, Poisson = True, Gaussian = False);        
        a.PMT_reconstruct(r,c,measure, user_evaluation = False, learning_rate = 2e-3,regularization = 1e-5, plot_step= 5000, max_epoch = 100000);
        
        YZ_SR.Load_img_mtx(a.img_reconstructed, condition);
        YZ_SR.Plot_water_mark();
        
        E = np.mean((a.img_reconstructed - a.img)**2);
        error.append(E);
        print("The Mean Square Error of reconstruction is", E);
        
    plt.figure();
    plt.plot(cycles, np.array(error))
    plt.xlabel("n_measurements / pixels");
    plt.ylabel("Mean Square Error");
    plt.savefig("Loss.jpg");
    plt.show();
    
    YZ_SR.Pdf_write(); 
    
    
def Non_compressed_inverse_method(total_photons = [1e6, 1e7, 1e8]):
    filename = "Non_compress_inv.pdf";
    
    r = 100;
    c = 100;   ## initialize the size (just for confirm)
    
    
    YZ_SR = SR();  ## YZ_save _result as PDF
    
    
    YZ_SR.Pdf_init(filename);
    
    noise_loop = [[True, False],[False, 'Poisson'],[True, 'Poisson']];
    ## I will try three noise combinations
    ## 1. only poisson 2. only gaussian 3. both
    title_dic = {(True,False):"Poisson (direct inverse and non compressed)",
                 (False,'Poisson'):"Gaussian_simu_Poisson (direct inverse and non compressed)",
                 (True,'Poisson'):"Poisson + Gaussian_simu_Poisson (direct inverse and non compressed)"}
    
    measure = []; ## record all the measured data
    for noise in noise_loop:  
        ## in this loop I will simulation measured results under different light level
        ## and save them in 'measure' (as elements in list, like a pointer)
        
        print("Measuringing...");
        
        n_measurements = r * c; ## number of measurements; if value = r*c, non compressive
        a.Make_DMD_basis(n_measurements);
        
        for light_level in total_photons:
            photon_num = light_level / (r*c) / n_measurements * 2;            
            #measure.append(a.PMT_measure_simu(photon_num, n_measurements, Poisson = noise[0], Gaussian = noise[1]));
            measure.append(a.PMT_measure_simu(photon_num, False, Poisson = noise[0], Gaussian = noise[1],upload_DMD_basis=a.DMD_basis));
    print("Measurement ended");     
    
    a.Get_direct_inverse(r,c); # get the direct inverse matrix self.direct_inverse_M
    
    task = 0; ##index the measurements
    for noise in noise_loop:
        error = []; ## record the error
        
        title = title_dic[tuple(noise)];
        
        YZ_SR.Make_preface(filename, title);    
        
        for light_level in total_photons:   
            meas = measure[task];
            a.PMT_direct_inverse_reconstruction(r, c, a.direct_inverse_M, meas);  
            
            condition  = "Light_level = " + '%.4e'%light_level;
            
            YZ_SR.Load_img_mtx(a.img_reconstructed, condition);
            YZ_SR.Plot_water_mark();
            
            E = np.mean((a.img_reconstructed - a.img)**2);
            error.append(E);
            print('Calculation %d is finished'%task);
            task += 1;
        print("Loop end");
        
        plt.figure();
        plt.scatter(np.log10(total_photons), np.array(error));
        plt.xlabel("log light level");
        plt.ylabel("Mean Square Error");
        plt.savefig("Loss.jpg");
        plt.show();
        
        YZ_SR.Pdf_write();
        

    
def Non_compressed_scan(total_photons = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8], mode = 'Raster', compress = False):
    ## for non
    if compress == False:
        filename = "Non_CS_"+mode+"_scan.pdf";
        #filename = 'test';
    else:
        filename = "CS_"+mode+"_scan.pdf";
        
    r = 100;
    c = 100;   ## initialize the size (just for confirm)
    
    
    YZ_SR = SR();  ## YZ_save _result as PDF
    
    
    YZ_SR.Pdf_init(filename);
    
    noise_loop = [[True, False],[False, 'Poisson'],[True, 'Poisson']];
    ## I will try three noise combinations
    ## 1. only poisson 2. only gaussian 3. both
    title_dic = {(True,False):"Poisson (direct inverse and non compressed)",
                 (False,'Poisson'):"Gaussian_simu_Poisson (direct inverse and non compressed)",
                 (True,'Poisson'):"Poisson + Gaussian_simu_Poisson (direct inverse and non compressed)"}
    
    '''
    In this simulation we only care about how the light level impacts 
    DMD measurement method, so for each series we make use the same matrix
    '''
    n_measurements = r * c; ## number of measurements; if value = r*c, non compressive
        
    a.Make_DMD_basis(n_measurements, scan_mode = mode); ## use a raster/basis scan eye matrix!
    
    measure = []; ## record all the measured data
    
    multi_chip_factor = {'Raster':r*c, 'Random':2, 'Basis':2};
    for noise in noise_loop:  
        ## in this loop I will simulation measured results under different light level
        ## and save them in 'measure' (as elements in list, like a pointer)
        
        print("Measuringing...");
        
        for light_level in total_photons:
            photon_num = light_level / (r*c) / n_measurements * multi_chip_factor[mode];    
            '''In raster scan, each measurement has only 1 pixel, so
            each pixel will has photon_num = light_level / n_measurement'''
            #measure.append(a.PMT_measure_simu(photon_num, n_measurements, Poisson = noise[0], Gaussian = noise[1]));
            #print(a.DMD_basis)
            measure.append(a.PMT_measure_simu(photon_num, n_measurements, Poisson = noise[0], Gaussian = noise[1],upload_DMD_basis=a.DMD_basis));
    print("Measurement ended");  
    
    ## get inverse !!!
    
    if mode != 'Raster':
        M = np.linalg.pinv(a.DMD_basis);
    else:
        M = a.DMD_basis.T;
    
    task = 0; ## looping index of the measurements
    print("Reconstructing...");
    for noise in noise_loop:
        error = []; ## record the error
        
        title = title_dic[tuple(noise)];
        
        YZ_SR.Make_preface(filename, title);    
        
        for light_level in total_photons:   
            meas = measure[task];
            
            a.PMT_direct_inverse_reconstruction(r, c, M, meas, CS = compress);  
            
            condition  = "Light_level = " + '%.4e'%light_level;
            
            YZ_SR.Load_img_mtx(a.img_reconstructed, condition);
            YZ_SR.Plot_water_mark();
            
            E = np.mean((a.img_reconstructed - a.img)**2);
            error.append(E);
            print('Calculation %d is finished'%task);
            task += 1;
        print("Loop end");
        
        plt.figure();
        plt.plot(np.log10(total_photons), np.array(error), marker = 'x');
        plt.xlabel("log light level");
        plt.ylabel("Mean Square Error");
        plt.savefig("Loss.jpg");
        plt.show();
        
        YZ_SR.Pdf_write();  
