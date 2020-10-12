# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:51:54 2020

@author: yluea
"""


import numpy as np;
import matplotlib.pyplot as plt;
import matplotlib.image as mpimg;
from scipy.fftpack import dct, idct;        
#from skimage import data, color;
#from skimage.transform import rescale, resize, downscale_local_mean;
from skimage.transform import resize;
import sys;

class Compressed_sensing_PMT:   
    def __init__(self):
        '''
        Don't forget to do No.8 First is you want to simulate.
        '''
        pass;
    
    def YZ_process_bar(self, ratio, comments = False, overwrite = True, length = 50, theme = False):
        #bar = 'Yizhou said 欲速则不达，施主稍安勿躁: | ';
        theme_dict = {"Solid":['▒','░'],
                      'Arrow':['>','·'],
                      'Equal':['=','·'],
                      'Bubble':['0','·'],
                      'Bean':['=','·']};
        if theme not in theme_dict:
            theme = 'Arrow';
        bar = '|';
        i = 0;
        while i < ratio * length:
            #bar += '▒';
            bar += theme_dict[theme][0];
            i += 1;
        if theme == 'Bean':
            bar = bar[:(i-1)]
            bar +=  '©';
            
        while i < length:
            #bar += '░';
            bar += theme_dict[theme][1];
            i += 1;        
        bar += ('| %s%%'%(int(ratio*1000)/10));
        if ratio == 1:
            bar += ' (^_^)/ Done!'
        if comments != False:
            bar = ('{'+ str(comments) + '} ' + bar);
        if overwrite == True:
            print('\r', end='');
        else:
            print('\n',end = '');
        print(bar, end='');
        sys.stdout.flush();
            
    ## These are modified dct2 functions
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    ## No.1
    def dct2(self,block,padding = False):
        #block = np.array(block);
        if not np.all(padding):
            #return dct(dct(block.T).T);
            return dct(dct(block.T, norm='ortho').T, norm='ortho');
        else:
            [c,r] = padding;
            #return dct(dct(block.T, n = c).T, n = r);
            return dct(dct(block.T, n = c, norm='ortho').T, n = r, norm='ortho');    
    def idct2(self,block,padding = False):    
        #block = np.array(block);
        if not np.all(padding):
            
            #r0,c0 = block.shape;
            #return idct(idct(block.T).T)/r0/c0;
            #return idct(idct(block.T, norm='ortho').T, norm='ortho')/r0/c0;
            return idct(idct(block.T, norm='ortho').T, norm='ortho');
        else:
            [c,r] = padding;
            #return idct(idct(block.T, n = c).T, n = r)/(r*c);
            return idct(idct(block.T, n = c, norm='ortho').T, n = r, norm='ortho');
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ## Generate the Frequence Domain; I choose DCTMTX here. 
    
    ## No.2
    def DCT_Matrix(self, N):        
        I = np.eye(N);
        dctmtx = dct(I, axis=0, norm = 'ortho');
        return dctmtx;   
    def IDCT_Matrix(self, N):
        I = np.eye(N);
        idctmtx = idct(I, axis=0, norm = 'ortho');
        return idctmtx;  
    
    ## No.3        
    def DCT_Basis(self,r_tot,c_tot): 
        '''
        This function receives the row and rol values to generate a matrix in
        this size; this matrix is very important, not only in this projec;
        Imaging you have a 2D transform, A* =  D1 A D2', where A is a image; 
        and you reshape A* in a vector vA*; 
        
        This function generate a matrix D12; if you reshape A in a vector vA, and 
        do D12 vA is exactly vA* ;
        
        dct2 a image and reshape <-> reshape a image and transform it with this 
        matrix!
        '''
        ## This is used to translate idct2 into a matrix tansforming a img vector
        #D1 = np.linalg.pinv(self.DCT_Matrix(r_tot));
        #D2 = np.linalg.pinv(self.DCT_Matrix(c_tot));
        D1 = self.IDCT_Matrix(r_tot);
        D2 = self.IDCT_Matrix(c_tot);
        N = r_tot * c_tot;
        Basis = np.zeros((N,N));
        i = 0;
        for ir in range(r_tot): ## rows in raw img
            for ic in range(c_tot): ## cols in raw img
                Basis_Vector = D1[ir:ir+1, :].T.dot(D2[ic:ic+1, :]);
                Basis[i,:] = np.reshape(Basis_Vector, (-1));
                i += 1;
        return Basis;                
        
        
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
    ## making the DMD patterns and measurement basis    
    ## No.4
    def DMD_Freq_pattern(self, r,c, cycles = False):
        ''' Triangle version
        This function generate a matrix similar to the image's DCT2 domain,
        it will start from LEFT TOP corner (Low Freq) to RIGHT BOTTOM corner 
        (High Freq); The selected positions will be 1 while others will be 0;
        This pattern will be returned. This pattern can be read by 
        'No.6' to generate a DMD_basis
        '''
        ## From low freq to high freq; selected freq will be 1
                   
        #r,c = shape;
        if (not cycles) or (cycles > r * c):
            cycles = r * c;
            
        freq_pattern = np.zeros((r,c));
        
        n_res = cycles;
        res_index = np.where(freq_pattern == 0);
        res_index = np.array(res_index);

        ranks = res_index[0,:] / (r+1) + res_index[1,:] / (c+1);
        #ranks = self.Poisson_noise(ranks);
        index_sum = np.vstack((res_index, ranks));
        #index_sum = np.vstack((res_index, res_index[0,:] / (r+1) + res_index[1,:] / (c+1)));

        ## The last row is the weighted Manhatan Distance:
        ## The selection for the residual candidates should start from low freq to high freq
        ## The weights are set to neutralize the impacts of the rectangle shape of a picture
        ## make it look like in a square coordinate system;
        ## so (r) / (r1+1) and (c) / (r1+1)
        
        
        sorted_index = index_sum[:,index_sum[2].argsort()];
        ## sort the list by the sum of x and y index :: quicker than for loop?       
        
        #np.random.shuffle(res_index.T);
        selection = sorted_index[0:2,0:n_res].astype(int);   
        
        freq_pattern[tuple(selection)] = 1;
                  
        
        self.img_DCT_domain = np.zeros((r,c));
        ## This is for generating the measurement matrix of 0 & 1 by doing idct2
        ## Just look at the following func
        
        return freq_pattern;
    
    def DMD_Freq_pattern_rectangle(self, r,c, cycles = False):
        '''
        This function generate a matrix similar to the image's DCT2 domain,
        it will start from LEFT TOP corner (Low Freq) to RIGHT BOTTOM corner 
        (High Freq); The selected positions will be 1 while others will be 0;
        This pattern will be returned. This pattern can be read by 
        'No.6' to generate a DMD_basis
        '''
        ## From low freq to high freq; selected freq will be 1
                   
        #r,c = shape;
        if (not cycles) or (cycles > r * c):
            cycles = r * c;
            
        freq_pattern = np.zeros((r,c));
        
        ratio = (cycles/r/c)**0.5;
        r1 = int(np.ceil(ratio*r))-1;
        c1 = int(np.ceil(ratio*c))-1; ## In case ratio = 1        
        freq_pattern[:r1,:c1] = 1; ## This paint the square part of the whole pattern
        
        n_res = cycles - r1 * c1;
        res_index = np.where(freq_pattern[:r1+1, :c1+1] == 0);
        res_index = np.array(res_index);  
        
        index_sum = np.vstack((res_index, res_index[0,:] / (r1+1) + res_index[1,:] / (c1+1)));
        ## The last row is the weighted Manhatan Distance:
        ## The selection for the residual candidates should start from low freq to high freq
        ## The weights are set to neutralize the impacts of the rectangle shape of a picture
        ## make it look like in a square coordinate system;
        ## so (r) / (r1+1) and (c) / (r1+1)
        
        
        sorted_index = index_sum[:,index_sum[2].argsort()];
        ## sort the list by the sum of x and y index :: quicker than for loop?       
        
        #np.random.shuffle(res_index.T);
        selection = sorted_index[0:2,0:n_res].astype(int);   
        
        freq_pattern[tuple(selection)] = 1;
                  
        
        self.img_DCT_domain = np.zeros((r,c));
        ## This is for generating the measurement matrix of 0 & 1 by doing idct2
        ## Just look at the following func
        
        return freq_pattern;
    ## No.5
    def DMD_pattern(self,r,c, size = 'null'): ## 0,1 matrix by idct2
        #x_r = np.array(range(r_tot)) / r_tot * np.pi;
        #x_c = np.array(range(c_tot)) / c_tot * np.pi;
        #y_r = np.cos(x_r * r) * 0.5; ## to avoid -1. we only need 0 and 1
        #y_c = np.cos(x_c * c) * 0.5;
        #y_r = np.reshape(y_r, (-1,1));
        #y_c = np.reshape(y_c, (-1,1)); 
        '''
        This function initally use self.img_DCT_domain, which is declared in 
        "DMD_Freq_pattern(self, r,c, cycles = False)" as a zero matrix
        The DMD pattern method is like, you firstly make 
        self.img_DCT_domain[r,c] = 1, and make do DCT2 to generate a pattern; 
        and then make self.img_DCT_domain[r,c] = 0 to revert it to zero matrix;
        Finally, ceil this pattern, to make it  0 - 1;
        
        if another 'size' is input, this will do a padding in IDCT2
        '''
        
        if size == 'null':        
            self.img_DCT_domain[r,c] = 1;
            DCT_pattern = self.idct2(self.img_DCT_domain);
            self.img_DCT_domain[r,c] = 0;
        else:
            
            DCT = np.zeros(size);
            DCT[r,c] = 1;
            DCT_pattern = self.idct2(DCT);

        #threshold = np.max(DCT_pattern) * ratio;
        #bin_pattern = np.sign(DCT_pattern-threshold) * 0.5;
        #bin_pattern = np.ceil(bin_pattern);
        #return bin_pattern;        
        return (DCT_pattern);
    ## No.6
    def DMD_measure_basis(self, freq_pattern, max_ratio = 0): 
        '''
        This function generate a self.DMD_basis matrix, whose rows represent
        a DMD pattern but rehsaped in a row!
        When this function receives a new freq_pattern, it will find the 1 in 
        this pattern, and do a idct2 transform to each position with 1 by 
        calling No.5 (in loop, one by one, each one -> one row in self.DMD_basis); 
        
        it can receive the pattern from No.4        
        e.g. self.DMD_measure_basis(self.DMD_Freq_pattern(r_tot, c_tot, cycles))
        
        Also you can input a matrix as self.DMD_basis manually;
        '''
        ## reshape each DMD_pattern in to a row in measure matrix
        index = np.where(freq_pattern == 1); 
        index = np.array(index).T;
        r_tot, c_tot = freq_pattern.shape;
        
        basis = np.zeros((len(index), r_tot * c_tot));
        
        for i in range(len(index)):
            bin_pattern = self.DMD_pattern(index[i][0], index[i][1]);
            bin_pattern -= np.max(bin_pattern) * max_ratio; ## subtract threshold
            bin_pattern = np.ceil(np.sign(bin_pattern)*0.5);
            basis[i,:] = np.reshape(bin_pattern, (1,-1));
            #basis[i,:] = np.reshape(self.DMD_pattern(index[i][0], index[i][1]), (1,-1));
        
        self.DMD_basis = basis;
        #return basis;        
    
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++     
    ## Define noise  
    
    ## No.7 
    '''
    These two functions generate two noises; 
    Keep in mind, Poisson is not additive, and everytime you measure once, a new 
    poisson noise will appear; Gaussian noise is additive, you can get the 
    measured data first and add a Gaissuan; but gaussian noise needs a SNR, 
    signal to noise ratio, in addtion.
    '''
    def Poisson_noise(self, array):
        return np.random.poisson(array);        
        
    def Gaussian_noise(self, array, SNR = False):
        if not SNR:
            SNR = 100;  
        
        std = np.mean(array) / SNR;            
        return np.random.normal(array,std);
    
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++     
    ## for simulation     
    
    ## No.8            
    def Load_img(self, img_pathname = False, RGB = False, size = 'null', anti_aliasing = True):
        '''
        If you use this class for simulation, this function is a prequiste!!!
        So, ALWAYS do it first right after you declare this class!!!
        
        You can load a image and make a simulation; later on you can load a new 
        image, but REMEMBER to update other class instances to avoid mismatch;
        
        Users can choose manual input of the image filepath, images RGB or BW,
        image size (if size inputed, deciding if anti_aliasing is necessary)
        '''
        if not img_pathname:
            img_pathname = input("Please input the pathname of the image you want to test.\n");
        m = mpimg.imread(img_pathname);
        
        if not RGB: ## image would be saved in the form of tensor
            #m = m.mean(axis = 2);
            m_gray = m[:,:,0] * 0.299 + m[:,:,1] * 0.587 + m[:,:,2] * 0.114;
            m = m_gray;
            # for grayscale image, remember plt.imshow(m, cmap = 'gray')
        else:
            pass;
            
        if size != 'null':
            r = size[0];
            c = size[1];            
            m = resize(m, (r, c),anti_aliasing);
            
        self.img = m / np.max(m);
        if len(self.img.shape) == 3:
            plt.imshow(self.img);
            self.img_tensor = self.img;
        elif len(self.img.shape) == 2:
            plt.imshow(self.img, cmap = 'gray');
            self.img_tensor = np.array([self.img]).transpose(1,2,0);
        plt.show();
        
        self.DMD_basis = [];
        self.img_DCT_domain = [];
        #self.img_DCT_domain = np.zeros(self.img.shape);        
    ## No.9    
    def Load_photon_numbers(self, n_photons):  
        '''
        This function load the environment brightness and save as self.n_photons
        This is not commonly used.
        '''
        if type(n_photons*1.0) == float:
            self.n_photons = np.array([n_photons]);
        elif type(n_photons) == list:
            self.n_photons = np.array(n_photons);
        elif type(n_photons) == np.ndarray:
            self.n_photons = n_photons;
        else:
            print("You should input the number of photons as a number, list or array");
    ## No.10        
    def Make_DMD_basis(self, cycles, scan_mode = 'Basis', ratio = 0):
        '''
        Make a DMD measurement basis for image vector, and save as self.DMD_basis;
        This is an import step right before "PMT_measure_simu"
        
        scan_mode = 
        'Basis' : you get a self.DMD_basis(how a DMD looks like throughout the
        whole measurement) by in DCT domain from low to high freq;
        'Raster': you get a self.DMD_basis by measuring one pixel by pixel, and 
        it is a unit matrix;
        'Random': you generate a random measure self.DMD_basis with 0 and 1;
        
        NOTE: Both 'Basis' and 'Random' can generate a self.DMD_basis for CS and 
        basis-scan
        '''
        ## get the size of self.img_tensor
        #if len(self.img.shape) == 2:
            #r_tot,c_tot = self.img.shape;
        #elif len(self.img.shape) == 3: ## make RGB image a n*3 matrix
            #r_tot,c_tot = self.img[:,:,0].shape;
        ## build the basis : more method can be explore later
        
        r_tot,c_tot = self.img_tensor[:,:,0].shape;## img is saved as a tensor (:,:,1) or (:,:,3)
        
        if scan_mode == 'Basis':
            self.DMD_measure_basis(self.DMD_Freq_pattern(r_tot, c_tot, cycles), ratio); 
        elif scan_mode == 'Raster':
            self.DMD_basis = np.eye(cycles, r_tot * c_tot);
        elif scan_mode == 'Random':
            self.DMD_basis = np.round(np.random.random((cycles, r_tot * c_tot)) * (1-ratio/2));
        else:
            print("Bad scan form");
            
    ## No.11        
    def PMT_measure_simu(self, photon_number, cycles, Poisson = False, Gaussian = False,
                         upload_DMD_basis = []):
        '''
        This func return a matrix recording the measured intensity! as 
        variable 'measurement';
        
        ***Dont forget to load a image by No.8 first!!!!!!!!
        Also, No.10 is encouraged to do before simu, or you ay have to  upload
        one DMD_basis;
        
        This function conducts a simulation under certain brightness: 
        *photon_number: how many photons a pixel can receive averagely during a 
        measurement
        *cycles: how many measurements in total
        *Poisson = True: add poisson noise;
        *Gaussian = 'Poisson': add a gausian noise with mean = variance(like 
        poisson, for comparison)
        
        *upload_DMD_basis: if no uploads, it is [] and this func will use
        self.DMD_basis (which is initialized in No.8 as [];); if no 
        self.DMD_basis is calculated, it will remain []; in this case, a new
        'Basis' mode self.DMD_basis will be calculated; 
        
        IF cycles != len(self.DMD_basis), cycles will be converted to that.
        '''
        
        #if len(self.img.shape) == 2:
            #r_tot,c_tot = self.img.shape;
            #img_vector = np.zeros((r_tot*c_tot,1))
            #img_vector[:,0] = np.reshape(self.img, (r_tot*c_tot));## make img a vector
            
        #elif len(self.img.shape) == 3: ## make RGB image a n*3 matrix
            #r_tot,c_tot = self.img[:,:,0].shape;
            #img_vector = np.zeros((r_tot*c_tot,3))
            
            #for i in range(3):
                #img_vector[:,i] = (np.reshape(self.img[:,:,i], (r_tot*c_tot)));
        #else:
            #print("Wrong image shape");
            #return;
            
        r_tot,c_tot, layers = self.img_tensor.shape;
        img_vector = np.zeros((r_tot*c_tot, layers));
        for i in range(layers):
            img_vector[:,i] = (np.reshape(self.img_tensor[:,:,i], (r_tot*c_tot)));
        
        img_vector *= photon_number; ## how bright the environment is
        
        if (upload_DMD_basis == []) and (self.DMD_basis == []):
            ## no upload, no self.DMD_basis, then generate a Basis scan self.DMD_basis
            print("No upload or self.DMD_basis, so generate a Basis scan matrix");
            self.Make_DMD_basis(cycles, scan_mode = 'Basis');  
            ## That function has no return:: it make the matrix of self.DMD_basis, which means
            ## cycles x measurements in vector form
            
        elif type(upload_DMD_basis) == np.ndarray:
            #print("Using this uploaded matrix...");
            ## with upload, and with proper variable kind: self.DMD_basis = upload
            self.DMD_basis = upload_DMD_basis;
               
        if cycles != len(self.DMD_basis):
            print("Reminder: The input number of meausurements is wrong, but have been corrected!");
            cycles = len(self.DMD_basis);
            
        measurement = np.zeros((cycles, len(img_vector[0,:]))); ## record the measured intensity
        
        for layer in range(len(img_vector[0,:])):
            measurement_layer = self.DMD_basis.dot(img_vector[:,layer]);
            measurement[:,layer] = measurement_layer;
            
            ''''
            measurement_layer = [];
            for i in range(len(self.DMD_basis)):   
                
                val = self.DMD_basis[i,:].dot(img_vector[:,layer]);
                if Poisson:
                    val = self.Poisson_noise(val);
                
                #if Poisson:(that can used for negative entries, not physically meaningful)
                    #val = self.DMD_basis[i,:].dot(self.Poisson_noise(img_vector[:,layer]));
                #else:
                    #val = self.DMD_basis[i,:].dot(img_vector[:,layer]);
                
                measurement_layer.append(val);
            measurement_layer = np.array(measurement_layer);   
            
            measurement_layer = self.DMD_basis.dot(img_vector[:,layer]);
            measurement[:,layer] = measurement_layer;
            '''
        if Poisson == True:
            measurement = self.Poisson_noise(measurement).astype(float);
            
        if Gaussian != False:## Gaussian noise from current, so added at the final
            if Gaussian == 'Poisson': ## Use Gaussian to simulate Poisson
                SNR = (np.mean(measurement))**0.5;
                measurement = ( self.Gaussian_noise(measurement, SNR));
            
            elif (type(Gaussian) == (float)) or (type(Gaussian) == (int)):
                noise_amplitude = Gaussian;
                SNR = np.mean(measurement) / noise_amplitude;
                measurement = ( self.Gaussian_noise(measurement, SNR));
            
            else:            
                choice = int(input("The noise is in which scale \n1. Relative 2. Abstact\n"));
                if choice == 1:
                    SNR = float(input("Please input the Signal to Noise Ratio (Type::Float):\n"));
                elif choice == 2:
                    noise_amplitude = float(input("Please input the noise amplitude (Type::Float):\n"));
                    SNR = np.mean(measurement) / noise_amplitude;
                measurement = ( self.Gaussian_noise(measurement, SNR));
        
        measurement /= photon_number; ##normalize back to light level = 1
        return measurement;  

        
        
    #--------------------------------------------------------------------------
    # using gradient descent    
    ## No.12
    def PMT_reconstruct(self, r_tot, c_tot, measurement, upload_DMD_basis = [],
                        upload_phi = [],
                        user_evaluation = True,
                        learning_rate =  5e-4, beta = 0.5, regularization = 0, 
                        plot_step = 1000, max_epoch = False, batch_size = 10,
                        display = True):
        
        '''
        This function reconstrcuted a image from a measurement matrix(simu or 
        experiments), and save as 'self.img_reconstructed'
        It will call No.14(which will call No.13); No.13 just do GD, and No.14
        will modify import parameters if the user think the optimal hasn't
        been achieved;
        
        *r_tot, c_tot: resolution for reconstructed image
        *measurement: the measured data
        *upload_DMD_basis: for experiment data, you HAVE to upload one; for 
        simu data, if not uploads, it will use 'self.DMD_basis' made in 
        simulation part;
        *if no DMD_basis from other sources, upload_DMD_basis = [];
        if use_evaluation == True, user will be asked if this result satisfying after certain steps;
        
        a DCT2 matrix for image VECTOR is saved as self.FT_Basis
        '''
        
        #self.FT_Basis = self.IDCT_Matrix(r_tot*c_tot);
        if upload_phi == []:
            self.FT_Basis = self.DCT_Basis(r_tot,c_tot);
        else:
            self.FT_Basis = upload_phi;
        
        cycles = len(measurement);
        if upload_DMD_basis != []:
            self.DMD_basis = upload_DMD_basis;
        elif upload_DMD_basis == [] and self.DMD_basis == []:
            self.DMD_measure_basis(self.DMD_Freq_pattern(r_tot, c_tot, cycles));
        ## get self.DMD_basis;
        
        self.Measure_matrix = self.DMD_basis.dot(self.FT_Basis);
        self.W = np.zeros((len(self.FT_Basis[0,:]),len(measurement[0,:]))); ## FT domain coef
        self.measurement = measurement;
        
        ## Find the optimal
        if user_evaluation:
            self.Optimal(learning_rate, beta, regularization, plot_step, max_epoch, batch_size);
        else:
            self.Gradient_Ascent(learning_rate, beta, regularization, plot_step, max_epoch, batch_size);
        
        self.img_reconstructed_vector = self.FT_Basis.dot(self.W);
        
        self.img_reconstructed = np.zeros((r_tot, c_tot, len(measurement[0,:])));
        
        for i in range(len(measurement[0,:])):
            self.img_reconstructed[:,:,i] = np.reshape(self.img_reconstructed_vector[:,i], (r_tot, c_tot));
            
        if len(measurement[0,:]) == 1: ## for grayscaled image
            self.img_reconstructed = self.img_reconstructed[:,:,0];
            
        #self.img_reconstructed[np.where(self.img_reconstructed > 1)] = 1;
        #self.img_reconstructed[np.where(self.img_reconstructed < 0)] = 0;
        #

        if display == True:
            plt.figure();
            plt.imshow(np.clip(self.img_reconstructed, a_min = 0,a_max = 1));
            plt.show();   
        
    ## No.13    
    def Gradient_Ascent(self,learning_rate, beta = 0, regularization = 0, plot_step = 1000, max_epoch = 50000, batch_size = 'all'):
        if not max_epoch:
            max_epoch = plot_step * 50;
        
        if type(batch_size) == int:
            if batch_size == 1:
                print("Stochastic Gradient Ascent!");
            else:
                print("Mini-batch (size %s) Gradient Ascent!" %batch_size);
        else:
            print("Batch Gradient Ascent!");

        data = np.hstack((self.Measure_matrix, self.measurement));                                

        def Make_Batch(size,perm = data):            
            if type(size) == int:
                if (size <= len(self.Measure_matrix)) and (size >= 1):
                    cols = len(self.Measure_matrix[0,:]);
                    rnd_indices = np.random.randint(0, len(perm), size);
                    selection = perm[rnd_indices];

                    x = np.reshape(selection[:,:cols],(size,-1));
                    y = np.reshape(selection[:,cols:],(size,-1));                    

                    return x,y;
            return self.Measure_matrix,self.measurement;
            
        
        if beta < 0 or beta > 1:
            print('wrong beta input: beta should between 0 and 1');
            return;
        costs = [];
        c = 1;
        epoch = 0;
        
        # init the momentum!
        # --------------------------------------------------------------------
        VdW = False;
            
        # -------------------------------------------------------------------- 
            
        while epoch <= max_epoch:

            X, T = Make_Batch(batch_size); ## X,Y and T are selected in the batch! 

            Y_optimizing = X.dot(self.W);
            
            if epoch%(plot_step) == 0:   
                Y_global = self.Measure_matrix.dot(self.W);
                c = np.sum((self.measurement-Y_global)**2 / len(self.measurement));
                #print("cost:",c);
                costs.append(c);
                
                self.YZ_process_bar(epoch/max_epoch, comments="cost = %.3e"%(c), overwrite=False);
                
                
            epoch += 1;
            
            diff = (T - Y_optimizing)/len(T);
            delta_W = X.T.dot(diff);
            
            
            if type(VdW) == bool:
                VdW = delta_W;                 
            VdW = beta * VdW + (1-beta) * delta_W; 
              
            self.W += learning_rate * (VdW - regularization * np.sign(self.W)); ## Lasso
            
        #self.YZ_process_bar(epoch/max_epoch, overwrite=False);        
        plt.figure();     
        plt.plot(np.log10(costs));        
        plt.title("Optimization with momentum beta = %s & regularization = %s" %(beta, regularization));
        plt.xlabel("steps /(%s)" %(plot_step));       

        plt.ylabel("Loss (in log)");
        plt.show();
        
        self.Loss = costs;
        
    ## No.14    
    def Optimal(self,learning_rate = 1e-7, beta = 0, regularization = 0, plot_step = 1000, max_epoch = False, batch_size = 'all'):
        if not max_epoch:
            max_epoch = plot_step * 50;
            
        name_map = {1:"1.Learning rate", 2:"2.Momentum", 3:"3.Regularization",\
                    4:"4.Steps for error plotting", 5:"5.Maximum runing steps",\
                    6:"6.The size for training batch"};        
        para_map = {1:(learning_rate),
                    2:(beta), 
                    3:(regularization), 
                    4:(plot_step),
                    5:(max_epoch), 
                    6:(batch_size)};
        
        Judge = 'N'; 
        while Judge != 'Y':
            para_map[1] = float(para_map[1]);
            para_map[2] = float(para_map[2]);
            para_map[3] = float(para_map[3]);
            para_map[4] = int(para_map[4]);
            para_map[5] = int(para_map[5]);  
            if (para_map[6]) != 'all':
                para_map[6] = int((para_map[6]))
                
            self.Gradient_Ascent(para_map[1], para_map[2], para_map[3], para_map[4], para_map[5], para_map[6]);
            print('Is the Optimal achieved?');
            Judge = input('Y/N\n');

            if Judge != 'Y':                
                Adjust = 'N';
                while Adjust != 'Y':
                    print('The current parameters are :\n');
                    for i in name_map:
                        print(name_map[i],para_map[i]);
                        
                    index = int(input("Which parameter to change? Please input its index.\n Input 0 if no change.\n"));
                    if index in para_map:
                        value = (input("Please input the desired value\n"));
                        para_map[index] = value;
                    
                    Adjust = input("ALL SET?\nY/N\n");
            
    #--------------------------------------------------------------------------
    # non gradient descent method
    
    ## No.15
    def Get_direct_inverse(self, r_tot, c_tot, regularization = 0, CS = True): 
        '''
        For NON_CS:
        This function get a inverse matrix of the DMD_basis;
        
        For CS, which optimizes DCT coefficients:
        This function get a inverse matrix of the DMD_basis.dot(FT_Basis);    
        
        '''
        
        
        if CS == True:            
            self.FT_Basis = self.DCT_Basis(r_tot,c_tot);
            self.Measure_matrix = self.DMD_basis.dot(self.FT_Basis);
            
            M = self.Measure_matrix.dot(self.Measure_matrix.T);
            #M = np.linalg.pinv(self.Measure_matrix.dot(self.Measure_matrix.T));
            
            M += regularization * np.eye(M.shape[0],M.shape[1]);
            M = np.linalg.pinv(M);
            M = self.Measure_matrix.T.dot(M);  
            
        else:
            M = self.DMD_basis.dot(self.DMD_basis.T);
            #M = np.linalg.pinv(self.DMD_basis.dot(self.DMD_basis.T));
            M += regularization * np.eye(M.shape[0],M.shape[1]); 
            M = np.linalg.pinv(M);
            M = self.DMD_basis.T.dot(M);

        self.direct_inverse_M = M;   
        return M;
         
    ## No.16                
    def PMT_direct_inverse_reconstruction(self, r_tot, c_tot, M, measurement, CS = True,
                                          display = True):
        '''
        DO No.15 first to get the M and load the M to this func
        M is the matrix for reconstruction inverse matrix
        if M = np.eye, it is a Raster scan;
        
        CS = Compressed Sensing; if CS == True, we reconstructed coefficients in 
        DCT domian; otherwise, just solve the pixels
        '''
        
        self.W = M.dot(measurement);
        if CS == True:
            self.img_reconstructed_vector = self.FT_Basis.dot(self.W); 
        else:
            self.img_reconstructed_vector = self.W;
            
        self.img_reconstructed = np.zeros((r_tot, c_tot, len(measurement[0,:])));
        
        for i in range(len(measurement[0,:])):
            self.img_reconstructed[:,:,i] = np.reshape(self.img_reconstructed_vector[:,i], (r_tot, c_tot));
            
        if len(measurement[0,:]) == 1:
            self.img_reconstructed = self.img_reconstructed[:,:,0];   
            
        if display == True:        
            plt.figure();
            plt.imshow(np.clip(self.img_reconstructed, a_min = 0,a_max = 1));
            plt.show();
        
        
'''        
r = 100;
c = 100;
   
a = Compressed_sensing_PMT();
a.Load_img("ECE.jpg",RGB = True, size = (r,c), anti_aliasing = False);        
measure = a.PMT_measure_simu(1000,round(r*c/5), Poisson = False);        
a.PMT_reconstruct(r,c,measure);        
'''        
        
