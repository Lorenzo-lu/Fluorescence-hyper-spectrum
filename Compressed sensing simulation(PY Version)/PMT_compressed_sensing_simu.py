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
from skimage.transform import rescale, resize, downscale_local_mean;

class Compressed_sensing_PMT:   
    def __init__(self):
        pass;
            
    ## These are modified dct2 functions
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dct2(self,block,padding = False):
        #block = np.array(block);
        if not np.all(padding):
            return dct(dct(block.T).T);
            #return dct(dct(block.T, norm='ortho').T, norm='ortho');
        else:
            [c,r] = padding;
            return dct(dct(block.T, n = c).T, n = r);
            #return dct(dct(block.T, n = c, norm='ortho').T, n = r, norm='ortho');    
    def idct2(self,block,padding = False):    
        #block = np.array(block);
        if not np.all(padding):
            
            r0,c0 = block.shape;
            return idct(idct(block.T).T)/r0/c0;
            #return idct(idct(block.T, norm='ortho').T, norm='ortho')/r0/c0;
        else:
            [c,r] = padding;
            return idct(idct(block.T, n = c).T, n = r)/(r*c);
            #return idct(idct(block.T, n = c, norm='ortho').T, n = r, norm='ortho');
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ## Generate the Frequence Domain; I choose DCTMTX here. 
    def DCT_Matrix(self, N):        
        I = np.eye(N);
        dctmtx = dct(I, axis=0, norm = 'ortho');
        return dctmtx;   
    def IDCT_Matrix(self, N):
        I = np.eye(N);
        idctmtx = idct(I, axis=0, norm = 'ortho');
        return idctmtx;  
        
    def DCT_Basis(self,r_tot,c_tot): ## This is used to translate idct2 into a matrix tansforming a img vector
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
    def DMD_Freq_pattern(self, r,c, cycles = False):
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
    
    def DMD_pattern(self,r,c, size = False): ## 0,1 matrix by idct2
        #x_r = np.array(range(r_tot)) / r_tot * np.pi;
        #x_c = np.array(range(c_tot)) / c_tot * np.pi;
        #y_r = np.cos(x_r * r) * 0.5; ## to avoid -1. we only need 0 and 1
        #y_c = np.cos(x_c * c) * 0.5;
        #y_r = np.reshape(y_r, (-1,1));
        #y_c = np.reshape(y_c, (-1,1));  
        
        if not np.all(size):        
            self.img_DCT_domain[r,c] = 1;
            DCT_pattern = self.idct2(self.img_DCT_domain);
            self.img_DCT_domain[r,c] = 0;
        else:
            #r_tot, c_tot = size
            DCT = np.zeros(size);
            DCT[r,c] = 1;
            DCT_pattern = self.idct2(DCT);
        
        return np.ceil(DCT_pattern);
    
    def DMD_measure_basis(self, freq_pattern): 
        ## reshape each DMD_pattern in to a row in measure matrix
        index = np.where(freq_pattern == 1);
        index = np.array(index).T;
        r_tot, c_tot = freq_pattern.shape;
        
        basis = np.zeros((len(index), r_tot * c_tot));
        
        for i in range(len(index)):            
            basis[i,:] = np.reshape(self.DMD_pattern(index[i][0], index[i][1]), (1,-1));
        
        self.DMD_basis = basis;
        #return basis;        
    
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++     
    ## Define noise     
    def Poisson_noise(self, array):
        return np.random.poisson(array);        
        
    def Gaussian_noise(self,array, SNR = False):
        if not SNR:
            SNR = 100;  
        
        std = np.mean(array) / SNR;            
        return np.random.normal(array,std);
    
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++     
    ## for simulation                 
    def Load_img(self, img_pathname = False, RGB = False, size = False, anti_aliasing = True):
        if not img_pathname:
            img_pathname = input("Please input the pathname of the image you want to test.\n");
        m = mpimg.imread(img_pathname);
        
        if not RGB:
            m = m.mean(axis = 2);
            # for grayscale image, remember plt.imshow(m, cmap = 'gray')
        else:
            pass;
            
        if np.all(size):
            r = size[0];
            c = size[1];            
            m = resize(m, (r, c),anti_aliasing);
            
        self.img = m / np.max(m);
        if len(self.img.shape) == 3:
            plt.imshow(self.img);
        elif len(self.img.shape) == 2:
            plt.imshow(self.img, cmap = 'gray');
        plt.show();
        
        self.DMD_basis = [];
        self.img_DCT_domain = [];
        #self.img_DCT_domain = np.zeros(self.img.shape);        
        
    def Load_photon_numbers(self, n_photons):        
        if type(n_photons*1.0) == float:
            self.n_photons = np.array([n_photons]);
        elif type(n_photons) == list:
            self.n_photons = np.array(n_photons);
        elif type(n_photons) == np.ndarray:
            self.n_photons = n_photons;
        else:
            print("You should input the number of photons as a number, list or array");
        
            
    def PMT_measure_simu(self, photon_number, cycles, Poisson = False, Gaussian = False):
        ## Photon Number investigates the impacts of the Intensity        
        
        if len(self.img.shape) == 2:
            r_tot,c_tot = self.img.shape;
            img_vector = np.zeros((r_tot*c_tot,1))
            img_vector[:,0] = np.reshape(self.img, (r_tot*c_tot));## make img a vector
            
        elif len(self.img.shape) == 3: ## make RGB image a n*3 matrix
            r_tot,c_tot = self.img[:,:,0].shape;
            img_vector = np.zeros((r_tot*c_tot,3))
            
            for i in range(3):
                img_vector[:,i] = (np.reshape(self.img[:,:,i], (r_tot*c_tot)));
        else:
            print("Wrong image shape");
            return;
        
        img_vector *= photon_number; ## how bright the environment is
        
        
        measurement = np.zeros((cycles, len(img_vector[0,:]))); ## record the measured intensity
        
        self.DMD_measure_basis(self.DMD_Freq_pattern(r_tot, c_tot, cycles)); 
        ## That function has no return:: it make the matrix of self.DMD_basis, which means
        ## cycles x measurements in vector form
        
        
        for layer in range(len(img_vector[0,:])):
            
            measurement_layer = [];
            for i in range(len(self.DMD_basis)):   
                
                val = self.DMD_basis[i,:].dot(img_vector[:,layer]);
                if Poisson:
                    val = self.Poisson_noise(val);
                
                measurement_layer.append(val);
            measurement_layer = np.array(measurement_layer);
               
            measurement[:,layer] = measurement_layer;
            
        if Gaussian != False:## Gaussian noise from current, so added at the final
            if Gaussian == 'Poisson': ## Use Gaussian to simulate Poisson
                SNR = (np.mean(measurement))**0.5;
                measurement = ( self.Gaussian_noise(measurement, SNR));
            
            else:            
                choice = int(input("The noise is in which scale \n1. Relative 2. Abstact\n"));
                if choice == 1:
                    SNR = float(input("Please input the Signal to Noise Ratio (Type::Float):\n"));
                elif choice == 2:
                    noise_amplitude = float(input("Please input the noise amplitude (Type::Float):\n"));
                    SNR = np.mean(measurement) / noise_amplitude;
                measurement = ( self.Gaussian_noise(measurement, SNR));
        
        measurement /= photon_number;
        return measurement;  

        
        
    #--------------------------------------------------------------------------
    # using gradient descent    
    def PMT_reconstruct(self, r_tot, c_tot, measurement, user_evaluation = True,\
                        learning_rate =  5e-4, beta = 0.5, regularization = 0, plot_step = 1000, max_epoch = False, batch_size = 10):
        
        #self.FT_Basis = self.IDCT_Matrix(r_tot*c_tot);
        self.FT_Basis = self.DCT_Basis(r_tot,c_tot);
        
        cycles = len(measurement);
        if not np.all(self.DMD_basis):
            self.DMD_measure_basis(self.DMD_Freq_pattern(r_tot, c_tot, cycles));
        ## get self.DMD_basis;
        
        self.Measure_matrix = self.DMD_basis.dot(self.FT_Basis);
        self.W = np.zeros((r_tot*c_tot,len(measurement[0,:]))); ## FT domain coef
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
            
        if len(measurement[0,:]) == 1:
            self.img_reconstructed = self.img_reconstructed[:,:,0];
            
        #self.img_reconstructed[np.where(self.img_reconstructed > 1)] = 1;
        #self.img_reconstructed[np.where(self.img_reconstructed < 0)] = 0;
        self.img_reconstructed = np.clip(self.img_reconstructed, a_min = 0,a_max = 1);
        
        plt.figure();
        plt.imshow(self.img_reconstructed);
        plt.show();   
        
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
                c = np.sum((self.measurement-Y_global)**2) / len(self.measurement);
                print("cost:",c);
                costs.append(c);              
                
            epoch += 1;
            
            diff = (T - Y_optimizing);
            delta_W = X.T.dot(diff)/len(T);
            
            if type(VdW) == bool:
                VdW = delta_W;                 
            VdW = beta * VdW + (1-beta) * delta_W;                
            self.W += learning_rate * (VdW - regularization * np.sign(self.W)); ## Lasso
                
        plt.figure();     
        plt.plot(np.log10(costs));        
        plt.title("Optimization with momentum beta = %s & regularization = %s" %(beta, regularization));
        plt.xlabel("steps /(%s)" %(plot_step));       

        plt.ylabel("Loss (in log)");
        plt.show();
        
        self.Loss = costs;
        
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
    def Get_direct_inverse(self, r_tot, c_tot,  regularization = 0):  
            self.FT_Basis = self.DCT_Basis(r_tot,c_tot);
            self.Measure_matrix = self.DMD_basis.dot(self.FT_Basis);
            
            M = np.linalg.pinv(self.Measure_matrix.dot(self.Measure_matrix.T));
            
            M += regularization * np.eye(M.shape[0],M.shape[1]);                           
            
            M = self.Measure_matrix.T.dot(M);  

            self.direct_inverse_M = M;            
                    
    def PMT_direct_inverse_reconstruction(self, r_tot, c_tot, M, measurement):
        
        #def Make_direct_inverse(r_tot, c_tot):
            #self.FT_Basis = self.DCT_Basis(r_tot,c_tot);
            #self.Measure_matrix = self.DMD_basis.dot(self.FT_Basis);
            
            #M = np.linalg.pinv(self.Measure_matrix.dot(self.Measure_matrix.T));
            
            #M += regularization * np.eye(M.shape[0],M.shape[1]);                           
            
            #M = self.Measure_matrix.T.dot(M); 
            #return M;
        
        #M = np.linalg.pinv(self.Measure_matrix);
        
        self.W = M.dot(measurement);
        self.img_reconstructed_vector = self.FT_Basis.dot(self.W);        
        self.img_reconstructed = np.zeros((r_tot, c_tot, len(measurement[0,:])));
        
        for i in range(len(measurement[0,:])):
            self.img_reconstructed[:,:,i] = np.reshape(self.img_reconstructed_vector[:,i], (r_tot, c_tot));
            
        if len(measurement[0,:]) == 1:
            self.img_reconstructed = self.img_reconstructed[:,:,0];            
        self.img_reconstructed = np.clip(self.img_reconstructed, a_min = 0,a_max = 1);        
        plt.figure();
        plt.imshow(self.img_reconstructed);
        plt.show();

        
        
'''        
r = 100;
c = 100;
   
a = Compressed_sensing_PMT();
a.Load_img("ECE.jpg",RGB = True, size = (r,c), anti_aliasing = False);        
measure = a.PMT_measure_simu(1000,round(r*c/5), Poisson = False);        
a.PMT_reconstruct(r,c,measure);        
'''        
        
