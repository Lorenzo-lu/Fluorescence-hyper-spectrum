# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:30:21 2020

@author: yluea
"""

import numpy as np;
import matplotlib.pyplot as plt;
#import matplotlib.image as mpimg;

import fitz
import os

import datetime;
'''
This class is used to convert the result into figures with watermark and
save as PDF
'''

class YZ_save_results:
    
    def __init__(self):
        self.total = 0; ## mark how many figs saved in total
        self.form = '.jpg';
        #pass;
        
    def Load_img_mtx(self, matrix, condition):
        '''
        Load an matrix as an image
        '''
        ## the condition should be a string describing the tests
        self.img = matrix;        
        self.condition = condition;
        
    def Plot_water_mark(self):
        '''
        add a water mark on a figure by plt
        '''
        if len(self.img.shape) == 2:
            #plt.imshow(self.img, cmap = 'gray');
            plt.imshow(np.clip(self.img, a_min = 0,a_max = 1), cmap = 'gray')
        elif len(self.img.shape) == 3:
            #plt.imshow(self.img);
            plt.imshow(np.clip(self.img, a_min = 0,a_max = 1));
        else:
            return False;
        
        plt.text(0, -(self.img.shape[0] * 0.1), self.condition, fontsize = (self.img.shape[0] * 0.1));
        #add text to fig        
        plt.savefig(str(self.total) + self.form); #save fig
        plt.show();
        self.total += 1; #counting the total number of pages
        
          
    def Pdf_init(self, filename = False, TEXT = False):
        '''
        Initialize a PDF; arg filename: of this PDF
        default TEXT = False        
        '''
        if filename == False:
            filename = input("Please input the filename to save the result\n(don't include .pdf)\n");
            filename += '.pdf';    
            
        self.filename = filename;    
        #self.Make_preface(remark);
        if os.path.exists(filename):
            os.remove(filename);
        
        pdf_new = fitz.open();  # create a new PDF file
        pdf_new.newPage(width=800, height=500); # Insert a white page
        page = pdf_new[-1]; # Last page
        
        if TEXT == False:
            TEXT = 'Digital Mirror Device Theory Investigation\nYizhou Lu\n\n';
            TEXT += '\nDepartment of Electrical and Computer Engineering';
            TEXT += '\nUniversity of Wisconsin Madison';
        
        #rect = fitz.Rect(0, 50, 250, 100);
        rect = fitz.Rect(0, 50, 800, 450); #where
        page.insertTextbox(rect, TEXT, fontsize = 20, # choose fontsize (float)
               fontname = "Times-Roman",       # a PDF standard font
               fontfile = None,                # could be a file on your system
               align = 1);  
        #page.insertPage(-1, TEXT, fontsize=8, width=255, \
                           #height=255, fontname="helv", fontfile=None, color=None); 
            
        pdf_new.save(filename);
        pdf_new.close();        
        
        
    def Make_preface(self, filename = False, title = False):
        '''
        Make a preface
        This function can be used to generate an important notes at an 
        indiviual page;
        '''
        if filename == False:
            if self.filename:
                filename = self.filename;
            else:
                filename = input("Please input the filename to save the result\n(don't include .pdf)\n");
                filename += '.pdf';
                
        pdf_new = fitz.open(filename);
        
        if title == False:
            title = input("Please input the experiment condition or other remarks\n");
        title += '\n';
        
        #pdf_new.newPage(width=250, height=150);
        pdf_new.newPage(width=800, height=500);
        page = pdf_new[-1];
            
        title += '\nTime:\n';
        title += str(datetime.datetime.now());
        #rect = fitz.Rect(50, 50, 250, 100);
        rect = fitz.Rect(100, 50, 700, 450);
        
        page.insertTextbox(rect, title, fontsize = 15, # choose fontsize (float)
               fontname = "Times-Roman",       # a PDF standard font
               fontfile = None,                # could be a file on your system
               align = 0);  
            
        pdf_new.saveIncr();
        pdf_new.close();         
        
        
        
    def Pdf_write(self, filename = False):  
        '''
        if loaded a series of images, this images with water marks will be saved
        as 0.jpg 1.jpg 2.jpg... (the ext can be changed in self.form)
        then, this function insert these images one by one to the pre initialized PDF
        '''
        if filename == False:
            if self.filename:
                filename = self.filename;
            else:
                filename = input("Please input the filename to save the result\n(don't include .pdf)\n");
                filename += '.pdf';
        
        pdf_new = fitz.open(filename);
        
        ## initialize the simulation remarks   
        
        for pic in range(self.total):
            #pdf_new.newPage(width=250, height=150);
            pdf_new.newPage(width=800, height=500);
            
            page = pdf_new[-1];
            img_name = str(pic) + self.form;
            #rect = fitz.Rect(50, 0, 200, 150);
            rect = fitz.Rect(50, 0, 750, 500);
            
            #pix = fitz.Pixmap(img_name);        # any supported image file
            page.insertImage(rect, filename = img_name);
            
        if os.path.exists('Loss.jpg'):
            #pdf_new.newPage(width=250, height=150);
            pdf_new.newPage(width=800, height=500);
            
            page = pdf_new[-1];
            #rect = fitz.Rect(50, 0, 200, 150);
            rect = fitz.Rect(50, 0, 750, 500);
            
            #pix = fitz.Pixmap("Loss.png");        # any supported image file
            page.insertImage(rect, filename = 'Loss.jpg', overlay=True);
            
            os.remove('Loss.jpg');
        
            
        ## save the results
        '''
        for pic in range(self.total):
            img_name = str(pic) + '.jpg';
            img = fitz.open(img_name);
            
            print(img_name);
            
            pdfbytes = img.convertToPDF();
            imgpdf=fitz.open("pdf",pdfbytes);
            pdf_new.insertPDF(imgpdf);
       
            
        if os.path.exists('Loss.jpg'):
            img = fitz.open('Loss.jpg');
            pdfbytes = img.convertToPDF();
            imgpdf=fitz.open("pdf",pdfbytes);
            pdf_new.insertPDF(imgpdf);
            
            os.remove('Loss.jpg');
         ''' 
             
        #if os.path.exists(filename):
            #os.remove(filename);
            
        #pdf_new.save(filename);
        pdf_new.saveIncr();
        pdf_new.close();
            
        for pg in range(self.total):
            os.remove((str(pg) + self.form));
            
        print('DONE.');
        self.total = 0; ## restore the image number cuz all the images has been deleted
        
        
    def Pdf_insert_addition(self,  image_name, pdf_name = False):
        '''
        New images can be inserted here. 
        pdf = fitz.open(pdfname), and pdf is a list
        pdf[n] means the n_th page in this PDF, and either texts or images
        can be added
        '''
        if pdf_name == False:
            if self.filename:
                pdf_name = self.filename;
            else:
                pdf_name = input("Please input the filename to save the result\n(don't include .pdf)\n");
                pdf_name += '.pdf';
                
        pdf_new = fitz.open(pdf_name);
        #pdf_new.newPage(width=250, height=150);
        pdf_new.newPage(width=800, height=500);
        
        page = pdf_new[-1];
        
        #rect = fitz.Rect(50, 0, 200, 150);
        rect = fitz.Rect(50, 0, 750, 500);
        
        page.insertImage(rect, filename = image_name);
        if os.path.exists(image_name):
            '''
            img = fitz.open(image_name);
            pdfbytes = img.convertToPDF();
            imgpdf=fitz.open("pdf",pdfbytes);
            pdf_new.insertPDF(imgpdf);            
            '''
            page.insertImage(rect, filename = image_name);
            
        pdf_new.saveIncr();
        pdf_new.close();    
        
            