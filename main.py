from PIL import ImageTk
from PIL import Image as PilImage
from tkinter import *
from tkinter import messagebox, filedialog
import os
# from compress_image import *
from skimage import io
from kmeans import KMeans
import pandas as pd
import time
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)

class Application(Tk):
    def __init__(self, winTitle, *args):
        super(Application, self).__init__()
        if args:
            self.configure(bg=args)
        self.xSize = self.winfo_screenwidth()
        self.ySize = self.winfo_screenheight() 
        self.geometry(f'{self.xSize}x{self.ySize}')
        self.title(winTitle)
        self.resizable(True, True)
        self.chooseClusters = Label(self, text="Number of Clusters (K)", font=("Courier", 10))
        self.chooseClusters.place(x=30, y=40)
        self.kValue = Scale(self, from_=1, to=64, orient=HORIZONTAL,length=200)
        self.kValue.place(x=220, y=20)
        self.openFileBtn = Button(text="Choose Image", command=self.getfile, bd=5)
        self.openFileBtn.place(x=40, y=80)
        self.computeClusterBtn = Button(text="Cluster", command=self.compute, bd=5)
        self.computeClusterBtn.place(x=300, y=80)
        self.mainloop()
  
    def getfile(self):
        self.fileLocation = filedialog.askopenfilename(initialdir=os.getcwd())
        self.getImageName = self.fileLocation.rsplit("/", 1)[1]
        if (self.getImageName.rsplit('.',1)[1] == 'csv'):
            self.df = pd.read_csv(self.fileLocation)
            fig = Figure(figsize = (6, 4), dpi = 100)
            cluster_plot = fig.add_subplot(111)
            cluster_plot.scatter(self.df[self.df.columns.values[0]], self.df[self.df.columns.values[1]])
            self.input_canvas = FigureCanvasTkAgg(fig,master = self)  
            self.input_canvas.draw()
            self.input_canvas.get_tk_widget().pack()
            self.input_canvas.get_tk_widget().place(x=20,y=180)
        else:
            raw_image=PilImage.open(self.fileLocation)
            height = raw_image.height
            width = raw_image.width
            if (width>600):
                height = int(height/width * 600)
                width = 600
            raw_image= raw_image.resize((width,height))
            raw_image=ImageTk.PhotoImage(raw_image)
            self.input_canvas=Label(self)
            self.input_canvas.pack()
            self.input_canvas.configure(image=raw_image)
            self.input_canvas.image=raw_image
            self.input_canvas.place(x=20,y=180)
            # raw_size=os.path.getsize(self.fileLocation)
            # lb2= Label(self)
            # lb2.pack()
            # lb2.configure(text=f"Size: {raw_size} Bytes")
            # lb2.place(x=270,y=180+height+50)

    def compute(self):
        self.kVal = self.kValue.get()       
        try:
            start = time.time()
            computeLocation = "compressed_"+self.getImageName
            if (self.getImageName.rsplit('.',1)[1] == 'csv'):
                model = KMeans(max_iter = 500, tolerance = 0.001, n_clusters = self.kVal, runs = 100)
                (clusters, data_with_clusters) = model.fit(self.df)
                fig = Figure(figsize = (6, 4), dpi = 100)
                clustered_plot = fig.add_subplot(111)
                for i,cluster in enumerate(clusters):
                    data_cluster_i = data_with_clusters[ data_with_clusters[:, -1] == i ]
                    clustered_plot.scatter(data_cluster_i[:, 0], data_cluster_i[:, 1])
                    clustered_plot.plot(cluster[0], cluster[1], label = 'Centroid ' + str(i), marker='*', markersize=15, markeredgecolor="k", markeredgewidth=1)
                self.output_canvas = FigureCanvasTkAgg(fig,master = self)  
                self.output_canvas.draw()
                self.output_canvas.get_tk_widget().pack()
                self.output_canvas.get_tk_widget().place(x=620,y=180)
            else:
                print (computeLocation)
                self.segment(self.fileLocation,computeLocation,self.kVal)
                computed_image=PilImage.open(computeLocation)
                height = computed_image.height
                width = computed_image.width
                if (width>600):
                    height = int(height/width * 600)
                    width = 600
                computed_image = computed_image.resize((width,height))
                computed_image=ImageTk.PhotoImage(computed_image)
                self.output_canvas=Label(self)
                self.output_canvas.pack()
                self.output_canvas.configure(image=computed_image)
                self.output_canvas.image=computed_image
                self.output_canvas.place(x=620,y=180)
                # computed_size=os.path.getsize(computeLocation)
                # lb2 = Label(self)
                # lb2.pack()
                # lb2.configure(text=f"Size: {computed_size} Bytes")
                # lb2.place(x=800,y=180+height+50)
            end = time.time()
            lb2 = Label(self)
            lb2.pack()
            lb2.configure(text=f"Clustered Computed in:{end-start} seconds")
            lb2.place(x=400,y=600)
        except:
            messagebox.showwarning("Error", "Something went wrong")
    
    def segment(self,image_name, compressed_image_name, n_clusters = 4):
        image = io.imread(image_name)
        orig_shape = image.shape
        image = image.reshape(-1, image.shape[2]) / 255 # Normalization. It improves the performance so much!
        img_shape = image.shape   
        
        model = KMeans(n_clusters=n_clusters, tolerance = 0.01, runs=100)
        cluster_means, image_data_with_clusters = model.fit(image)
        compressed_image = np.zeros(img_shape)

        for i, cluster in enumerate(image_data_with_clusters[:, -1]):
            compressed_image[i, :] = cluster_means[ int(cluster) ]
        compressed_image = compressed_image * 255
        compressed_image_reshaped = compressed_image.reshape(orig_shape).astype('uint8') # Can't write float type matrix to an image file
        io.imsave(compressed_image_name, compressed_image_reshaped)
    

application = Application("K-means Clustering")