import Tkinter
import csv
import tkFileDialog
import tkMessageBox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import string

filename = ""
fr_c = 0
pressed = 0
rows = []
lab = []

##################### TRAINING #####################################

# function to extract haralick textures from an image
def extract_features(image):
	# calculate haralick texture features for 4 types of adjacency
        means = cv2.mean(image)
        means = means[:3]
        (means, stds) = cv2.meanStdDev(image)
        f1=means.flatten()
        f2=stds.flatten()
        moments=cv2.HuMoments(cv2.moments(image)).flatten()
	#hog = cv2.HOGDescriptor()
	#h = hog.compute(image)
        textures = mt.features.haralick(image)
        ht_mean  = textures.mean(axis=0)
        blur = cv2.GaussianBlur(image,(5,5),0)
        ret,thresh2 = cv2.threshold(blur,200,255,cv2.THRESH_BINARY_INV)
        kernel = np.ones((6,6),np.uint8)
        closing = cv2.morphologyEx(thresh2,cv2.MORPH_CLOSE, kernel)
        _,contours,hierarchy = cv2.findContours(closing, 1, 2)
       # cv2.drawContours(img, contours, -1, (0,255,0), 3)
        cnt = contours[0]
        fe_vec=np.concatenate([f1, f2, moments,ht_mean]).flatten()
       # print fe_vec
	#fe_vec = np.concatenate([means, stds, ht_mean])
        return fe_vec


def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)

def read_csv(File,File1) :
        global rows
        global lab
        
        # csv file name
        f = File
        f1 = File1
         
        # reading features csv file
        with open(f, 'r') as csvfile:
            # creating a csv reader object
            csvreader = csv.reader(csvfile)
         
            # extracting each data row one by one
            for row in csvreader:    
                rows.append(row)
                #print rows
                

        # reading labels csv file
        with open(f1, 'r') as csvfile1:
            # creating a csv reader object
            csvreader1 = csv.reader(csvfile1)
         
            # extracting each data row one by one
            for col in csvreader1:    
                lab.append(string.join(col, ""))
               

### load the training dataset
##train_path  = "E:/BE_PROJECT/Project"
##train_names = os.listdir(train_path)

# empty list to hold feature vectors and train labels
##train_features = []
##train_labels   = []

# loop over the training dataset
##i = 1
#print ("[STATUS] Started extracting haralick textures..")
##for train_name in train_names:
##	cur_path = train_path + "/" + train_name
##	cur_label = train_name
##	i = 1
##
##	for file in glob.glob(cur_path +"/*.jpg"):
##		#print ("Processing Image - {} in {}".format(i, cur_label))
##		# read the training image
##		image = cv2.imread(file)
##
##		# convert the image to grayscale
##		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
##
##		# extract haralick texture from the image
##		features = extract_features(gray)
##
##		# append the feature vector and label
##		train_features.append(features)
##		train_labels.append(cur_label)
##
##		# show loop update
##		i += 1

# have a look at the size of our feature vector and labels
##print ("Training features: {}".format(np.array(rows).shape))
##print ("Training labels: {}".format(np.array(lab).shape))

##data =train_features
##path = "features.csv"
##csv_writer(data, path)
##
##labels =train_labels
##path = "labels.csv"
##csv_writer(labels, path)

#labels = np.unique(train_labels)
#print(labels)
 
# create the classifier
#print ("[STATUS] Creating the classifier..")
clf_svm = LinearSVC(multi_class='crammer_singer')

# fit the training data and labels
#print ("[STATUS] Fitting data/label to model..")
File = "E:/BE_PROJECT/Working_Module/features_small.csv"
File1 = "E:/BE_PROJECT/Working_Module/labels_small.csv"
read_csv(File,File1)
print ("Training features: {}".format(np.array(rows).shape))
print ("Training labels: {}".format(np.array(lab).shape))

clf_svm.fit(rows, lab)


class login:
   def __init__(self, master):
      self.master = master
##      self.path = "Fruit.jpg"
      master.configure(background='sky blue')      
      Tkinter.Label(root, text="FRUIT CLASSIFIER",font=("Mannberg",50),bg = 'sky blue',fg = 'red').place(x=415,y=150)
      Tkinter.Label(root, text="Username",font=(10),width=10,fg='dark green',bg = 'sky blue').place(x=525,y=300)
      Tkinter.Label(root, text="Password",font=(10),width=10,fg='dark green',bg = 'sky blue').place(x=525,y=350)
      self.e1 = Tkinter.Entry(master,width = 20,font=(10))
      self.e2 = Tkinter.Entry(master,width = 20,show='*',font=(10))
      self.e1.pack(ipady=10)
      self.e2.pack(ipady=10) 
      self.e1.place(x=675,y=300)
      self.e2.place(x=675,y=350)
      #Tkinter.Button(root, text='Quit', command=root.quit).place(x=800,y=500)
      Tkinter.Button(root, text='LOGIN', command=self.show_entry_fields,width=30,bg = 'dark green',fg = 'white').place(x=600,y=400)


   def show_entry_fields(self):
      if(self.e1.get()=="fruit" and self.e2.get()=="skn"):
                c = Tkinter.Canvas(root,height=1200,width=1920,bg = 'sky blue')
                c.pack()

                label1 = Tkinter.Label(c, text='Selected Image', fg='white', bg='dark green',height=1,width=20,font=(1))
                label1.pack()
                c.create_window(250, 500, window=label1)

                label2 = Tkinter.Label(c, text='Thresholded Image', fg='white', bg='dark green',height=1,width=20,font=(1))
                label2.pack()
                c.create_window(600, 500, window=label2)

                label3 = Tkinter.Label(c, text='Closing Operation', fg='white', bg='dark green',height=1,width=20,font=(1))
                label3.pack()
                c.create_window(950, 500, window=label3)

                label4 = Tkinter.Label(c, text='Foreground Image', fg='white', bg='dark green',height=1,width=20,font=(1))
                label4.pack()
                c.create_window(1300, 500, window=label4)

                label5 = Tkinter.Label(c, text='FRUIT : ', fg='black', bg='indian red',height=2,width=20,font=(1))
                label5.pack()
                c.create_window(550, 750, window=label5)

                label6 = Tkinter.Label(c, text='FRUIT COUNT : ', fg='black', bg='indian red',height=2,width=20,font=(1))
                label6.pack()
                c.create_window(1000, 750, window=label6)

                panel1 = Tkinter.Label(root)
                panel2 = Tkinter.Label(root)
                panel3 = Tkinter.Label(root)
                panel4 = Tkinter.Label(root)
                        

                def select():
                        global filename

                        if fr_c > 0 :
                                btn2.configure(text = "SHOW RESULT")
                                label5.configure(text='FRUIT : ')
                                label6.configure(text='FRUIT COUNT : ')
                                panel1.configure(image='')
                                panel2.configure(image='')
                                panel3.configure(image='')
                                panel4.configure(image='')
                        filename = tkFileDialog.askopenfilename(filetypes = (("JPEG", "*.jpg;*.jpeg")
                                                                             ,("PNG", "*.png")
                                                                             ,("All files", "*.*") ))

                        if(filename != ""):

                                global pressed
                                pressed = 1
                                
                                #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
                                img1 = ImageTk.PhotoImage(file=filename)

                                #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
                                panel1.configure(image=img1)
                                panel1.image=img1
                                #The Pack geometry manager packs widgets in rows or columns.
                                panel1.pack(side = "bottom", fill = "both", expand = "yes")
                                panel1.place(x=120,y=140)                                


                        
                def result() :
                    if(pressed == 1):    
                            global fr_c
                
                            # convert to grayscale
                            img_f = cv2.imread(filename)
                            gray_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY)

                            # extract haralick texture from the image
                            features1 = extract_features(gray_f)

                            features2=features1.reshape(1,-1)
                            z = clf_svm.predict(features2)                          

                            th, im_th = cv2.threshold(gray_f, 220, 255, cv2.THRESH_BINARY_INV);
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
                            closing = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel)

                            # Copy the thresholded image.
                            im_floodfill = im_th.copy()
                             
                            # Mask used to flood filling.
                            # Notice the size needs to be 2 pixels than the image.
                            h, w = im_th.shape[:2]
                            mask = np.zeros((h+2, w+2), np.uint8)
                             
                            # Floodfill from point (0, 0)
                            cv2.floodFill(im_floodfill, mask, (0,0), 255);
                             
                            # Invert floodfilled image
                            im_floodfill_inv = cv2.bitwise_not(im_floodfill)
                             
                            # Combine the two images to get the foreground.
                            im_out = im_th | im_floodfill_inv

                            # Erosion operation
                            kernel = np.ones((7,7), np.uint8)
                            #im_erosion = cv2.erode(im_th, kernel,iterations=4)
                              
                            ret, labels = cv2.connectedComponents(closing)
                            output = cv2.connectedComponentsWithStats(closing,8)
                            # Get the results
                            # The first- cell is the number of labels
                            num_labels = output[0]
                            # The second cell is the label matrix
                            labels = output[1]
                            # The third cell is the  stat matrix
                            stats = output[2]
                            # The fourth cell is the centroid matrix
                            centroids = output[3]
                            fr_c= (num_labels - 1)

                            fruit = string.join(z," ")
        
                            im2=Image.fromarray(im_th)
                            imgtk2 = ImageTk.PhotoImage(image=im2)
                            #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
                            panel2.configure(image=imgtk2)
                            panel2.image=imgtk2
                            #The Pack geometry manager packs widgets in rows or columns.
                            panel2.pack(side = "bottom", fill = "both", expand = "yes")
                            panel2.place(x=470,y=140)



                            #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
                            im3 = Image.fromarray(closing)
                            imgtk3 = ImageTk.PhotoImage(image=im3)
                            #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
                            panel3.configure(image=imgtk3)
                            panel3.image=imgtk3
                            #The Pack geometry manager packs widgets in rows or columns.
                            panel3.pack(side = "bottom", fill = "both", expand = "yes")
                            panel3.place(x=815,y=140)


                            #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
                            im4 = Image.fromarray(im_out)
                            imgtk4 = ImageTk.PhotoImage(image=im4)
                            #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
                            panel4.configure(image=imgtk4)
                            panel4.image=imgtk4
                            #The Pack geometry manager packs widgets in rows or columns.
                            panel4.pack(side = "bottom", fill = "both", expand = "yes")
                            panel4.place(x=1165,y=140)                          
                            

                            label5.configure(text='FRUIT : ' + fruit)
                            label6.configure(text='FRUIT COUNT : ' + str(fr_c))            


                btn1 = Tkinter.Button(root,height=1,width=25,text="SELECT IMAGE",command=select,font=(1),fg='white',bg = 'steel blue')
                btn1.pack()
                w1 = c.create_window(750,100,window=btn1)

                btn2 = Tkinter.Button(root,height=1,width=25,text="SHOW RESULT",command=result,font=(1),fg='white',bg = 'steel blue')
                btn2.pack()
                w2 = c.create_window(750,620,window=btn2)

      else :
              # An error box
              tkMessageBox.showerror("Error","Wrong Credentials...!!\n\nPlease enter again")
            
         
root = Tkinter.Tk()
root.title("Fruit Classifier")
root.geometry("1920x1200")
login_screen=login(root)
root.mainloop( )






