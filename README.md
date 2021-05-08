# Final_project_CS570

# DCGAN: 
1. Prepare data:
Download the data from https://drive.google.com/file/d/1RjyPm1mmGBxzairo43XTWoEwUeBiUmuC/view?usp=sharing
Then, extract the data into this repository. 

2. Training: 
To train, just run file DCGAN_train.py 

There are some parameters to configure:
- There is a variable named IMG_SIZE in the file DCGAN_train.py, when the program loads images from the data directory, then it will resize image into (IMG_SIZE, IMG_SIZE). Therefore, if we want to train a DCGAN to generate images of resolution (128,128), then we need to set IMG_SIZE = 128
- The following line is to load data from the data directory: 
   X_normal= np.array(load_images('Pneumonia_NORMAL',size=(IMG_SIZE, IMG_SIZE))) 
   Here 'Pneumonial_NORMAL' is the name of our data directory. 
- n_epoch is number of epochs 
- batch_size 

3. Result:
During training process, the program will create a new subfolder under the directory "save_DCGAN", and then model weight would be saved there. 

# CGAN (conditional GAN)
1. Download the data from https://drive.google.com/file/d/1vgiNEfc2Ia1AetRjJw1aT5VSyi1eXOeB/view?usp=sharing
Then, extract the data into this repository 

2. Training:
To train, just run file CGAN_train.py 
Therere are some parameters to configure:
- There is a variable named IMG_SIZE in the file CGAN_train.py, when the program loads images from the data directory, then it will resize image into (IMG_SIZE, IMG_SIZE). Therefore, if we want to train a DCGAN to generate images of resolution (128,128), then we need to set IMG_SIZE = 128
- n_epoch is number of epochs 
- batch_size 

3. Result:
During training process, the program will create a new subfolder under the directory "save_CGAN", and then model weight would be saved there. 


# Notice 
I have trained the DCGAN model for 500 epochs and CGAN model for 200 epochs.
The trained weights can be found here: 
https://drive.google.com/drive/folders/1OZrccNu7_TAwRBTVPT9MOJJbgch1xaN7?usp=sharing

The CGAN may need to be trained for 500 epochs but i don't have enough time. Please train it again!
