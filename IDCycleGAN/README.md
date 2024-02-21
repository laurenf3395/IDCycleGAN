## Project 1 : Improving Unsupervised Face Translation between Images and Videos

Information about the models and losses is given in https://drive.google.com/file/d/1MfqAskq8_zWkk0pgq7BZ5gMfPyZJwmRR/view?usp=sharing  
Sample dataset and facenet pretrained model : https://drive.google.com/drive/folders/1J0DivJ4EgjEbgmDTnbVqfZXa_o8SJS6l?usp=sharing  
Small image datatset(sample) is given in the folder: IDCycleGAN/test_image_128 as well as the index file is in the same folder: index_file_img_test_128  
Video dataset(sample) given in the folder: IDCycleGAN/videos with index file: train_data.txt    

(For large dataset) : The image dataset CelebA can be downloaded from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. For the corresponding video dataset, Trailor faces from YouTube  


1. Training IdCycleGAN-model1

python `main_train_lau_best_models.py --experiment_name=idcyclegan_128alignhq_model1 --mode=idcyclegan_model1 --batch_size=16 --crop_size_img=128 --learning_rate=0.00028 --recover_model=False --sample_every=25 --save_model_every=1000 --num_epochs=50000`


2. Training IdCycleGAN-model2

python `main_train_lau_best_models.py --experiment_name=idcyclegan_128alignhq_model2 --mode=idcyclegan_model2 --batch_size=16 --crop_size_img=128 --learning_rate=0.00028 --recover_model=False --sample_every=25 --save_model_every=1000 --num_epochs=50000`


3. Training IdCycleGAN-model3

python `main_train_lau_best_models.py --experiment_name=idcyclegan_128alignhq_model3 --mode=idcyclegan_model3 --batch_size=16 --crop_size_img=128 --learning_rate=0.00028 --recover_model=False --sample_every=25 --save_model_every=10 --num_epochs=50000`

Packages needed:
- tensorflow
- numpy
- random
- scipy
- scikit
- Pillow
- ffmpeg( conda install -c conda-forge ffmpeg)

## Results
![Result](https://github.com/laurenf3395/Semester_Projects/blob/master/Project1-%20IDCycleGAN/Img_to_Video_to_Img.PNG/)

Real Image to fake videos generated(32 frames: shown here Frame 0, 8, 16, 32) to images generated from fake videos

## Results with different losses
![Result](https://github.com/laurenf3395/Research_Projects/blob/master/Project1-%20IDCycleGAN/Results_different_losses.PNG)

