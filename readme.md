<h1 align="center">Extending TransUNet for 
Synthetic vs. Real Image Detection
</h1> 

<p align="center"> <a href="https://www.linkedin.com/in/jojyalex/">Jojy Alex</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp<a href="https://www.linkedin.com/in/manish-singh-6a46b7108/">Manish Singh</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp<a href="https://www.linkedin.com/in/sanjaykrishnaswami/">Sanjay Krishnaswami</a>

<p align="center"><a href="https://github.com/krsanjay01/MSIS_Capstone_Team1">Project Page</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <a href=https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/R50%2BViT-B_16.npz">Pre-trained Model</a> </p>



> The detection of synthetic images vs real image has become increasingly challenging 
> due to advancements in several image generative models like GLIDE, BigGAN, StyleGAN, 
> Stable Diffusion etc. particularly when images are compressed. This project aims to 
> enhance the compressed image detection accuracy by extending the TransUNet architecture, 
> which we call “ TransUNet Classifier”. Our results show an improvement in the accuracy 
> with the TransUNet Classifier, achieving over 90% accuracy on compressed images. The 
> TransUNet Classifier trained on mixed images, created from multiple image generators, 
> performs the best, making it a powerful general purpose synthetic image detector. 
> The project has important implications for image forensics, AI-generated content 
> verification, and legal security, particularly in areas where digital image authenticity 
> is crucial.



### Installation

This project was tested using Python 3.10 with a GPU. However, it is not necessary to have a GPU for the testing
process.
The required dependencies are specified in the `requirements.txt` file.

### Usage

After setting up the repository you may train the model or test on image datasets using the 
checkpoints provided. We have provided the checkpoint for the Mixed - Uncompressed Image dataset.

We provide code for three models as described below.

#### <h2>Deep Image Fingerprint

You can train and test the model for the Deep Image Fingerprint (DIF) model. We have provided the checkpoint for the Mixed 
uncompressed image dataset. You can test any dataset using this or train the model using other datasets.

#### Training the Model

To run `train_dif.py`, you need to specify the data directory and the model directory.
The data directory should include two subdirectories: `train` and `test`. Each of these directories should contain two subdirectories: `0_real` and `1_fake`, for real and fake images, respectively. The
model directory will be used to store the extracted fingerprints.

Example for Mixed Images model:

```
python train_dif.py data/artifact/mixed/train checkpoint/artifact/dif/mixed/uncompressed --e=200 --tr=1024
```

#### Testing the Model

We included models for the Mixed uncompressed image dataset.
In both cases models were trained with 1024 real and 1024 fake images. In addition, we provide all 15 datasets we tested with in the
in `/data/artifact` folder

To reproduce the results per model run `eval_dif.py` and specify fingerprint directory and data directory.
Example for Mixed Images model:

```
python eval_dif.py checkpoint/artifact/dif/mixed/uncompressed data/artifact/mixed/test
```
#### <h2>TransUNet Classifier

You can train and test the model for the TransUNet Classifier model. We have provided the checkpoint for the Mixed 
uncompressed image dataset. You can test any dataset using this or train the model using other datasets.

#### Training the Model

To run `train_trans_unet.py`, you need to specify the data directory and the model directory.
The data directory should include two subdirectories: `train` and `test`. Each of these directories should contain two subdirectories: `0_real` and `1_fake`, for real and fake images, respectively. The
model directory will be used to store the extracted fingerprints.

Example for Mixed Images model:

```
python train_trans_unet.py data/artifact/mixed/train checkpoint/artifact/tran_classifier/mixed/uncompressed --epochs=25 --crop_size=224 --train_size=1024
```

#### Testing the Model

We included models for the Mixed uncompressed image dataset.
In both cases models were trained with 1024 real and 1024 fake images. In addition, we provide all 15 datasets we tested with in the
in `/data/artifact` folder

To reproduce the results per model run `eval_trans_unet.py` and specify fingerprint directory and data directory.
Example for Mixed images model:

```
python eval_trans_unet.py checkpoint/artifact/trans_classifier/mixed/uncompressed data/artifact/mixed/test
```

#### <h2>TransUNet Fingerprint

You can train and test the model for the TransUNet Fingerprint (DIF) model. We have provided the checkpoint for the Mixed 
uncompressed image dataset. You can test any dataset using this or train the model using other datasets. This model does not converge and you
will see that the training loss does not reduce.

#### Training the Model

To run `train_dif_trans_vit.py`, you need to specify the data directory and the model directory.
The data directory should include two subdirectories: `train` and `test`. Each of these directories should contain two subdirectories: `0_real` and `1_fake`, for real and fake images, respectively. The
model directory will be used to store the extracted fingerprints.

Example for Mixed Images model:

```
python train_dif_trans_vit.py data/artifact/mixed/train checkpoint/artifact/trans_fingerprint/mixed/uncompressed --e=200 --tr=1024
```

#### Testing the Model

We included models for the Mixed uncompressed image dataset.
In both cases models were trained with 1024 real and 1024 fake images. In addition, we provide all 15 datasets we tested with in the
in `/data/artifact` folder

To reproduce the results per model run `eval_dif.py` and specify fingerprint directory and data directory.
Example for Mixed images model:

```
python eval_dif.py checkpoint/artifact/trans_fingerprint/mixed/uncompressed data/artifact/mixed/test
```