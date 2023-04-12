# ImageTranslationWithSelfAttention
This repository is the implementation of GAN based image translation model with self-attention module.

# Usage
Install Python 3.7.13, torch==1.11.0, torchvision==0.12.0 <br />
Install package with: <br />
```bash
pip install -r requirements.txt
```
Train the model with the following command:
```bash
python train.py --dataroot image_folder_path --name init_rrff_v1 --model cycle_ff --save_epoch_freq 1 --netG rff
```
# Reference
[1] <a href="url">https://github.com/JoshuaEbenezer/cwgan</a> <br />
[2] <a href="url">https://blog.shikoan.com/sagan-self-attention/</a>
