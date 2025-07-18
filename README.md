# ACMMM2024-Oral:M3-Unity
Official repo for '[Deciphering Perceptual Quality in Colored Point Cloud: Prioritizing Geometry or Texture Distortion?](https://openreview.net/forum?id=YE7G4Soi7k)' ACM MM 2024.

## Motivation

<img src="https://github.com/cwi-dis/ACMMM2024-Oral/blob/main/imgs/motivation.jpg" align="left" />

Point clouds represent one of the prevalent formats for 3D content. Distortions introduced at various stages in the point cloud processing pipeline affect the visual quality, altering their geometric composition, texture information, or both. Understanding and quantifying the impact of the distortion domain on visual quality is vital for driving rate optimization and guiding post-processing steps to improve the quality of experience. In this paper, we propose a multi-task guided multi-modality no reference metric (M3-Unity), which utilizes 4 types of modalities across attributes and dimensionalities to represent point clouds. An attention mechanism establishes inter/intra associations among 3D/2D patches, which can complement each other, yielding local and global features, to fit the highly nonlinear property of the human vision system. A
multi-task decoder involving distortion-type classification selects the best association among 4 modalities, aiding the regression task and enabling the in-depth analysis of the interplay between geometrical and textural distortions. Furthermore, our framework design and attention strategy enable us to measure the impact of individual attributes and their combinations, providing insights into how these associations contribute particularly in relation to distortion type. Extensive experimental results on 4 datasets consistently outperform the state-of-the-art metrics by a large margin.


## Framework

<p align="center">
  <img src="https://github.com/cwi-dis/ACMMM2024-Oral/blob/main/imgs/framework.jpg" /> 
</p>

First, we preprocess the colored point cloud and extract multimodal features with 3D and 2D encoders, respectively. Second, we introduce the cross-attributes attentive fusion module, which captures the
local and global associations at both the intra- and inter-modality perception. Last, we employ dual decoders to jointly learn both quality regression and distortion-type classification. The design for this framework is for further analysis, and we have separated the modality and associations to measure the individual contribution to the visual quality.

# How to run the code 

## Environment Build

We train and test the code on the Ubuntu 18.04 platform with open3d and python=3.7. 
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
The GPU is A100 with 48 GB memory,  batchsize = 4.

## Begin training

You can simply train the M3-Unity by referring to train.sh. For example, train M3-Unity on the SJTU-PCQA dataset with the following command:

```
python -u train.py \
--wandbkey "" \ # add your wandb key here otherwize the code will not run successfully
--learning_rate 0.00005 \
--model M3_Unity \
--batch_size  4 \
--database SJTU  \
--data_dir_texture_img path to sjtu_projections/ \
--data_dir_depth_img path to sjtu_depth_maps/ \
--data_dir_normal_img path to sjtu_normal_maps/ \
--data_dir_texture_pc path to sjtu_patch_texture/ \
--data_dir_position_pc path to sjtu_patch_position/ \
--data_dir_normal_pc path to sjtu_patch_normal/ \
--loss l2rank \
--num_epochs 100 \
--k_fold_num 9 \
--use_classificaiton 1 \
--use_local 1 \
--method_label with_dep_nor
```

 **The training data of the projections and patches, will be accessed soon.**  

# Example Visualization
<p align="left">
  <img src="https://github.com/cwi-dis/ACMMM2024-Oral/blob/main/imgs/unicorn_mos.jpg" /> 
</p>

# Anlysis
<p align="center">
  <img src="https://github.com/cwi-dis/ACMMM2024-Oral/blob/main/imgs/ranking_4_datasets.jpg" /> 
</p>

**The most interesting part is by this framework design, we can gain some insights about how the association of texture and geometry impacts the final quality, instead of running an extensive subjective study for point cloud quality assessment, we can estimate the relationship by M3-Unity.**  We have 4 modalities, 6 associations, and we have 6 equations. However, if we want to derive the relationship of the 4 modalities, either we remove two redundant equations or we have more variables.  

(𝑇_2𝐷 • 𝐺_2𝐷)= A  

(𝑇_3𝐷 • 𝐺_3𝐷)= B 

(𝑇_2𝐷 • 𝑇_3𝐷)= C

(𝑇_2𝐷 • 𝐺_3𝐷)= D

(𝐺_2𝐷 • 𝑇_3𝐷)= E  

(𝐺_2𝐷 • 𝐺_3𝐷)= F  

In this paper, we re-define that any combination of 2D and 3D can compose a texture, the same for geometry, which is:  

(𝑇_2𝐷,𝑇_3𝐷)=𝑇′  

(𝐺_2𝐷,𝐺_3𝐷 )=𝐺′

Hence we have the following results:
<p align="left">
  <img src="https://github.com/cwi-dis/ACMMM2024-Oral/blob/main/imgs/Average_Ranking.png" /> 
</p>  

# Future Work
However, each assumption needs to be verified by subjective studies[1]. The simplest assumption is that we just remove the least important associations, and then we solve 4 variables. 


# Bibtex 
-----------
If you find our code useful please cite the paper   
```
@inproceedings{zhou2024deciphering,
  title={Deciphering Perceptual Quality in Colored Point Cloud: Prioritizing Geometry or Texture Distortion?},
  author={Zhou, Xuemei and Viola, Irene and Chen, Yunlu and Pei, Jiahuan and Cesar, Pablo},
  booktitle={ACM Multimedia 2024}
}
```
If you encounter any issues with the code or training dataset, please contact xuemei.zhou@cwi.nl
# Rerference
[1]Lazzarotto, D., Testolina, M., & Ebrahimi, T. (2024). Subjective performance evaluation of bitrate allocation strategies for MPEG and JPEG Pleno point cloud compression. EURASIP Journal on Image and Video Processing , 2024 (1).
