# Face Spoofing Detection based on Diffuse Reflection for Web Application

## Paper

## Overview
Spoofing detection plays an important role of biometrics system, especially if the system uses face modality. Here, we proposed an efficient face presentation attack detection (PAD) algorithm that requires minimal hardware and low compute, making it suitable for servers-side. Utilizing one monocular visible light camera, the algorithm takes two facial photos, one of which is taken with flash and other without flash. The proposed descriptor is constructed by leveraging diffuse reflection that will spread throughout the face region and will form the 3D structure of a subject’s face. The structure pattern will be classified with SVM to distinguish between real and spoof.

## Model
[SVM Model](https://drive.google.com/drive/u/0/folders/1zE7ar6ZP1CSY2YRb76JWwZ4LEUj5yiG5)

## Dataset
[SpecDiff Dataset](https://github.com/Akinori-F-Ebihara/SpecDiff_in_house_database_sample)

## How to Use
1. Web Application: Run notebook `Web_Anti_Spoofing.ipynb` in  
