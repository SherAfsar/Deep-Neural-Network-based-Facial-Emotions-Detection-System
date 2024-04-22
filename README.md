# Introduction
Facial Emotion Detection System is created using Convolutional Neural Network and Attention mechanism integrated Convolutional Neural Network called Deep-Attention CNN. 
# Dataset
Both neural networks are trained on Cohn-Kanade (Ck+)dataset. https://www.kaggle.com/datasets/shawon10/ckplus 
# Data Preprocessing
The following data preprocessing steps were employed: resizing images, normalization of pixels, data augmentation, one-hot encoding etc. 
# Performance Evaluation
To evaluate the performance of both architectures. Different techniques are employed.For Baseline CNN, K-Fold cross validation technique is used. For Attention-CNN, stratified K-Fold cross validation is utilized.The former achieved validation accuracy around 92.78% on Fold-2. The 
latter achieved validation accuracy of 97.93% with Support Vector Machine (SVM) classifier on 
Fold-4, and with Random Forest (RF) it stood at 97.94%.
# Results
Please find attached the PowerPoint presentation containing all the results obtained during the implementation of this project.
