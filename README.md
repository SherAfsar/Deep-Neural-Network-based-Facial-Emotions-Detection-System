
# Facial Emotions Detection System Using DNN
This project presents two deep neural network architectures for emotion detection, trained on the Cohn-Kanade (CK+) dataset: a Baseline CNN and an Attention CNN with SVM and RF classifiers. The Baseline CNN, evaluated using K-Fold cross-validation, achieved a validation accuracy of 92.78% on Fold-2. The Attention CNN, assessed with stratified K-Fold cross-validation, reached 97.93% accuracy with the SVM classifier and 97.94% with the RF classifier on Fold-4, showing nearly identical performance due to their similar architectures.
## Technologies used
- Numpy
- Pandas
- Matplotlib
- Tensor Flow
- Keras
- Scikit-learn
- And others

## Dataset
Both neural networks are trained on Cohn-Kanade (Ck+)dataset. 
https://www.kaggle.com/datasets/shawon10/ckplus 

## Dataset Preprocessing
The dataset preprocessing for emotion detection involved several steps: 

- First, the list of images was obtained using the listdir() function to gather the full paths of all images. 

- Next, OpenCV was used to read these images.

- The images were then resized to a uniform dimension of 48x48 pixels to ensure consistency and reduce noise.

- Each pixel value was normalized to a range between 0 and 1 by dividing by 255 to standardize the intensity values. 

- Data augmentation techniques were applied to increase the variability of the dataset, making the model more generalizable.

- Labels were assigned to each image based on their respective classes, and these labels were converted into a one-hot encoding format to avoid unintended ordinal relationships. 

- Finally, the dataset was shuffled with a fixed random state to ensure consistent shuffling across different runs, preventing the model from learning patterns based on the sequence of the dataset and improving its performance on unseen data.

## Model Creation
For the creation of model, following steps were taken: 

- The Baseline CNN model for emotion detection takes a 3D array input of image dimensions and channels, constructed sequentially with each layer's output serving as the next layer's input.

- It features a first 2D convolutional layer with six 5x5 filters using ReLU activation and zero-padding, followed by a 2x2 MaxPooling2D layer. 

- This is followed by a second convolutional layer with sixteen 5x5 filters and another MaxPooling2D layer. 

- The third convolutional layer has sixty-four 3x3 filters, also followed by MaxPooling2D.

- A Flatten layer converts the 3D output into a 1D vector, feeding into a fully connected layer with 128 neurons, then another with seven neurons for classification. 

- A Dropout layer with a 0.5 rate prevents overfitting. The output layer uses the SoftMax function for class probabilities, employing the categorical_crossentropy loss function and RMSProp optimizer to minimize errors and improve predictions.

## Model Training
The model training section covers the details of training parameters and techniques. 

- The batch size is set to 16, and the number of epochs to 100, both determined through hyperparameter tuning.

- A 5-fold cross-validation technique is used, dividing the dataset into five parts, each serving as the test set once while the rest are used for training, with each fold iterating 100 times.

- Two callbacks are employed: Model Checkpoint, which saves the best model weights based on the lowest training loss, and Early Stopping, which halts training if the accuracy does not improve for eight consecutive epochs.

- The model is trained using the fit_generator() method, which augments training data batches via aug.flow().
  
## Performance Evaluation
To evaluate the performance of both architectures. Different techniques are employed:

- For Baseline CNN, K-Fold cross validation technique is used.

- For Attention-CNN, stratified K-Fold cross validation is utilized.

- The former achieved validation accuracy around 92.78% on Fold-2. The latter achieved validation accuracy of 97.93% with Support Vector Machine (SVM) classifier on Fold-4, and with Random Forest (RF) it stood at 97.94%.

## Results
Please find attached the PowerPoint presentation containing all the results obtained during the implementation of this project.

## Jupyter Notebooks
I've uploaded two .py files: one comprising code for a Convolutional Neural Network, and the other containing code for a CNN with an integrated Attention mechanism

## 🚀 About Me
Highly skilled Data Analyst with 2 years of experience in leveraging data-driven insights to inform strategic decision-making. Proficient in SQL, Python, and advanced Excel, with a strong background in data visualization tools such as Tableau. Demonstrated expertise in data cleaning, analysis, and interpretation to support business objectives and drive efficiency. Adept at translating complex datasets into actionable insights and effective business solutions. Passionate about continuous learning and staying updated with industry trends to deliver innovative analytical solutions. 

Feel free to connect on LinkedIn

LinkedIn Profile: https://www.linkedin.com/in/sherafsardataanalyst/



