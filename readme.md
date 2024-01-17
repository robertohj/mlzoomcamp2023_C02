# **Animal classification  - ML Zoomcamp Capstone Project**
TL;DR:  
This project resulted in a streamlit application that can be accessed following in  **https://animalprediction.streamlit.app/**   This app uses a deep learning model to classify animals; the model is serving from the cloud in Azure.
NOTE: the first time the app is run from the cloud it may take some time while the virtual hardware is provisioned.

## Problem description
In this project, we will use Convolutional Neural Networks to classify animals in JPEG images.  

The prediction of the class will use a CNN based on EfficientNetB3 with fine tuning on learning rate, dropout, batch normalization and data augmentation.  The project is divided into the following main stages:
1. Introduction  
1.1 Using the base, pretrained model  
1.2 Making predictions using the pretrained model  
2. Training and checkpointing  
2.1 Making the model  
2.2 Transfer learning  
2.3 Fine tuning on learning rate, dropout, batch normalization and data augmentation  
2.4 Training on a larger dataset  
2.4 Data augmentation  
3. Model evaluation (Best models)  
3.1  Accuracy  
3.2  Model with no augmentation  
3.3  Precision, Recall, F1 and classification report
4. Using the model  
4.1 Loading the model  
4.2 Getting predictions  
4.3 Testing on completely random, difficult imgages  
5. Conclusion


### Dataset  
The dataset was obtained from Kaggle and can be found here:  
https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals/data  

The dataset contains 5400 Animal Images in 90 different categories obtained from Wikipedia.  

The most obvious way to apply the dataset is in multiclass classification of animals, though there are many different approaches to achieve this, most of them using transfer learning. One basic difference is the base pretrained model to extend from, such as Xception (used in the course), or EfficientNetBx (used in this project).

For reference, although Xception is a good Deep Learning architecture, [EfficientNet](https://pypi.org/project/efficientnet/) has become the go-to architecture for many challenging tasks, particularly in object recognition applications.  
  
See more: https://blog.roboflow.com/what-is-efficientnet/#:~:text=EfficientNet%20addresses%20this%20challenge%20by,of%20efficiency%20without%20compromising%20accuracy.  

<img src="https://lh5.googleusercontent.com/8DT6a-PxS3doi3J-72hotDhWfS1BlHlR48Y2OCQV_i9gyGC2F-i9E46lV7Kx8gzZFMt6YlGEIbWgqzORo7aowNYrY0YAK8EXeFzZ3g2nIf9GZG-CRcz7AxIuGEH1YEm03fyUcXOaX1_PLpo15DRzPCk"></img>

## Jupyter Notebooks  
Several jupyter notebooks were created in this project. 
In the folder local_deployment, The *c01_animal_classification.ipynb* file is the Notebook where training, transfer learning, evaluation and testing were done. More details can be seen inside. Hint: you can use the "OUTLINE" panel if you are using VSCode to easily navigate through the sections marked with MD language, in the notebook.  Additionally, the model was tested in several ways, as can be seen in the notebooks *c01_animal_classification_PREDICT_with_flask (test).ipynb* and *predict_on_the_cloud (test).ipynb*. The other notebook, *keras to tf_lite conversion.ipynb*, shows the process of converting the keras model to a tflite model, which was the final version of the model used in deployment.   

## Training and Testing scripts  
The training and testing sections from the notebook were 
exported as separate .py scripts to be executed independently and at will. Such files are:  
 + train.py
 + predict.py
 + predict_tflite_test.py  
 + predict_tflite_flask.py 

### Evidence of execution:  
The final best model was tested in several experiments, and evidence of that local execution can be found in the folder called *execution* inside *local_deployment*

## Model deployment  
The model was deployed using Flask in the file predict_tflite_flask.py  and the model was also containerized using docker. 

## Environment  
A .venv virtual environment was used to isolate the packages used in this project. In that environment, the corresponding modules were installed using pip. The modules and their version numbers were dumped into the requirements.txt file.  Please note that the requirements file inside a given folder servers a different purpose (local testing, containerization and cloud deployment).

The corresponding environment inside a given folder can be recreated by using the following command:  
```python -m venv -r requirements.txt```  

## Production  
Waitress was used to build the model production-ready and serve. This package was also used to launch the API from its Docker container. 

## Containerization  
A container was used in Docker to isolate the software package and dependencies in the OS. The docker file Dockerfile is provided with the corresponding configuration that sets the production environment, installs dependencies and launches the Flask application to serve in production using Waitress, as Windows was used in this project.  

## Interaction  
In the notebook, the folder *execution* provides evidence of the configuration, testing and production of the model. The final product of this project is a website that serves the tflite model from the cloud through a streamlit application hosted as https://animalprediction.streamlit.app/. NOTE: the first time the model is run from the cloud it may take some time while the virtual hardware is provisioned.

## Cloud deployment  
The model was deployed to the cloud using an Azure function. All the model files, the scripts, requirements and configuration files can be found in the *azure_function* folder.  

## Final product  
The final product is a streamlit application that can be reproduced locally using the predict_app.py script, and was also deployed to streamlit so it can be accessed following this link:  https://animalprediction.streamlit.app/  NOTE: the first time the model is run from the cloud it may take some time while the virtual hardware is provisioned.








