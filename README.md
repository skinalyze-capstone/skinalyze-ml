# Detection Model

This repository contains the implementation of a detection model using machine learning techniques. Below is an overview of the pipeline implemented in this project.

## Features
- Image preprocessing and dataset preparation.
- Deep learning model creation and training.
- Evaluation of model performance with accuracy and loss visualization.
- Testing the model with unseen images.

## Workflow
1. **Import Libraries**
   - All necessary libraries are imported at the beginning of the notebook.
2. **Dataset Import**
   - The dataset is loaded and prepared for further processing.
3. **Preprocessing**
   - Splitting the dataset into training and validation sets.
   - Ensuring proper data formatting for training.
4. **Modeling**

   
    - The model implemented in this project is a **Sequential** neural network with the following layers:

| Layer (type)                 | Output Shape         | Param #     |
|------------------------------|----------------------|-------------|
| Rescaling (Rescaling)        | (None, 256, 256, 3) | 0           |
| Conv2D (conv2d_6)            | (None, 254, 254, 32)| 896         |
| MaxPooling2D (max_pooling2d_6)| (None, 127, 127, 32)| 0           |
| Conv2D (conv2d_7)            | (None, 125, 125, 64)| 18,496      |
| MaxPooling2D (max_pooling2d_7)| (None, 62, 62, 64)  | 0           |
| Conv2D (conv2d_8)            | (None, 60, 60, 128) | 73,856      |
| MaxPooling2D (max_pooling2d_8)| (None, 30, 30, 128) | 0           |
| Conv2D (conv2d_9)            | (None, 28, 28, 128) | 147,584     |
| MaxPooling2D (max_pooling2d_9)| (None, 14, 14, 128) | 0           |
| Conv2D (conv2d_10)           | (None, 12, 12, 256) | 295,168     |
| MaxPooling2D (max_pooling2d_10)| (None, 6, 6, 256)  | 0           |
| Conv2D (conv2d_11)           | (None, 4, 4, 512)   | 1,180,160   |
| MaxPooling2D (max_pooling2d_11)| (None, 2, 2, 512)  | 0           |
| Flatten (flatten_1)          | (None, 2048)        | 0           |
| Dense (dense_3)              | (None, 512)         | 1,049,088   |
| Dropout (dropout_1)          | (None, 512)         | 0           |
| Dense (dense_4)              | (None, 1024)        | 525,312     |
| Dense (dense_5)              | (None, 6)           | 6,150       |


   - Compiling the model with appropriate loss functions and optimizers.
   - Training the model with callback mechanisms for improved performance.
6. **Evaluation**
   - Plotting accuracy and loss curves to analyze performance.
7. **Testing**
   - Testing the trained model with new images to verify results.

## How to Use
1. Clone the repository.
2. Install the required libraries (refer to the `requirements.txt` file).
3. Run the Jupyter notebook (`Detection_Model.ipynb`) step by step to execute the pipeline.

## Visualizations
Accuracy and loss plots are generated to help analyze the training and validation performance.

## Dependencies
Ensure the following libraries are installed before running the notebook:
- TensorFlow
- NumPy
- Matplotlib
- Other dependencies as specified in the notebook.

## Author
This project was developed as part of Capstone Project Bangkit. Contributions and suggestions are welcome!

