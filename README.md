# ASAP-Internship-Project
Astronomical Data Processing

# Galaxy Classification using AlexNet-like CNN

## Project Overview

This project demonstrates the implementation of a Convolutional Neural Network (CNN), inspired by the AlexNet architecture, to classify galaxy images into two distinct morphological types: 'Round' and 'Edge-on'. The goal is to build a deep learning model capable of accurately distinguishing between these two classes of galaxies based on their visual characteristics.

## Methodology

The project follows a standard machine learning pipeline:

1.  **Data Acquisition and Extraction:**
    *   The `galaxy-zoo-the-galaxy-challenge` dataset was downloaded from Kaggle using the Kaggle API.
    *   Raw image and solution files were extracted from `.zip` archives.

2.  **Data Preparation:**
    *   The `training_solutions_rev1.csv` file, containing galaxy classification probabilities, was loaded using `pandas`.
    *   High-confidence 'Round' galaxies (Class1.1 > 0.9) and 'Edge-on' galaxies (Class2.1 > 0.9) were identified.
    *   A balanced subset of 757 images for each class was randomly sampled and copied into dedicated directories (`subset_data/Round`, `subset_data/EdgeOn`).
    *   `ImageDataGenerator` from `tensorflow.keras` was used to:
        *   Rescale pixel values to the range [0, 1].
        *   Split the data into 80% for training and 20% for validation.
        *   Resize images to 227x227 pixels, matching the AlexNet input requirement.
        *   Generate batches of images for training and validation.

3.  **Model Definition (AlexNet-like CNN):**
    *   A sequential Keras model was constructed, mirroring the AlexNet architecture.
    *   It comprises multiple `Conv2D` layers with ReLU activation, interleaved with `MaxPooling2D` layers for spatial downsampling.
    *   A `Flatten` layer connects the convolutional base to fully connected (`Dense`) layers.
    *   Two `Dense` layers with 4096 units each, followed by `Dropout` (0.5) layers, provide high-level feature interpretation and mitigate overfitting.
    *   The output layer is a `Dense` layer with 2 units (for the two classes) and a `softmax` activation function.

4.  **Model Training:**
    *   The model was compiled with the `adam` optimizer, `categorical_crossentropy` loss function, and `accuracy` as the evaluation metric.
    *   Training was performed for 10 epochs using the `train_generator` and `validation_generator`.

5.  **Model Evaluation:**
    *   Training and validation accuracy and loss curves were plotted over the epochs to visualize the model's learning progress and identify potential issues like overfitting.

## System Architecture

The core of this project is a deep Convolutional Neural Network (CNN) designed with an architecture similar to AlexNet. Key components include:

*   **Input Layer:** Accepts 227x227x3 RGB image data.
*   **Convolutional Blocks:** A series of `Conv2D` layers (e.g., 96 filters, 11x11 kernel, stride 4; 256 filters, 5x5 kernel, etc.) with `ReLU` activation for hierarchical feature extraction.
*   **Max Pooling:** Used after convolutional layers to reduce dimensionality and increase translation invariance.
*   **Fully Connected Layers:** Two `Dense` layers, each with 4096 neurons and `ReLU` activation, process the flattened features.
*   **Dropout:** Applied to the fully connected layers to prevent overfitting.
*   **Output Layer:** A `Dense` layer with 2 units and `softmax` activation for binary classification probabilities.

## Implementation Details

*   **Libraries:** `pandas` for data handling, `os` and `shutil` for file system operations, `tensorflow.keras` for model building and training, and `matplotlib` for plotting.
*   **Kaggle Integration:** The Kaggle API was used directly in the Colab notebook to download the dataset.
*   **Data Preparation Script:** Custom Python code was written to filter galaxies based on confidence scores and copy images into class-specific directories, ensuring a balanced dataset for training.
*   **`ImageDataGenerator`:** Central to efficient data loading and augmentation, providing normalized and resized image batches for the model.
*   **Model Definition:** The AlexNet architecture was meticulously constructed layer-by-layer within a `create_alexnet` function.
*   **Training Loop:** The model was trained using the `fit` method with specified epochs and validation data.
*   **Visualization:** `matplotlib` plots display the model's performance metrics (accuracy and loss) over the training epochs.

## How to Run

1.  **Kaggle API Key Setup:**
    *   Download your `kaggle.json` file from your Kaggle account (Profile -> Account -> Create New API Token).
    *   Upload `kaggle.json` to your Colab environment in the first code cell.
    *   Ensure proper permissions are set (`!chmod 600 ~/.kaggle/kaggle.json`).
2.  **Execute Cells Sequentially:** Run all code cells in the notebook from top to bottom.
    *   The first cell will download and extract the dataset.
    *   Subsequent cells will handle data preparation, model definition, training, and visualization.

## Results


The training history plots (Accuracy and Loss) at the end of the notebook provide insights into the model's performance. These plots indicate how well the model learned on the training data and generalized to the unseen validation data over 10 epochs.
