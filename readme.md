# Handwriting text OCR using ANN and Transformer

We have performed handwritten OCR (Optical Character Recognition) using an Artificial Neural Network (ANN) model. It first loads and preprocesses the training and validation data by reading images, resizing them, and encoding labels. Then, it creates an ANN model using Keras, consisting of different layers and fully connected layers, and compiles it with the Adam optimizer and evaluation metrics. The model is trained on the training data and evaluated on the validation data. Performance metrics such as accuracy, precision, recall, and AUC are extracted from the training history. The trained model is saved for future use. The code also includes functions for handwritten text recognition using a trained model. The predict function takes an image and generates the predicted text using our ANN model. We can detect letters of the words using these method since our dataset avaliable is for letters only. So to cope up with this we are using transformers to detect whole sentences and words at once.

## Dataset : https://www.dropbox.com/scl/fo/qgzl8bhvmqp2dkt037f5o/h?rlkey=8juz357wuzw0fm73oaseyzw8d&dl=0

## Steps
1. Importing Libraries: The code starts by importing necessary libraries such as numpy, pandas, matplotlib, seaborn, OpenCV, Keras, scikit-learn, and others. These libraries provide various functionalities for data manipulation, visualization, image processing, and machine learning.

2. Loading and Preprocessing Data:
   - The code defines a directory path for the training data and initializes an empty list called `train_data` to store the training images and corresponding labels.
   - It loops through each subdirectory in the specified directory and reads the images using OpenCV. The images are resized to a specified size (32x32) and appended to the `train_data` list along with their labels.
   - The same process is repeated for the validation data, and the images and labels are stored in the `val_data` list.
   - The training and validation data are then shuffled randomly.
   - The images are normalized by dividing the pixel values by 255.0 and reshaped to a 4-dimensional array.
   - The labels are one-hot encoded using the LabelBinarizer from scikit-learn.

3. Creating the ANN Model:
   - The code defines a function `ANN` that creates the ANN model using Keras.
   - The model consists of different layers followed by fully connected layers.
   - The model is compiled with the Adam optimizer, categorical cross-entropy loss, and various evaluation metrics such as accuracy, precision, recall, and AUC.
   - The summary of the model is printed and saved as `ANN_model.summary()`.


4. Training the ANN Model:
   - The code trains the ANN model using the training data and validation data.
   - The `fit` function is used to train the model for a specified number of epochs and batch size.
   - The training history is stored in the `ANN_history` variable.

5. Performance Metrics and Model Saving:
   - The code defines a function `p_m` to extract performance metrics from the training history.
   - The function takes the history object and the number of epochs as inputs and returns a list of accuracy, precision, recall, and AUC for the last epoch.
   - The function also plots the training history using the `plot_history` function from the `plot_keras_history` library.
   - The function returns the performance metrics.
   - The performance metrics for the ANN model trained for 50 epochs are printed and plotted.
   - The trained ANN model is saved as "ANN_handwriting_det_model.hdf5".

6. **Handwritten Text Recognition**:
   - The code defines a function `predict` to perform handwritten text recognition on an input image.
   - The function takes the image path as input, opens the image using the PIL library, and displays it.
   - The preprocessed image is fed into the model to generate predicted text using the model.
   - The predicted text is printed.
   - We further use transformer for accurate prediction of full sentences and words.

