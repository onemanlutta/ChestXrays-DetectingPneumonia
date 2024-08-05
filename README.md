# ChestXrays-DetectingPneumonia

![image](https://github.com/user-attachments/assets/677ec393-ccbf-4346-a929-b880a7ad80ad)


## Background

This project involves the development of a deep-learning model to detect pneumonia from chest X-ray images. The model was trained using a convolutional neural network (CNN) approach to classify images of chest radiographs as showing signs of pneumonia or normal.


### Steps to Develop the Model

1. **Data Collection:**
   - The dataset was sourced from publicly available medical image repositories. It includes labeled chest X-ray images indicating whether the image shows signs of pneumonia.

2. **Data Preprocessing:**
   - Images were resized to a consistent size (150x150 pixels).
   - Data augmentation techniques were applied to increase the variability of the training data and prevent overfitting.

3. **Model Architecture:**
   - The CNN model was built with several convolutional layers to extract features from the images, followed by pooling layers to reduce dimensionality.
   - A fully connected dense layer was added for classification, and dropout was used to prevent overfitting.

4. **Model Training:**
   - The model was compiled with the Adam optimizer and binary cross-entropy loss function.
   - Training involved using early stopping and learning rate reduction callbacks to optimize performance.

5. **Model Evaluation:**
   - The model was evaluated on a separate test set, achieving high accuracy and providing detailed classification metrics.

---
## Streamlit Application

![image](https://github.com/user-attachments/assets/a05dfd76-6e9e-4537-a20e-ff180afd2944)


### Application Overview
The ChestXrays-DetectingPneumonia application uses a machine learning model to detect pneumonia from chest X-ray images. The app is designed to be user-friendly and provides accurate results along with an easy-to-understand interpretation. 

### Repository and Deployment
- **GitHub Repository:** [ChestXrays-DetectingPneumonia](https://github.com/onemanlutta/ChestXrays-DetectingPneumonia)
- **Deployed Application:** [Streamlit App](https://chestxraysdetectingpneumonia.streamlit.app/)

### How to Use the Application
1. **Access the App:**
   - Visit the deployed application [here](https://chestxraysdetectingpneumonia.streamlit.app/).
   
2. **Navigate Through the App:**
   - Use the sidebar to navigate between the 'About' page and the 'Upload X-ray' page.
   - The 'About' page provides information on pneumonia, its diagnosis, and treatment.
   - The 'Upload X-ray' page allows you to upload a chest X-ray image for pneumonia detection.

3. **Upload a Chest X-ray:**

![image](https://github.com/user-attachments/assets/6364124a-8b3f-4734-93cb-2f0492341e18)


   - On the 'Upload X-ray' page, you can upload a chest X-ray image by clicking the "Select a Chest X-ray image to upload..." button.
   - Supported file type: `.jpg` and `.jpeg`
   - Once uploaded, the app will display the image and provide a prediction indicating whether the X-ray suggests 'Pneumonia Infection' or 'Normal Radiograph'.
   - The app also displays the accuracy of the prediction.

![image](https://github.com/user-attachments/assets/d5cda8bd-9dd1-48b5-8b73-205816edf700)


### Understanding the Model's Capabilities
- **Model Used:** The app uses a pre-trained convolutional neural network (CNN) model to detect pneumonia.
- **Prediction Interpretation:**
  - The result will be either 'Pneumonia Infection' or 'Normal Radiograph' based on the prediction.
  - The app provides the accuracy of the prediction, giving users confidence in the results.

### About Page Details
The 'About' page provides detailed information about pneumonia, including:
- What pneumonia is
- Epidemiology
- Signs and symptoms
- Diagnosis
- Treatment
- Use of chest X-rays in diagnosing pneumonia

### Sample X-ray Images
The app includes sample X-ray images to help users understand what normal and pneumonia-affected X-rays look like:
- **Normal Chest X-ray**
- **Chest X-ray with Pneumonia**

### Contact and Support
For any issues or questions, please open an issue in the [GitHub repository](https://github.com/onemanlutta/ChestXrays-DetectingPneumonia/issues).

---

By following this documentation, users can effectively navigate and utilize the ChestXrays-DetectingPneumonia application to detect pneumonia from chest X-ray images.


### Capabilities

- **Real-time Predictions:** Provides immediate results upon image upload.
- **User-Friendly Interface:** Simple and intuitive interface for easy interaction.

### Limitations

- **Model Accuracy:** Performance can vary based on image quality and characteristics.
- **Not a Substitute for Medical Advice:** The app is intended as a supplementary tool and should not replace professional medical evaluation.


---

## Conclusion and Recommendations

The Pneumonia Detection App demonstrates the capabilities of deep learning models in medical image analysis. For best results:
- Ensure that the uploaded images are of high quality and properly aligned.
- Use the app as a preliminary tool and consult healthcare professionals for an accurate diagnosis.

## Project Directories

- `app.py`: Main Streamlit application script.
- `model/`: Contains the trained model file (`pneumonia_detection_model.h5`).
- `requirements.txt`: Lists the dependencies required to run the app.
- `images/`: Includes sample images.


## References

- **Dataset Source:** The dataset used for training is from [source link].
- **Model Details:** Detailed information on model architecture and training can be found in [this documentation or research paper].

## Disclaimer(s)

- **Accuracy Disclaimer:** The app's predictions are based on a machine learning model and may not be 100% accurate.
- **Medical Disclaimer:** The application is not intended to replace professional medical advice, diagnosis, or treatment.


## Acknowledgments

- TensorFlow and Keras for providing the deep learning framework.
- Streamlit for the web application development platform.
- Contributors and data providers for their invaluable resources.
- [African Centre for Data Science and Analytics Ltd](https://africdsa.com/)

