import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model once and store it in session state
if 'model' not in st.session_state:
    st.session_state.model = load_model('pneumonia_detection_model.h5')

# Function to predict pneumonia
def predict_pneumonia(img):
    model = st.session_state.model
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    result = 'Pneumonia Infection' if prediction > 0.5 else 'Normal Radiograph'
    accuracy = prediction[0][0] if prediction > 0.9 else 1 - prediction[0][0]
    return result, accuracy

# Define the pages
def about_page():
    st.markdown('<h1 style="color:red; text-align:center;"><b>Detecting Pneumonia</b></h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:royalblue;"><u><b>Pneumonia in Brief </b></u></h2>', unsafe_allow_html=True)

     # Display local images
    pneumonia_infograph = './pneumonia.jpg'

    st.image(pneumonia_infograph, caption='Pneumonia', use_column_width=True)


    st.markdown("""

    <h4 style="color:royalblue;">What is Pneumonia?</h4> 

    Pneumonia is an infection that `inflames the air sacs` in one or both lungs, which `may fill with fluid or pus`, causing symptoms such as cough, fever, and difficulty breathing.
    - It may be caused by infection with certain viruses or bacteria and less commonly by other microorganisms, certain medications or conditions such as autoimmune diseases.
    <br><br>

    <h4 style="color:royalblue;">Epidemiology</h4>
    
    Pneumonia affects people of all ages but is particularly dangerous for `young children`, `the elderly`, and those with `weakened immune systems`.
    <br><br>

    <h4 style="color:royalblue;">Signs and Symptoms</h4> 
    Common signs include: 
    
    - _cough with phlegm_, 
    - _fever_, 
    - _chills_, 
    - _difficulty breathing_, and 
    - _chest pain_.
    
    <br><br>

    <h4 style="color:royalblue;">Diagnosis</h4> 

    Pneumonia is typically diagnosed using a combination of physical examination, medical history, and imaging tests like chest X-rays.
    <br><br>

    <h4 style="color:royalblue;">Treatment:</h4>
    Treatment often involves:
    1. antibiotics or antiviral medications, and 
    2. supportive care. 
    <br>
    Consult your physician for a proper diagnosis and treatment plan.
    <br><br>

    <h4 style="color:royalblue;">Use of Chest X-rays:</h4>
    Chest X-rays are used to diagnose pneumonia by identifying signs of infection or inflammation in the lungs. 
    They can help distinguish pneumonia from other respiratory conditions.

    """, unsafe_allow_html=True)


def chest_xray_page():
    st.markdown("<h1 style='color:green; text-align:center;'><b>Upload a Chest X-ray</b></h1>", unsafe_allow_html=True)
    st.subheader('Upload a Chest X-ray to Detect Pneumonia')

    st.write("""
    **Welcome to the Chest X-ray Pneumonia Detection page.** Here you can upload a chest X-ray image to check if it suggests pneumonia.

    **Sample X-rays:**
    """)

    # Display local images
    normal_xray_path = 'images/NORMAL2-IM-1427-0001.jpeg'  
    pneumonia_xray_path = 'images/person1946_bacteria_4874.jpeg'

    
    st.image(normal_xray_path, caption='A Normal Chest X-ray', use_column_width=True)
    st.image(pneumonia_xray_path, caption='A Chest X-ray with Pneumonia', use_column_width=True)
    
    # File uploader for user input
    uploaded_file = st.file_uploader("Select a Chest X-ray image to upload...", type="jpg")
    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(150, 150))
        result, accuracy = predict_pneumonia(img)
        
        st.image(img, caption='Uploaded Image', use_column_width=True)

        st.write(f"We Predict Findings Characteristic of a: <h6 style='color:red;'><b><u>{result}</u></b></h6>", unsafe_allow_html=True )

        st.write(f"The Accuracy of Predicting these Findings Based on your Radiograph is <h6 style='color:royalblue;'><b>{accuracy*100:.2f}%</b></h6>", unsafe_allow_html=True)

# def read_xray_page():
  #  st.title('Interpreting a Chest X-ray')

    


# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'About'




# Menu bar and page routing
st.sidebar.title('Menu')
selection = st.sidebar.radio("Pneumonia", ["About", "Upload X-ray"]) #, "Interpret X-ray"

# Initialize session state if not already initialized
if 'page' not in st.session_state:
    st.session_state.page = 'About'

# Update page state based on sidebar selection
if selection == "About":
    st.session_state.page = 'About'
elif selection == "Upload X-ray":
    st.session_state.page = 'Upload X-ray'
# elif selection == "Reading X-ray":
   # st.session_state.page = 'Interpret X-ray'

# Display the selected page
if st.session_state.page == 'About':
    about_page()
elif st.session_state.page == 'Upload X-ray':
    chest_xray_page()
elif st.session_state.page == 'Interpret X-ray':
    read_xray_page()

