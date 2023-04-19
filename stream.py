import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from models.vision import *
import warnings
warnings.filterwarnings('ignore')

st.title('Facial Expression Recognition using Webcam')
st.write('This app uses a pre-trained ResNet model to detect facial expressions in real-time using your webcam.')

model_name = st.sidebar.selectbox('Select a model:', ('resnet_attention', 'resnet_multi_attention'))
face_cascade_path = st.sidebar.text_input('Enter the path to the face detection model:', 'saved_models/haarcascade_frontalface_default.xml')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model based on the user input
if model_name == 'resnet_attention':
    model = ResNetWithAttention().to(device)
    model_path = 'saved_models/model_attention_pre_1.pt'
elif model_name == 'resnet_multi_attention':
    model = ResNetWithMAttention().to(device)
    model_path = 'saved_models/model_attention_2.pt'
else:
    st.write('Invalid model choice!')
    st.stop()

# Load the saved model
loaded_model = torch.load(model_path)
loaded_model = loaded_model.to(device)
loaded_model.eval()

# Define the transformations
def resize_image(image):
    return cv2.resize(image, (112, 112))

def to_tensor(image):
    return transforms.ToTensor()(image)

def normalize(image):
    mean = [0.5079, 0.5079, 0.5079]
    std = [0.2563, 0.2563, 0.2563]
    return transforms.Normalize(mean=mean, std=std)(image)

# Define the transformations to be applied to each frame
def apply_transformations(image):
    image = resize_image(image)
    image = to_tensor(image)
    image = normalize(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image

# Define the main function
def run_app():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    # Load the face detection model
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Define the transformations to be applied to each frame
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5079, 0.5079, 0.5079],
                            std=[0.2563, 0.2563, 0.2563])
    ])

    st.title("Facial Expression Recognition using Webcam")
    st.write("This app uses a deep learning model to recognize facial expressions in real-time using your webcam.")
    st.write("To start, simply click the button below:")

    # Create a button to start the webcam
    if st.button("Start"):
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Crop the face from the frame
                face = frame[y:y+h, x:x+w]

                # Convert the face to a PIL image
                pil_image = Image.fromarray(face)

                pil_image = pil_image.convert('L')
                # Convert the image to a numpy array
                np_image = np.array(pil_image)
                # Create a 3-channel image by repeating the grayscale values
                np_image = np.repeat(np_image[:, :, np.newaxis], 3, axis=2)
                # Convert the numpy array back to a PIL image
                pil_image = Image.fromarray(np_image)

                # Apply the transformations to the PIL image
                tensor_image = transform(pil_image)

                # Add a batch dimension to the tensor image
                tensor_image = tensor_image.unsqueeze(0)

                # Move the tensor image to the device
                tensor_image = tensor_image.to(device)

                # Make a prediction on the tensor image
                with torch.no_grad():
                    output = loaded_model(tensor_image)
                    _, predicted = torch.max(output, 1)
                    predicted_class = predicted.item()

                # Display the predicted class on the frame
                cv2.putText(frame, str(classes[predicted_class]), (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # Convert the frame from OpenCV BGR to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in Streamlit
            st.image(frame, channels='RGB')

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture and destroy the window
        cap.release()
        cv2.destroyAllWindows()


run_app()