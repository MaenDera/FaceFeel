from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import base64
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)

# --- Configuration Parameters ---
# Path to the pre-trained Keras model file
MODEL_PATH = 'last_model_test.keras'
# Target size for resizing images before feeding to the model (height, width)
TARGET_SIZE = (238, 238)
# List of class names (emotions) that the model predicts
CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad', 'surprise']

# --- Model and Face Detector Loading ---
# Load the pre-trained TensorFlow Keras model into memory once when the app starts.
tf_model = load_model(MODEL_PATH)
# Construct the full path to the OpenCV Haar Cascade XML file for face detection.
# 'cv2.data.haarcascades' provides the path to the installed Haar cascade files.
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' # type: ignore
# Initialize the Haar Cascade classifier for face detection.
face_cascade = cv2.CascadeClassifier(cascade_path)


# --- Utility Functions ---
def encode_image_to_data_uri(image_bgr, ext='.png'):
    """
    Encodes a given BGR image (Numpy array) into a base64 data URI.
    This is useful for embedding images directly into HTML or JSON responses.

    Args:
        image_bgr (np.array): The image in BGR format (OpenCV default).
        ext (str): The file extension for encoding (e.g., '.png', '.jpg').

    Returns:
        str: A data URI string (e.g., 'data:image/png;base64,...') or None if encoding fails.
    """
    # Encode the image into a byte buffer using the specified extension.
    success, buf = cv2.imencode(ext, image_bgr)
    if not success:
        return None  # Return None if encoding was not successful
    img_bytes = buf.tobytes()  # Convert the buffer to bytes
    # Encode bytes to base64 and decode to UTF-8 string, then prepend data URI header.
    return 'data:image/png;base64,' + base64.b64encode(img_bytes).decode('utf-8')

def detect_all_faces(image_bytes):
    """
    Detects all faces in an image provided as bytes, crops them, and returns
    a list of (data_uri, cropped_image_bgr) tuples for each detected face.

    Args:
        image_bytes (bytes): The raw bytes of the image file.

    Returns:
        list: A list of tuples, where each tuple contains:
              - data_uri (str): Base64 data URI of the cropped face.
              - crop (np.array): The cropped face image in BGR format.
    """
    # Convert image bytes to a NumPy array
    np_img = np.frombuffer(image_bytes, np.uint8)
    # Decode the NumPy array into a BGR image (OpenCV default)
    img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    # Convert the BGR image to grayscale for face detection (Haar cascades work best on grayscale)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image.
    # scaleFactor: How much the image size is reduced at each image scale.
    # minNeighbors: How many neighbors each candidate rectangle should have to retain it.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    results = []
    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Crop the face region from the original BGR image
        crop = img_bgr[y:y+h, x:x+w]
        # Encode the cropped face to a data URI
        uri = encode_image_to_data_uri(crop)
        # Add the data URI and the cropped image (BGR) to the results list
        results.append((uri, crop))
    return results

def preprocess_for_model(image_bgr):
    """
    Preprocesses an image (either a full image or a face crop) for input into the
    TensorFlow Keras model. This involves converting BGR to RGB, resizing,
    converting to an array, and normalizing pixel values.

    Args:
        image_bgr (np.array): The image in BGR format (OpenCV default).

    Returns:
        np.array: A NumPy array ready for model prediction, with batch dimension.
    """
    # Convert BGR image to RGB (TensorFlow/Keras models typically expect RGB)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Convert the NumPy array (RGB) to a PIL Image, then ensure it's in 'RGB' mode.
    pil_img = Image.fromarray(img_rgb).convert('RGB')
    # Resize the PIL image to the target size required by the model
    pil_img = pil_img.resize(TARGET_SIZE)
    # Convert the PIL image back to a NumPy array
    x = img_to_array(pil_img)
    # Normalize pixel values to be between 0 and 1 (common for neural networks)
    x /= 255.0
    # Add a batch dimension to the array (models expect input in the shape [batch_size, height, width, channels])
    return np.expand_dims(x, axis=0)

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

def make_predictions_on(image_bgr):
    """
    Given a BGR image (either a whole image or a face crop),
    runs the loaded TensorFlow model to predict emotions.

    Args:
        image_bgr (np.array): The image in BGR format (OpenCV default).

    Returns:
        dict: A dictionary containing:
              - 'face' (str): Base64 data URI of the input image.
              - 'prediction' (str): The predicted emotion label.
              - 'probabilities' (dict): A dictionary of probabilities for each emotion class.
    """
    # Encode the input image to a data URI for display in the response
    uri = encode_image_to_data_uri(image_bgr)
    # Preprocess the image to prepare it for the model
    x = preprocess_for_model(image_bgr)
    # Make a prediction using the loaded model. [0] is used to get the single prediction array
    preds = tf_model.predict(x)[0]
    # Get the label (emotion) with the highest probability
    label = CLASS_NAMES[int(np.argmax(preds))]
    # Format probabilities as a dictionary with class names and float values (rounded to 4 decimal places)
    probs = {cls: float(f'{p:.4f}') for cls, p in zip(CLASS_NAMES, preds)}
    # Return the results as a dictionary
    return {'face': uri, 'prediction': label, 'probabilities': probs}

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """
    Renders the main index.html page when a GET request is made to the root URL.
    This page typically contains the image upload form.
    """
    return render_template('index.html')

@app.route('/research', methods=['GET'])
def research():
    """
    Renders the research.html page when a GET request is made to the /research URL.
    This page might contain information about the model or research behind it.
    """
    return render_template('research.html')

@app.route("/privacy")
def privacy():
    """
    Renders the privacy.html page when a GET request is made to the /privacy URL.
    This page would typically outline the application's privacy policy.
    """
    return render_template("privacy.html")

@app.route('/', methods=['POST'])
def predict():
    """
    Handles POST requests to the root URL, expecting an image file upload.
    It performs emotion prediction either on the full image or on detected faces,
    depending on the 'crop_face' flag.
    """
    # Get the uploaded image file from the request
    file = request.files.get('image')
    # Get the 'crop_face' checkbox value from the form.
    # It will be 'on' if checked, otherwise 'off'. Convert to boolean.
    crop_flag = request.form.get('crop_face', 'off') == 'on'

    # Check if an image file was provided
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    # Read the image file content into bytes
    image_bytes = file.read()
    results = []  # Initialize a list to store prediction results

    if crop_flag:
        # If 'crop_face' is enabled, detect all faces in the image
        face_list = detect_all_faces(image_bytes)
        if not face_list:
            # If no faces are detected, return an error
            return jsonify({'error': 'No faces detected'}), 400

        # Iterate through each detected face and run prediction
        for uri, face in face_list:
            results.append(make_predictions_on(face))
    else:
        # If 'crop_face' is not enabled, run prediction on the full image
        # Convert image bytes to a NumPy array
        np_img = np.frombuffer(image_bytes, np.uint8)
        # Decode the NumPy array into a BGR image
        orig_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        results.append(make_predictions_on(orig_bgr))

    # Return the prediction results as a JSON response
    return jsonify({'results': results})

# --- Application Entry Point ---
if __name__ == '__main__':
    # Run the Flask application.
    # In a production environment, you would typically use a production-ready WSGI server
    # like Gunicorn or uWSGI instead of app.run().
    app.run()
