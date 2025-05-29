# FaceFeel Flask App

This is a web application built with Flask that allows users to upload an image and get emotion predictions using a pre-trained deep learning model. The application can either predict emotions on the entire image or detect faces and predict emotions for each detected face.

---

## Features

* **Image Upload:** Easily upload images through a web interface.
* **Emotion Prediction:** Utilizes a Keras model to classify emotions (angry, happy, neutral, sad, surprise).
* **Face Detection:** Optionally detects multiple faces within an image and provides individual emotion predictions for each.
* **Probability Scores:** Displays the probability scores for each emotion class.
* **Simple Interface:** User-friendly web interface for interaction.

---

## Technologies Used

* **Flask:** Web framework for building the application.
* **TensorFlow/Keras:** For loading and running the pre-trained deep learning model.
* **OpenCV (`cv2`):** Used for image processing, including face detection (Haar Cascades) and image manipulation.
* **NumPy:** For numerical operations, especially with image data.
* **PIL (Pillow):** For image resizing and format conversions.
* **HTML/CSS/JavaScript:** For the front-end user interface.

---

## Setup and Installation

Follow these steps to get the application up and running on your local machine.

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone <https://github.com/MaenDera/FaceFeel>
cd <FaceFeel>
```

### 2. Create a Virtual Environment (Recommended)

It's good practice to use a virtual environment to manage dependencies:

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Place the Model File

The application requires a pre-trained Keras model.
**Place your `last_model_test.keras` file in the root directory of the project.**

### 5. Run the Application

Once everything is set up, you can run the Flask application:

```bash
python app.py
```

---

## Usage

1.  **Access the Application:** Open your web browser and go to `http://127.0.0.1:5000/`.
2.  **Upload Image:** Click on the "Choose File" button to select an image from your computer.
3.  **Choose Prediction Mode:**
    * **"Crop Faces & Predict"**: Check this box if you want the application to detect faces in the image and predict emotions for each individual face.
    * **(Unchecked)**: If left unchecked, the model will attempt to predict emotions on the entire uploaded image.
4.  **Get Prediction:** Click the "Upload and Predict" button.
5.  **View Results:** The application will display the predicted emotion(s), along with the probability scores for each class.

---

## Project Structure

```
.
├── app.py                  # Main Flask application file
├── last_model_test.keras   # Pre-trained Keras model
├── requirements.txt        # Python dependencies
├── static/                 # Static files (CSS, JS)
│   └── style.css/
│   └── app.js/
└── templates/              # HTML templates
    ├── index.html          # Main page with upload form
    ├── research.html       # Page for research info
    └── privacy.html        # Page for privacy policy
```

---

## Contributing

Feel free to fork the repository, make improvements, and submit pull requests.

---

## License

This project is open-source.

---
