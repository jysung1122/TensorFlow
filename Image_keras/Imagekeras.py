import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classifier")

        # Load the Keras model
        self.model = load_model("keras_Model.h5", compile=False)
        self.class_names = [line.strip() for line in open("labels.txt", encoding="utf-8")]

        # Create the widgets
        self.btn_select_image = QPushButton("Select Image", self)
        self.btn_select_image.clicked.connect(self.select_image)
        self.lbl_image = QLabel(self)
        self.lbl_confidence = QLabel(self)

        # Set the layout
        self.btn_select_image.setGeometry(10, 10, 120, 30)
        self.lbl_image.setGeometry(10, 50, 400, 400)
        self.lbl_confidence.setGeometry(10, 460, 400, 30)

    def select_image(self):
        # Open file dialog to select an image
        filename, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            # Load and preprocess the image
            image = Image.open(filename).convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Predict the image
            prediction = self.model.predict(data)
            index = np.argmax(prediction)
            class_name = self.class_names[index]
            confidence_score = prediction[0][index]

            # Display the image and confidence score
            pixmap = QPixmap(filename)
            self.lbl_image.setPixmap(pixmap.scaled(400, 400, aspectRatioMode=1))
            self.lbl_confidence.setText(f"Class: {class_name}, Confidence Score: {confidence_score:.4f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(100, 100, 420, 500)
    window.show()
    sys.exit(app.exec_())
