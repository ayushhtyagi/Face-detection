---

# Face Detection Model

This project builds a face detection model using a dataset from Kaggle. The dataset is used to train a convolutional neural network (CNN) to classify images as containing faces or non-faces.

## Dataset

The dataset used in this project is the **Face Detection Dataset** from Kaggle. It contains two classes:
- `face`: Images with human faces.
- `non_face`: Images without human faces.

You can find the dataset [here](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset).

## Project Structure

```
.
├── face_detection.ipynb      # Main Jupyter notebook for training and testing the model
├── face_detection_model.h5   # Saved model after training
├── kaggle.json               # Kaggle API credentials (not included in the repository)
└── README.md                 # This file
```

## Prerequisites

1. Google Colab environment.
2. Kaggle API installed.
3. Python libraries:
   - `tensorflow`
   - `numpy`
   - `opencv-python`
   - `matplotlib`

## How to Run the Project

### 1. Install Kaggle API

First, you need to install the Kaggle API on Colab and authenticate it.

```bash
!pip install kaggle
```

Upload your `kaggle.json` file to authenticate:

```python
from google.colab import files
files.upload()  # Upload kaggle.json
```

Configure the Kaggle environment:

```python
import os
os.environ['KAGGLE_CONFIG_DIR'] = '/content'
```

### 2. Download the Dataset

Use the Kaggle API to download the dataset:

```bash
!kaggle datasets download -d fareselmenshawii/face-detection-dataset
```

Unzip the dataset:

```python
import zipfile

with zipfile.ZipFile('face-detection-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/face-detection-dataset')
```

### 3. Explore the Dataset

You can explore the dataset by listing the files and displaying some sample images:

```python
import os
import matplotlib.pyplot as plt
import cv2

dataset_dir = '/content/face-detection-dataset'

def load_and_display_samples(data_dir, num_samples=5):
    classes = os.listdir(data_dir)
    for cls in classes:
        class_dir = os.path.join(data_dir, cls)
        images = os.listdir(class_dir)
        plt.figure(figsize=(10, 5))
        
        for i, img_name in enumerate(images[:num_samples]):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.title(cls)
        
        plt.show()

load_and_display_samples(dataset_dir)
```

### 4. Train the Model

The model is a Convolutional Neural Network (CNN) built using TensorFlow. It consists of several convolutional and pooling layers, followed by fully connected layers.

Run the following to load, preprocess the dataset, and train the model:

```python
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
IMG_WIDTH, IMG_HEIGHT = 128, 128
BATCH_SIZE = 32
EPOCHS = 50

def load_and_preprocess_data(data_dir):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training')

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation')
    
    return train_generator, validation_generator

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

data_dir = '/content/face-detection-dataset'
train_generator, validation_generator = load_and_preprocess_data(data_dir)
model = create_model()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)
```

### 5. Save the Model

After training, save the model to a file:

```python
model.save('face_detection_model.h5')
```

### 6. Use the Model for Face Detection

You can test the model on new images:

```python
import numpy as np
import cv2

def detect_face(model, image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    prediction = model.predict(img_batch)
    
    if prediction[0][0] > 0.5:
        print("Face detected!")
        cv2.putText(img, "Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        print("No face detected.")
        cv2.putText(img, "No Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

test_image_path = 'path/to/test/image.jpg'
detect_face(model, test_image_path)
```

## License

This project is open source and available under the MIT License.

---
