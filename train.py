from src.models.sudoku_architecture import Sudoku
from keras import optimizers, datasets
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.utils import shuffle
import seaborn as sns
import os
from PIL import Image

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="checkpoints/digit_classifier.keras", help="path to save model after training")
args = vars(ap.parse_args())

# Initialize the initial learning rate, number of epochs to train for, and batch size
INIT_LR = 1e-3
EPOCHS = 50
BS = 64

# Grab the MNIST dataset
print("[INFO] getting MNIST dataset...")
((trainDataMnist, trainLabelsMnist), (testDataMnist, testLabelsMnist)) = datasets.mnist.load_data()
# Add a channel (i.e., grayscale) dimension to the digits
trainDataMnist = trainDataMnist.reshape((trainDataMnist.shape[0], 28, 28, 1))
testDataMnist = testDataMnist.reshape((testDataMnist.shape[0], 28, 28, 1))
# # Scale data to the range of [0, 1]
trainDataMnist = trainDataMnist.astype("float32") / 255.0
testDataMnist = testDataMnist.astype("float32") / 255.0

# Load Chars74K dataset
print("[INFO] Getting Chars74K dataset...")
path = "datasets/Chars74K-Digital-English-Font"
folders = os.listdir(path)

chars74k_images = []
chars74k_labels = []

for folder in folders:
    label = int(folder)  # Labels are stored as folder names
    folder_path = os.path.join(path, folder)
    images = os.listdir(folder_path)

    for img_file in images:
        img_path = os.path.join(folder_path, img_file)
        
        # Open, convert to grayscale, resize to 28x28
        img = Image.open(img_path).convert("L").resize((28, 28))
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype="float32") / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        
        chars74k_images.append(img_array)
        chars74k_labels.append(label)

# Convert lists to numpy arrays
chars74k_images = np.array(chars74k_images)
chars74k_labels = np.array(chars74k_labels)

# Convert labels to categorical (Assuming Chars74K has 10 classes)
chars74k_labels = utils.to_categorical(chars74k_labels, num_classes=10)

# Split Chars74K into train and test sets (80-20 split)
trainDataChars74K, testDataChars74K, trainLabelsChars74K, testLabelsChars74K = train_test_split(
    chars74k_images, chars74k_labels, test_size=0.2, random_state=42
)

# Grab the TMNIST dataset
print("[INFO] getting TMNIST dataset...")
tmnist = pd.read_csv('datasets/tmnist/data.csv')
pixels = tmnist.drop(columns = ['names', 'labels']).values / 255.0
pixels  = pixels.reshape((tmnist.shape[0], 28, 28, 1))

label_encoder = LabelEncoder()
numerals = label_encoder.fit_transform(tmnist['labels'])
numerals = utils.to_categorical(numerals)

# Split the data into training and testing sets
(trainDataTmnist, testDataTmnist, numeralsTrainTmnist, numeralsTestTmnist) = train_test_split(pixels, numerals, test_size = 0.2, random_state = 42)

print("trainDataMnist.shape",trainDataMnist.shape)
print("trainDataTmnist.shape",trainDataTmnist.shape)
print("testDataMnist.shape",testDataMnist.shape)
print("testDataTmnist.shape",testDataTmnist.shape)
print("trainLabelsMnist.shape",trainLabelsMnist.shape)
print("numeralsTrainTmnist.shape",numeralsTrainTmnist.shape)
print("testLabelsMnist.shape",testLabelsMnist.shape)
print("numeralsTestTmnist.shape",numeralsTestTmnist.shape)
print("trainDataChars74K.shape",trainDataChars74K.shape)
print("testDataChars74K.shape",testDataChars74K.shape)
print("trainLabelsChars74K.shape",trainLabelsChars74K.shape)
print("testLabelsChars74K.shape",testLabelsChars74K.shape)

# One-hot encode trainLabelsMnist and testLabelsMnist because numeralsTrainTmnist and numeralsTestTmnist are already one-hot encoded
trainLabelsMnist = utils.to_categorical(trainLabelsMnist, num_classes=10)
testLabelsMnist = utils.to_categorical(testLabelsMnist, num_classes=10)

# Print shapes after one-hot encoding
print(f"trainLabelsMnist shape after one-hot encoding: {trainLabelsMnist.shape}")
print(f"testLabelsMnist shape after one-hot encoding: {testLabelsMnist.shape}")

# Print sizes of the datasets
print(f"trainDataMnist size: {trainDataMnist.shape[0]}")
print(f"trainDataTmnist size: {trainDataTmnist.shape[0]}")
print(f"testDataMnist size: {testDataMnist.shape[0]}")
print(f"testDataTmnist size: {testDataTmnist.shape[0]}")
print(f"trainDataChars74K size: {trainDataChars74K.shape[0]}")
print(f"testDataChars74K size: {testDataChars74K.shape[0]}")

# Combine the datasets
trainDataCombined = np.concatenate((trainDataMnist, trainDataTmnist, trainDataChars74K), axis=0)
testDataCombined = np.concatenate((testDataMnist, testDataTmnist, testDataChars74K), axis=0)
trainLabelsCombined = np.concatenate((trainLabelsMnist, numeralsTrainTmnist, trainLabelsChars74K), axis=0)
testLabelsCombined = np.concatenate((testLabelsMnist, numeralsTestTmnist, testLabelsChars74K), axis=0)

# Print shapes after combining
print(f"Combined training data shape: {trainDataCombined.shape}")
print(f"Combined test data shape: {testDataCombined.shape}")
print(f"Combined training labels shape: {trainLabelsCombined.shape}")
print(f"Combined test labels shape: {testLabelsCombined.shape}")
print(f"Combined training data size: {trainDataCombined.shape[0]}")
print(f"Combined test data size: {testDataCombined.shape[0]}")

# Shuffle the combined dataset
trainDataCombined, trainLabelsCombined = shuffle(trainDataCombined, trainLabelsCombined, random_state=42)
testDataCombined, testLabelsCombined = shuffle(testDataCombined, testLabelsCombined, random_state=42)

# Print shapes after combining and shuffling
print(f"Combined training data shape: {trainDataCombined.shape}")
print(f"Combined test data shape: {testDataCombined.shape}")
print(f"Combined training labels shape: {trainLabelsCombined.shape}")
print(f"Combined test labels shape: {testLabelsCombined.shape}")

# Learning rate scheduler
# lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Compile the model
print("[INFO] Compiling model...")
opt = optimizers.Adam(learning_rate=INIT_LR)
model = Sudoku.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# early_stopping = callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=5,
#     restore_best_weights=True
# )

# Train the model
print("[INFO] Training network...")
history = model.fit(
    trainDataCombined, trainLabelsCombined,
    validation_data=(testDataCombined, testLabelsCombined),
    batch_size=BS,
    epochs=EPOCHS,
    # callbacks=[lr_scheduler, early_stopping],
    # callbacks=[lr_scheduler],
    verbose=1
)

# Plot the training loss and accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Combined Numeral Prediction CNN Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Combined Numeral Prediction CNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Evaluate the model
_, test_acc = model.evaluate(testDataCombined, testLabelsCombined, verbose=0)
print(f'Test Accuracy: {100 * test_acc:.2f}%')

_, train_acc = model.evaluate(trainDataCombined, trainLabelsCombined, verbose=0)
print(f'Train Accuracy: {100 * train_acc:.2f}%')

# Confusion matrix for the combined test dataset
predicted_probs = model.predict(testDataCombined)
predicted_labels = np.argmax(predicted_probs, axis=1)
true_labels = np.argmax(testLabelsCombined, axis=1)

conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# predicted_probs = model.predict(testDataCombined)
predicted_labels = np.argmax(predicted_probs, axis = 1)
true_labels = np.argmax(testLabelsCombined, axis = 1)
incorrect_indices = np.where(predicted_labels != true_labels)[0]

print(f'Number of incorrect predictions: {len(incorrect_indices)}')

def display_incorrect_predictions(images, true_labels, predictions, indices, n_display = 10, class_names = None):
    if len(indices) > 0:
        display_indices = np.random.choice(indices, size = min(n_display, len(indices)), replace = False)

        plt.figure(figsize = (1.5 * n_display, 3 * n_display))
        for i, idx in enumerate(display_indices, start=1):
            img = images[idx].squeeze()
            pred_label = predictions[idx]
            true_label = true_labels[idx]
            plt.subplot(1, n_display, i)
            plt.imshow(img, cmap = 'gray')
            title_text = f'Predicted: {class_names[pred_label] if class_names else pred_label}\nTrue: {class_names[true_label] if class_names else true_label}'
            plt.title(title_text)
            plt.axis('off')
        plt.show()
    else:
        print('No incorrect predictions to display.')

display_incorrect_predictions(testDataCombined, true_labels, predicted_labels, incorrect_indices)

# Save the model to disk
print("[INFO] saving model...")
model.save(args["model"])