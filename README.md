# 🧩 **SudokuVision**

**SudokuVision** is an OCR-powered Sudoku solver that uses cutting-edge machine learning 🤖 and image processing 📸 techniques to extract Sudoku grids from images and solve them efficiently. Whether your Sudoku puzzle is handwritten ✍️ or printed 📰, **SudokuVision** ensures an accurate solution every time! 🎯

---

## ✨ **Overview**

**SudokuVision** integrates Optical Character Recognition (OCR) for grid extraction 🧠 and deep learning algorithms 🔍 to recognize digits within the grid. The application then uses the **PySudoku** library 🧩 to solve the puzzle, providing a seamless and fast solution ⏱️.

- **OCR-Based Sudoku Grid Detection** 📸
- **Digit Recognition Using Deep Learning** 🤖
- **PySudoku-Based Solver** 🧩
- **Works with Printed and Handwritten Sudoku Puzzles** ✍️
- **Supports PNG, JPG, JPEG Images** 🖼️

---

## ✨ **Features**

- **Grid Extraction**: Automatically extracts Sudoku grids from images using advanced image processing techniques 🎯.
- **Digit Recognition**: Identifies digits within the grid using deep learning models 🔢.
- **Fast Solver**: Solves puzzles using the **PySudoku** backtracking solver ⏱️.
- **Command Line and Web Interface**: Provides both command-line and Streamlit-based web interface for ease of use 🖥️.
- **Multiple Image Formats**: Works with various image formats like PNG, JPG, and JPEG 📸.
- **Real-time Visualization**: Displays the solved Sudoku puzzle directly in the web app 🌐 and on the command line interface using **OpenCV** 💡.

---

## 📊 **Datasets Used**

The following datasets were used for training the deep learning model for digit recognition and grid extraction:

### 🧑‍🏫 **1. MNIST Dataset**
The **MNIST** dataset, containing 60,000 handwritten digit images 🖋️, was used to train the deep learning model for digit recognition. The dataset includes grayscale images 🖤 of size 28x28 pixels and is ideal for training models to recognize handwritten digits.

- **Preprocessing**:  
  - Grayscale normalization to a range of [0, 1] 🌑.
  - Reshaped to 28x28 pixels 🖼️.

### 🔠 **2. Chars74K Dataset**
The **Chars74K** dataset contains images of characters in various fonts 🔠, including digits, used to supplement the training process with diverse digital font variations 🅰️.

- **Preprocessing**:  
  - Resized to 28x28 pixels 🖼️.
  - Grayscale conversion and normalization to a range of [0, 1] 🌑.

This dataset enhances the model's ability to recognize digits in digital fonts, improving accuracy across various types of input 📏.

### ✍️ **3. TMNIST Dataset**
The **TMNIST** dataset is another handwritten digit dataset used to further train and diversify the digit recognition capabilities 🤖. It contains images in the same format as MNIST and was used to train the model on additional handwritten digits ✍️.

- **Preprocessing**:  
  - Data is scaled to a range of [0, 1] 🌑.
  - Labels are encoded using **LabelEncoder** 🔣 and converted to categorical values 📊.

---

## 🔍 **How It Works**

1. **Puzzle Extraction** 🧩  
   The uploaded image is processed using **OpenCV** 🖼️ for grid extraction. The grid's edges are detected, and the puzzle is segmented into individual cells 🏷️.

2. **Digit Recognition** 🔢  
   Each individual cell in the grid is processed by a deep learning model that recognizes the digits 🧠. The model is trained on the **MNIST**, **Chars74K**, and **TMNIST** datasets 📊.

3. **Puzzle Solving** 🧩  
   Once the digits are identified, they are passed to the **PySudoku** solver, which uses a backtracking algorithm 🔄 to solve the puzzle 🧩.

4. **Result Display** 🎥  
   The original and solved puzzles are displayed:  
   - **Web App** 🌐: The result is shown directly in the browser 🌍.  
   - **Command Line** 💻: The solved puzzle is displayed directly in the terminal using **OpenCV** 🖼️. The result is visualized without saving it to a file, using `cv2.imshow()` to show the solved puzzle 🧩.

---

## ⚙️ **Setup and Installation**

To set up **SudokuVision** 🧩 on your local machine 💻, follow the instructions below:

### 📝 **1. Clone the repository**
```bash
git clone https://github.com/ArchitJ6/SudokuVision.git  
cd SudokuVision  
```

### 📦 **2. Install Dependencies**
Create a virtual environment (recommended) 🌱 and install required packages:  
```bash
python -m venv venv  
source venv/bin/activate  # On Windows use `venv\Scripts\activate`  
pip install -r requirements.txt  
```

### 📥 **3. Download the Datasets**  
Make sure to download the **Chars74K**, **MNIST**, and **TMNIST** datasets. The datasets should be organized in the following directory structure:

```
/datasets  
    /Chars74K-Digital-English-Font  
        Extract the files and place the folders for digits 0 to 9 here, each containing images of the corresponding digit (labeled accordingly).  
    /tmnist  
        This dataset contains a `data.csv` file with the data for handwritten digits.  
```

- **MNIST**: The MNIST dataset will be used directly from Keras.  
- **Chars74K-Digital-English-Font**: Extract the files and organize them into folders for digits 0 to 9, with images of each digit placed inside the corresponding folder, labeled by the digit.  
- **TMNIST**: This dataset includes a `data.csv` file that contains the data for handwritten digits. 
---

## 🧑‍💻 **Usage**

### 🌐 **1. Streamlit Interface**

To run the web interface using Streamlit 🖥️, follow these steps:

1. **Run the Streamlit app**:  
```bash
streamlit run app.py  
```

2. **Upload the Image** 🖼️:  
   - After the app starts, open the URL provided by Streamlit 🌐.  
   - Upload an image of the Sudoku puzzle (printed or handwritten) 🧩.  
   - Click **"Solve Sudoku"** 🧠 to process and get the solution 🧩.

3. **Output** 🎥:  
   The original puzzle with solved values will be displayed directly on the web interface 🌐.

### 💻 **2. Command-Line Interface (CLI)**

To use the command-line interface 💻:

1. **Run the script**:  
```bash
python solve.py --image <path_to_image> --debug -1  
```

- `--image`: Path to the Sudoku image 🖼️.
- `--debug`: Set to `1` for debug mode 🛠️, which visualizes the grid and digit extraction process 🔍.

2. **Output** 🎥:  
   The solved Sudoku puzzle 🧩 will be displayed directly in the new window using **OpenCV** 🖼️. The window will automatically close when any key is pressed ⏳.

### 📚 **3. Model Training**

To train the model for digit recognition 🧠, use the following script:

```bash
python train_model.py  
```

This will load the datasets 📊, preprocess the data 🔄, train the model 🤖, and save the trained model for future use 💾.

---

## 💡 **Best Practices for Usage**

To get the most accurate results, keep the following tips in mind:

- Ensure good lighting when capturing handwritten Sudoku puzzles ✍️ for optimal digit recognition.
- Use high-resolution images 🖼️ for better grid and digit extraction.
- For handwritten puzzles, maintain legibility of digits for improved accuracy ✍️.

---

## 🔧 **Troubleshooting**

If you run into issues, check these common solutions:

- **Missing Dependencies**: Make sure all packages are installed by running `pip install -r requirements.txt` 📦.
- **Image Processing Errors**: Ensure that the uploaded image is clear and contains a proper Sudoku grid 📸.
- **Solver Not Working**: Make sure the digits are clearly detected by checking the debug output with the `--debug` flag 🛠️.

---

## 🔒 **Security and Privacy**

Your uploaded images are processed locally and are not stored long-term. We respect your privacy and ensure that no sensitive information is exposed during the image upload and processing process 🔐.

---

## 🤝 **How to Contribute**

We welcome contributions! 🎉 To contribute to **SudokuVision** 🧩, follow these steps:

1. Fork the repository 🍴.
2. Create a new branch (`git checkout -b feature-name`) 🌱.
3. Make your changes ✍️.
4. Commit your changes (`git commit -m 'Add feature'`) 💬.
5. Push to the branch (`git push origin feature-name`) 🚀.
6. Open a pull request with a description of your changes 📄.

---

## 🙏 **Acknowledgments**

- **PySudoku Library** 🧩: For providing an efficient backtracking-based solver.
- **MNIST** 📚: For the dataset used for training the digit recognition model.
- **Chars74K** 🔠: For the dataset of digital fonts, enriching the model's ability to recognize various types of digits.
- **TMNIST** ✍️: For further diversifying the training data and enhancing recognition accuracy.

---

## 📜 **License**

This project is licensed under the [**MIT License**](LICENSE) ⚖️.