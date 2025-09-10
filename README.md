# 🐱 Cat vs Non-Cat Classifier( Andrew NG deep learning specialization course [c1 ,w4 exercises] )

This project implements a **Deep Neural Network (DNN)** from scratch using **NumPy**, trained to classify images as **cat** or **non-cat**.  
It is based on assignments from the **Deep Learning Specialization (Andrew Ng)** and provides a step-by-step implementation of forward/backward propagation, cost computation, and parameter updates.

---

## 📂 Project Structure

├── images/ # Sample images for testing the model

├── dnn_app_utils_v3.py # Helper functions (provided by the course)

├── dnn_utils.py # Utility functions (activation, initialization, etc.)

├── MAIN FILE ---- cat classification.py # Main entry point to run the model

├── train_catvnoncat.h5 # Training dataset (cat vs non-cat)

├── test_catvnoncat.h5 # Test dataset

└── testCases.py # Unit tests for core functions


---

## 🚀 How to Run
1. Install requirements:
   ```bash
   pip install numpy h5py matplotlib
2. Run the main script:
python "MAIN FILE ---- cat classification.py"

🧠 Model Overview

Two model options are available:

- 2-layer Neural Network (shallow model)
- L-layer Deep Neural Network (flexible depth, e.g., 4 layers)

You can switch between the two implementations and experiment with different hyperparameters:

- Number of hidden layers
- Size of each hidden laye
- Learning rate
- Number of iterations

📊 Results

Both the 2-layer and L-layer models are trained on the cats vs non-cats dataset.
The deep model typically achieves higher accuracy, while the 2-layer model is easier to interpret and debug.
By tweaking hyperparameters, you can compare performance and learning curves between shallow and deep architectures.

✨ Features

- Full NumPy-only implementation (no TensorFlow/PyTorch)
- Supports both shallow (2-layer) and deep (L-layer) neural networks
- Clean and modular codebase
- Educational project for understanding the mechanics of deep learning
- you can test the model by your own pictures by putting them in 'images' folder and also write the name of your picture in the code! 
