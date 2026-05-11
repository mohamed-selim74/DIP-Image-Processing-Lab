# 📸 Interactive Digital Image Processing Lab

A modern, interactive desktop application for image enhancement and processing. Developed as part of my **Computer Engineering** studies at Menofia National University (MNU).

## 🚀 Overview
This application serves as a complete **Image Processing Pipeline**. Instead of applying filters in isolation, this tool allows users to apply filters sequentially, where each filter works on the output of the previous step. This mimics real-world data pipelines used in DevOps and Computer Vision.

## ✨ Key Features
* **Modern UI:** Built with `CustomTkinter` for a sleek, dark-themed professional look.
* **Sequential Pipeline:** Every operation is applied to the latest result, not the original image.
* **Real-time Analysis:** Dual-histogram display (Original vs. Processed) to visualize pixel distribution changes.
* **Advanced Filter Groups:**
    * **Smoothing:** Mean, Median, and Gaussian Blurring.
    * **Sharpening:** Laplacian and Unsharp Masking.
    * **Edge Detection:** Canny and Sobel operators.
    * **Morphology:** Dilation, Erosion, Opening, and Closing.
* **✨ Auto-Enhance:** An intelligent function that detects noise, blur, and low contrast, then automatically applies the necessary fixes.
* **State Management:** Full **Undo/Redo** history and a **Reset** button.

## 🛠 Tech Stack
* **Language:** Python
* **Processing:** OpenCV
* **UI Framework:** CustomTkinter
* **Visualization:** Matplotlib & NumPy

## 🔧 How to Run
1. Install dependencies:
   ```bash
   pip install opencv-python customtkinter matplotlib pillow numpy
