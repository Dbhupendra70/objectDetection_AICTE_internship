# objectDetection_AICTE_internship
- Edunet foundation AICTE shell internship cycle 4
  
# Number Plate Recognition System Using OpenCV

A lightweight **license plate detection and anonymization system** built with Python and OpenCV. This tool detects vehicle number plates in **images**, draws a bounding box around them, **blurs the plate for privacy**, and saves the result — ideal for anonymizing vehicle images in datasets or surveillance footage.

> Note: This project **only supports image input** (no video processing).

---

## Description

- **`readimage.py`**  
  Reads an image from `data/img2.png`, detects license plates using a Haar Cascade classifier, blurs each detected plate region, overlays a "License Plate" label, and saves the result to the `output/` directory.

- **`haarcascade_russian_plate_number.xml`**  
  A pre-trained Haar Cascade model (included with OpenCV) used for detecting license plates. Works best on plates similar in style to Russian number plates but may generalize to others.

- **`Data/`**  
  Folder containing the input image (`img2.png`). Place your own test images here and update the path in the script if needed.

- **`output/`**  
  Automatically created directory where the processed image (e.g., `blurred4.jpg`) is saved.

---

## Prerequisites

- Python 3.7 or higher
- OpenCV (`opencv-python`)



Install dependencies:
```bash
pip install opencv-python
```
## Usages
- 1. Place your input image in the data/ folder (or update the path in readimage.py).
- 2. Run the script:
   ```bash
  python readimage.py
  ```
- 3. Find the anonymized output in the output/ folder.
