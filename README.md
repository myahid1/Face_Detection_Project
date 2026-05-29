# Face Detection using OpenCV

## Project Overview

This project explores face detection using OpenCV’s Haar Cascade classifier. The goal was to detect face regions in images and draw bounding boxes around them.

Unlike face recognition, this project does not identify individuals. It only detects where face-like regions appear in an image.

## What This Project Covers

* Loading and displaying images with OpenCV
* Converting images from BGR to RGB for display
* Converting images to grayscale for face detection
* Using OpenCV Haar Cascades for face detection
* Drawing bounding boxes around detected faces
* Batch processing 1,800 images
* Testing different `minNeighbors` values
* Analysing missed detections and false positives

## Dataset

The dataset contained 1,800 images.

The images were stored locally and are not included in this repository due to size and usage considerations.

Expected folder structure:

```text
images/
└── dataset/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Methodology

The face detection workflow was:

1. Load each image using OpenCV.
2. Convert the image to grayscale.
3. Apply OpenCV’s Haar Cascade face detector.
4. Draw bounding boxes around detected faces.
5. Save the processed images.
6. Record the number of detected faces per image.
7. Compare different `minNeighbors` settings.

## Parameter Comparison

Two `minNeighbors` settings were tested:

| Setting          | Images with Faces | Images with No Faces | Images with Multiple Detections | Detection Rate |
| ---------------- | ----------------: | -------------------: | ------------------------------: | -------------: |
| `minNeighbors=5` |             1,728 |                   72 |                             264 |         96.00% |
| `minNeighbors=7` |             1,711 |                   89 |                             174 |         95.06% |

`minNeighbors=7` was selected as the final setting because it reduced multiple detections while maintaining a high detection rate.

## Final Results

Final detector setting:

```text
scaleFactor = 1.1
minNeighbors = 7
minSize = (30, 30)
```

Final results:

| Metric                                 |  Value |
| -------------------------------------- | -----: |
| Total images processed                 |  1,800 |
| Images with at least one detected face |  1,711 |
| Images with no detected faces          |     89 |
| Images with multiple detections        |    174 |
| Detection rate                         | 95.06% |

## Key Findings

Haar Cascade worked well on clear, front-facing faces with good lighting.

The detector struggled more with:

* Side-facing or tilted faces
* Sunglasses and accessories
* Low lighting or shadows
* Small faces in full-body images
* Complex backgrounds
* False positives from face-like regions

The project also showed that detection rate is not the same as accuracy. Since there were no manually labelled bounding boxes, the results measure how often the detector produced a face detection, not whether every bounding box was perfectly correct.

## Project Structure

```text
Face_Detection_Project/
├── images/
│   └── dataset/
├── notebooks/
│   └── 01_face_detection.ipynb
├── outputs/
│   ├── face_detection_results.csv
│   ├── face_detection_summary.csv
│   ├── parameter_comparison.csv
│   ├── success_examples/
│   ├── failure_examples/
│   └── multiple_detection_examples/
├── charts/
│   ├── faces_detected_distribution_final.png
│   └── minneighbors_comparison.png
├── final_report.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Technologies Used

* Python
* OpenCV
* Pandas
* Matplotlib
* NumPy
* Jupyter Notebook

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Face_Detection_Project.git
cd Face_Detection_Project
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add the image dataset into:

```text
images/dataset/
```

5. Open and run the notebook:

```text
notebooks/01_face_detection.ipynb
```

## Notes

The full dataset and bulk processed images are not included in this repository. Only selected examples, charts, and result CSV files are included to keep the repository clean.
