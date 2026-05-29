# Face Detection using OpenCV Haar Cascades

## 1. Project Overview

This project explores face detection using classical computer vision techniques in OpenCV. Unlike an image classification task, where the model predicts one label for an entire image, face detection is a localisation task. The goal is to identify where the face appears in the image by drawing a bounding box around it.

The project uses OpenCV’s pretrained Haar Cascade frontal face detector to process a dataset of 1,800 images. The detector was applied to each image, and the number of detected faces was recorded. Detected images were saved with bounding boxes, and the results were analysed quantitatively and qualitatively.

The project focuses on:

* Loading and displaying images using OpenCV
* Understanding BGR and RGB colour formats
* Converting images to grayscale
* Applying Haar Cascade face detection
* Drawing bounding boxes around detected faces
* Batch processing a full image dataset
* Tuning the `minNeighbors` parameter
* Comparing detection settings
* Analysing successful detections, missed detections, and false positives

The final selected detector setting used `minNeighbors=7`, which provided a better balance between maintaining a high detection rate and reducing excessive multiple detections.

## 2. About Computer Vision

Computer vision focuses on extracting useful information from visual data such as images and videos. In an image processing workflow, an image is represented as a grid of pixel values, where each pixel contains information about colour or intensity. Computer vision techniques use these pixel patterns to identify visual structures such as edges, shapes, textures, objects, and faces.

This project focuses specifically on face detection, which is a localisation task. Instead of assigning a single label to an entire image, the detector identifies regions within the image that are likely to contain human faces. The output is represented using bounding boxes, where each box marks the location and size of a detected face region.

Each bounding box is described using four values:

x, y, width, height

The x and y values represent the top-left corner of the detected region, while width and height define the size of the bounding box. These coordinates allow the detected face region to be highlighted visually by drawing a rectangle over the image.

This project is limited to face detection and does not perform face recognition. Face detection identifies the presence and location of faces, while face recognition attempts to identify or verify a specific person. Since this project does not compare identities or recognise individuals, it is focused only on the technical task of detecting face-like regions in images.

## 3. OpenCV and Image Loading

OpenCV was used as the main computer vision library for this project. Images were loaded using `cv2.imread()`. One important detail is that OpenCV stores colour images in BGR format by default, while Matplotlib expects RGB format for correct visual display [1].

Because of this, images were converted from BGR to RGB before being displayed using Matplotlib:

```python
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

This step does not change the face detection itself, but it is important for displaying images correctly in the notebook. Without this conversion, images may appear with incorrect colours.

For face detection using Haar Cascades, the images were also converted to grayscale:

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

Grayscale conversion simplifies the image by reducing it from three colour channels to one intensity channel. Haar Cascade detection relies on contrast-based visual patterns rather than full colour information, so grayscale images are commonly used for this method.

## 4. Haar Cascade Face Detection

The main detector used in this project was OpenCV’s pretrained Haar Cascade frontal face detector:

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
```

Haar Cascade detection is a classical machine learning-based object detection method. It is based on the Viola-Jones object detection framework, which uses Haar-like features and a cascade of classifiers to detect objects efficiently [2].

The method works by scanning different regions of an image and checking whether each region contains patterns that resemble the target object, in this case, a frontal human face. Haar-like features are simple rectangular patterns that compare differences in pixel intensity between neighbouring regions. These features can capture patterns such as darker eye regions, brighter cheeks, or contrast between facial structures.

The “cascade” part of Haar Cascade means that the detector applies a sequence of increasingly specific classifier stages. Simple stages quickly reject regions that are unlikely to contain a face, while later stages perform more detailed checks. This makes detection efficient because the algorithm does not spend equal effort on every part of the image.

In OpenCV, the detector returns bounding boxes for detected objects. Each detected face is returned as:

```text
x, y, w, h
```

These values were then used to draw green rectangles around the detected face regions.

## 5. Detection Parameters

The detector was applied using OpenCV’s `detectMultiScale()` method:

```python
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=7,
    minSize=(30, 30)
)
```

The key parameters used were:

### `scaleFactor`

The `scaleFactor` controls how much the image size is reduced at each scale during detection. Since faces can appear at different sizes, the detector checks the image at multiple scales. A smaller `scaleFactor` may allow finer detection across different face sizes, but can also increase computation. In this project, `scaleFactor=1.1` was used.

### `minNeighbors`

The `minNeighbors` parameter controls how many neighbouring candidate rectangles are required before a detection is retained. In simple terms, it affects how strict the detector is [3].

A lower `minNeighbors` value makes the detector more permissive. This can detect more faces, but it may also create more false positives. A higher `minNeighbors` value makes the detector stricter. This can reduce false positives, but it may also miss some real faces.

This project compared two settings:

```text
minNeighbors = 5
minNeighbors = 7
```

The comparison helped evaluate the trade-off between detection coverage and likely false positives.

### `minSize`

The `minSize` parameter sets the minimum object size that the detector should consider. In this project, `minSize=(30, 30)` was used to ignore very small regions that were unlikely to be meaningful face detections.

## 6. About MTCNN

Although the implemented model in this project used Haar Cascades, it is useful to briefly compare this approach with MTCNN.

MTCNN stands for Multi-task Cascaded Convolutional Networks. It is a deep learning-based face detection and alignment framework introduced by Zhang et al. The method uses a cascaded structure with three stages of convolutional neural networks that detect faces and facial landmarks in a coarse-to-fine manner [4].

Compared to Haar Cascades, MTCNN is generally more advanced because it uses learned deep features rather than hand-designed Haar-like features. It is also designed to handle more challenging real-world conditions such as pose variation, illumination changes, occlusion, and facial landmark alignment [4].

However, this project focused on Haar Cascades because they are lightweight, easy to use through OpenCV, and useful for understanding classical face detection before moving into deep learning-based detectors. MTCNN would be a good future extension of this project, especially for comparing classical computer vision with deep learning-based face detection.

## 7. Dataset and Processing Workflow

The dataset contained 1,800 images. Each image was processed using the following workflow:

1. Load the image using OpenCV.
2. Convert the image from BGR to RGB for display.
3. Convert the image to grayscale for Haar Cascade detection.
4. Apply the Haar Cascade detector using `detectMultiScale()`.
5. Draw bounding boxes around detected faces.
6. Save the output image.
7. Record the number of detected faces for each image.

The detection results were stored in a dataframe with the following columns:

```text
filename
faces_detected
min_neighbors
```

The final results were saved to:

```text
outputs/face_detection_results.csv
outputs/face_detection_summary.csv
outputs/parameter_comparison.csv
```

Charts were saved to:

```text
charts/faces_detected_distribution_final.png
charts/minneighbors_comparison.png
```

## 8. Parameter Comparison

Two `minNeighbors` settings were tested: `minNeighbors=5` and `minNeighbors=7`.

| Setting        | Total Images | Images with Faces | Images with No Faces | Images with Multiple Detections | Detection Rate |
| -------------- | -----------: | ----------------: | -------------------: | ------------------------------: | -------------: |
| minNeighbors=5 |        1,800 |             1,728 |                   72 |                             264 |         96.00% |
| minNeighbors=7 |        1,800 |             1,711 |                   89 |                             174 |         95.06% |

With `minNeighbors=5`, the detector achieved a slightly higher detection rate of 96.00%. However, it also produced 264 images with multiple detections. Based on visual inspection, some of these multiple detections were likely false positives, where the detector identified non-face regions as faces.

With `minNeighbors=7`, the detection rate decreased slightly to 95.06%. The number of images with no detected faces increased from 72 to 89. However, the number of images with multiple detections decreased from 264 to 174.

This suggests that increasing `minNeighbors` made the detector more conservative. The stricter setting reduced excessive detections while only slightly lowering the overall detection rate. For this reason, `minNeighbors=7` was selected as the final setting.

## 9. Final Detection Results

The final detector used:

```text
scaleFactor = 1.1
minNeighbors = 7
minSize = (30, 30)
```

The final results were:

| Metric                                 |  Value |
| -------------------------------------- | -----: |
| Total images processed                 |  1,800 |
| Images with at least one detected face |  1,711 |
| Images with no detected faces          |     89 |
| Images with multiple detections        |    174 |
| Detection rate                         | 95.06% |

The detector found at least one face-like region in 1,711 out of 1,800 images. This gives a detection rate of 95.06%.

However, this should not be interpreted as true accuracy. The dataset did not contain manually labelled bounding boxes, so there was no ground truth available to calculate exact precision, recall, or Intersection over Union. Instead, the detection rate measures how often the detector produced at least one detected face region per image.

## 10. Qualitative Error Analysis

The quantitative results showed that Haar Cascade detected at least one face in most images. However, visual inspection was still necessary because detection count alone does not confirm whether the bounding boxes were correct.

### Images with 0 Faces Detected

Some images with zero detected faces still contained visible faces. These cases were treated as possible missed detections.

Common causes included:

* Non-frontal face angles
* Tilted or rotated heads
* Sunglasses or facial accessories
* Shadows or uneven lighting
* Smaller faces in full-body images
* Partial occlusion from hair, hats, or pose

This shows that Haar Cascades can struggle when images differ from the frontal-face patterns that the detector was trained to recognise.

### Images with 1 Face Detected

Images with one detected face were generally successful, especially when the face was:

* Front-facing
* Large enough in the image
* Well-lit
* Not heavily occluded
* Clearly separated from the background

However, one detected face does not always mean perfect detection. Some boxes may be slightly misaligned or may capture only part of the face. This is why detection count should be interpreted carefully.

### Images with Multiple Faces Detected

Images with multiple detections were especially useful for identifying false positives. In some cases, the detector correctly found the main face but also detected extra face-like regions in the background, clothing, accessories, or other parts of the image.

This explains why the `minNeighbors=7` setting was preferred. It reduced the number of multiple-detection cases from 264 to 174, suggesting that it helped reduce likely false positives.

## 11. Key Findings

The Haar Cascade detector worked well on many clear, frontal face images. The final setting detected at least one face in 95.06% of the 1,800 images.

The `minNeighbors` parameter had a noticeable effect on the detector’s behaviour. A lower value of 5 produced a slightly higher detection rate, but also produced more multiple detections. A higher value of 7 reduced multiple detections significantly, while only slightly reducing the detection rate.

This suggests that parameter tuning is important in computer vision pipelines. A detector’s default settings may work, but small changes can affect the balance between missed detections and false positives.

The project also showed the difference between detection rate and detection accuracy. Without labelled bounding boxes, it is not possible to calculate true object detection accuracy. A high detection rate only means that the detector produced at least one detection in most images. It does not guarantee that every bounding box was correct.

## 12. Limitations

This project has several limitations.

First, the dataset did not include manually labelled bounding boxes. As a result, the project could not calculate proper object detection metrics such as precision, recall, F1-score, or Intersection over Union.

Second, Haar Cascade detection is more limited than modern deep learning-based methods. It can work well for simple frontal faces, but it may struggle with side profiles, unusual angles, occlusion, low lighting, and complex backgrounds.

Third, multiple detections do not always mean that the detector is wrong. Some images may genuinely contain more than one face. However, visual inspection showed that some multiple detections were likely false positives.

Fourth, zero detections do not always mean that the image contains a missed face. Some images may have unclear or partially visible faces. Manual review is needed to understand whether zero-detection cases are true failures.

Finally, this project used a pretrained Haar Cascade detector rather than training a new model. The goal was to understand and evaluate a classical face detection method, not to build a custom detector from scratch.

## 13. Future Improvements

Future improvements could include using a manually labelled dataset with ground truth bounding boxes. This would allow proper evaluation using object detection metrics such as precision, recall, and Intersection over Union.

Another improvement would be to compare Haar Cascades with MTCNN. Since MTCNN is a deep learning-based detector designed for more challenging face detection and alignment scenarios, it may perform better on angled faces, occluded faces, and images with difficult lighting conditions.

A further extension would be to test other modern face detectors or object detection models, such as YOLO-based face detection. This would allow a broader comparison between classical computer vision methods and modern deep learning approaches.

## 14. Conclusion

This project demonstrated a complete face detection workflow using OpenCV and Haar Cascades. The detector was applied to 1,800 images, and detection results were saved, analysed, and compared across different parameter settings.

The final selected setting used `minNeighbors=7`, which achieved a detection rate of 95.06%. Although this was slightly lower than the 96.00% detection rate from `minNeighbors=5`, it reduced the number of images with multiple detections from 264 to 174. This made it a better overall setting for reducing likely false positives while still detecting faces in most images.

Overall, the project showed how classical computer vision methods can be used for face detection, how OpenCV handles image processing, and why parameter tuning and qualitative error analysis are important. It also highlighted the limitations of Haar Cascades and provided a natural next step toward comparing classical methods with deep learning-based detectors such as MTCNN.

## References

[1] OpenCV Documentation. *Color conversions*. OpenCV explains colour space conversion and the BGR/RGB channel ordering used in OpenCV image processing.  
https://docs.opencv.org/master/d8/d01/group__imgproc__color__conversions.html

[2] OpenCV Documentation. *Face Detection using Haar Cascades*. OpenCV describes Haar feature-based cascade classifiers as an object detection method based on the Viola-Jones framework.  
https://docs.opencv.org/3.4/d2/d99/tutorial_js_face_detection.html

[3] OpenCV Documentation. *CascadeClassifier detectMultiScale*. OpenCV describes `minNeighbors` as the parameter specifying how many neighbouring candidate rectangles are required for a detection to be retained.  
https://docs.opencv.org/3.4/javadoc/org/opencv/objdetect/CascadeClassifier.html

[4] Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. *Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks*. arXiv, 2016.  
https://arxiv.org/abs/1604.02878
