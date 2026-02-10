
# BG10 – WasteNet: A Structured-Pruned andEMA-Augmented RepVGG Framework forReal-Time Trash Classification

## Team Info
- 22471A0597 — **kakumanu Sravani** ( [LinkedIn](https://www.linkedin.com/in/kakumanu-sravani-206663363/) )
_Work Done: Designed and implemented the RepVGG-MEM framework integrating MSCA attention, EMA, and pruning techniques. Managed experimental setup, hyperparameter tuning, and final model evaluation.Prepared research documentation, methodology diagrams, and result analysis for the paper

- 22471A05A4 — **kondavarju Ramya** ( [LinkedIn](https://www.linkedin.com/in/ramya-kondavarju-060662282/) )
_Work Done: Developed the system workflow and block diagram of the proposed RepVGG-based waste classification pipeline.Assisted in preprocessing visualization (before–after results), dataset distribution analysis.

- 22471A0584— **Chinthalanka Poojitha** ( [LinkedIn](www.linkedin.com/in/ch-poojitha-501a62287) )
_Work Done:Contributed to dataset collection and organization from TrashNet, Kaggle, and PublicGarbageNet sources. Performed data preprocessing including resizing, Gaussian blur, CLAHE enhancement, normalization, and augmentation techniques. Assisted in training the RepVGG-based model and evaluating performance using accuracy and F1-score metrics.


---

## Abstract
Achieving efficient classification of garbage boosts recycling and helps with pollution control as well as sustainable waste management. To address this challenge,we developed RepVGG-MEM, which is a lightweight and highly accurate Deep Learning model for real-time waste classification. Built on RepVGG, the model applies EMA,Mixup data augmented training, and structured pruning to optimization to maintain a strong speed performance ratio. The model and its described metrics, performance of 95.06% accuracy and a 94.5% F1 score, even with closely resembling waste categories surpass several previous versions. Moreover, its accuracy makes it ideal for edge devices and smart mobile waste management systems.RepVGGMEM shows better performance and generalization than earlier versions. Its lightweight design makes it easy to deploy on edge devices such as smart bins and mobile platforms. This capability positions it as a promising solution for automated waste management in real-world situations.

---

## Paper Reference (Inspiration)
👉 **[Paper Title RepVGG-MEM: A Lightweight Model for Garbage Classification Achieving a Balance Between Accuracy and Speed
  – Author Names Qiuxin Si and Sang Ik Han, 2025, IEEE  
 ](https://ieeexplore.ieee.org/document/10900382)**
Original conference/IEEE paper used as inspiration for the model.

---

## Our Improvement Over Existing Paper
This work improves existing waste classification models by combining Exponential Moving Average (EMA), Mixup data augmentation, and structured magnitude-based pruning within a single RepVGG-based framework.
Unlike earlier methods that focus only on accuracy or use heavy ensemble models, our approach achieves high accuracy with reduced computational cost, making it suitable for real-time and edge-device deployment. The pruned RepVGG-MEM model achieves 95.06% accuracy and 94.5% F1-score, outperforming baseline RepVGG and other lightweight CNN models.

---

## About the Project

What the project does
The project automatically classifies waste images into multiple categories such as plastic, paper, metal, glass, clothes, shoes, batteries, and biological waste using a deep learning model.
Why it is useful
Automated waste classification reduces manual sorting effort, improves recycling efficiency, minimizes environmental pollution, and supports smart waste management systems.
General Workflow
Waste image → Image preprocessing & augmentation → RepVGG-MEM deep learning model → Predicted waste category
 (input → processing → model → output)

---

## Dataset Used
👉 **[Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)**

**Dataset Details:**
Dataset Size-15,000+ labeled waste images
Waste Categories- 14 classes including plastic, metal, glass, paper, cardboard, clothes, shoes, batteries, and biological waste
Data Collection-Images collected from public datasets and real-world environments
Image Variations-Different lighting conditions,Varied backgrounds,Multiple object orientations
Data Nature-Includes both recyclable and non-recyclable waste,Presence of class imbalance
Dataset Split-60% training data,20% validation data,20% testing data

---

## Dependencies Used
Python, PyTorch, NumPy, OpenCV, Matplotlib, CUDA


---

## EDA & Preprocessing
Dataset distribution analysis revealed class imbalance, with clothes and glass having higher representation.
Images were resized and center-cropped to 224×224 pixels.
Gaussian blur was applied to reduce noise.
Images were converted to grayscale and enhanced using CLAHE.
Min–Max normalization was applied.
Images were converted back to RGB format.

---

## Model Training Info
Model: RepVGG-MEM
Optimizer: Adam
Learning rate: 0.001
Training strategy:
Mixup data augmentation
Exponential Moving Average (EMA)
Structured magnitude-based pruning
Hardware: NVIDIA GPU
Dataset split: 60% training, 20% validation, 20% testing


---

## Model Testing / Evaluation
The model was evaluated using:
Accuracy
F1-score
Confusion matrix analysis
Testing was performed on unseen data to measure generalization performance, with special attention to visually similar waste categories.


---

## Results
Accuracy: 95.06%
F1-score: 94.5%
Reduced model size and FLOPs after pruning
Strong performance across most waste categories
Minor misclassification between visually similar classes
Limitations & Future Work

---

## Limitations & Future Work
Dataset imbalance affects underrepresented classes
Overlapping or mixed waste objects are not handled
Fixed waste categories require retraining for new classes
Future Work:
Collect more real-world waste images
Improve classification of mixed and damaged waste
Further optimize for low-power edge devices
Integrate vision models with sensor-based systems

---

## Deployment Info
The lightweight and pruned RepVGG-MEM model can be deployed on:
Smart waste bins
Edge devices
Mobile platforms
Real-time automated waste sorting systems


---
