Traffic Sign Detection & Classification

Approach: Transfer Learning with MobileNetV2 + Fine-Tuning
Dataset: 43 Traffic Sign Classes | 39,209 Train Images | 12,630 Test Images
Final Accuracy: 90%+

Overview
An end-to-end deep learning pipeline to detect and classify traffic signs across 43 categories using Transfer Learning with MobileNetV2 as the backbone architecture. The model is trained in two phases — frozen base feature extraction followed by selective fine-tuning — achieving over 90% test accuracy on the German Traffic Sign Recognition Benchmark (GTSRB) dataset. This project is directly applicable to real-world autonomous driving and Advanced Driver Assistance Systems (ADAS).

Problem Statement
Manual identification of traffic signs is impractical for autonomous vehicles and intelligent transportation systems. The goal was to build a highly accurate image classification model that can identify any of 43 traffic sign types from raw image input without relying on handcrafted features or traditional computer vision techniques.

Dataset

Source: German Traffic Sign Recognition Benchmark (GTSRB)
Train images: 39,209 across 43 classes
Test images: 12,630 across 43 classes
Format: Class-wise image folders + Train.csv, Test.csv, Meta.csv


Tech Stack

Python, TensorFlow, Keras
MobileNetV2 (ImageNet pretrained weights)
ImageDataGenerator, flow_from_dataframe
NumPy, Pandas, Matplotlib, Seaborn
Scikit-learn (classification_report, confusion_matrix)
Google Colab


Project Workflow
Step 1 — Data Loading and EDA
Loaded Train.csv, Test.csv, and Meta.csv files. Performed class distribution analysis and visualized sample images from each of the 43 traffic sign categories to understand visual variation across classes.
Step 2 — Preprocessing and Augmentation
Images were resized to 100×100 pixels and normalized using MobileNetV2's preprocess_input function (pixel range [-1, 1]). Data augmentation was applied on the training set only — random rotation (±10°), width/height shifts (10%), and zoom (10%). Horizontal flipping was intentionally excluded since traffic signs are directional and flipping would create misleading training samples.
Step 3 — Model Architecture
MobileNetV2 pretrained on ImageNet was used as the feature extraction backbone with its top classification layers removed. A custom classification head was built on top: GlobalAveragePooling2D → BatchNormalization → Dense(256, ReLU) → Dropout(0.4) → Dense(128, ReLU) → Dropout(0.3) → Dense(43, Softmax).
Step 4 — Phase 1 Training (Frozen Base)
All MobileNetV2 base layers were frozen. Only the custom head was trained using the Adam optimizer (learning rate = 1e-3). Callbacks used: EarlyStopping (patience=4), ReduceLROnPlateau (factor=0.5), and ModelCheckpoint to save the best model. End of Phase 1 accuracy: ~68%.
Step 5 — Phase 2 Fine-Tuning
The last 30 layers of the MobileNetV2 base were unfrozen and retrained alongside the classification head using a much smaller learning rate (1e-5). This allowed the higher-level feature detectors to adapt to traffic sign-specific visual patterns. Accuracy improved from ~68% to 90%+.
Step 6 — Evaluation
Model evaluated on 12,630 test images using test accuracy, a 43×43 confusion matrix, and a full classification report (precision, recall, F1-score per class).

Results
PhaseAccuracy:
                Phase 1 (Frozen base)~68%
                Phase 2 (Fine-tuned)90%+

Key Design Decisions

MobileNetV2 over training from scratch: ImageNet pretraining provides general visual features that transfer strongly to traffic signs, drastically reducing training time and data requirements.
No horizontal flip augmentation: Traffic signs are directional. Flipping a left-turn sign creates an invalid right-turn sample that corrupts model learning.
flow_from_dataframe: Loads images on-the-fly from disk using CSV path mappings — memory-efficient for large image datasets.
Two-phase training: Protects pretrained weights during initial head training, then carefully refines them with a very small learning rate during fine-tuning.


Real-World Relevance
Traffic sign recognition is a core perception task in autonomous vehicle systems, ADAS, smart traffic infrastructure, and road safety analytics. The techniques applied — Transfer Learning, fine-tuning, domain-aware augmentation, and callback-driven training — directly reflect how production computer vision systems are built in industry.
