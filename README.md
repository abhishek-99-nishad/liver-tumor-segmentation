# Liver Tumor Segmentation using U-Net

This project performs liver tumor segmentation on 3D CT scans using a 2D U-Net model in TensorFlow.

## Steps:
- Preprocessing of NIfTI CT & mask files
- Training U-Net with slice-wise data
- Predicting tumor masks
- Evaluation (Precision, Recall, F1, Dice, Confusion Matrix)

## Dependencies
- TensorFlow
- NumPy
- OpenCV
- matplotlib
- scikit-learn
- nibabel
