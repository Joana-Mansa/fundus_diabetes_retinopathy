# fundus_diabetes_retinopathy
In this project, we grade fundus images on a scale of 0-4, towards an automatic system for diagnoising diabetic retinopathy. We evaluate the model with kappa score 

# Diabetic Retinopathy Severity Grading with Explainability

Automated grading of diabetic retinopathy (DR) severity from fundus images using deep learning, with GradCAM visualizations to support clinical interpretability.

## Overview

Diabetic retinopathy is the leading cause of blindness in working-age adults. Early detection through regular screening can prevent vision loss, but access to trained ophthalmologists remains limited globally. This project develops an AI system to automatically grade DR severity from retinal fundus images.

### Task

Grade fundus images into five DR severity levels:

| Grade | Severity | Clinical Meaning |
|-------|----------|------------------|
| 0 | No DR | No visible signs of retinopathy |
| 1 | Mild | Microaneurysms only |
| 2 | Moderate | More than microaneurysms but less than severe |
| 3 | Severe | Extensive intraretinal hemorrhages, venous beading |
| 4 | Proliferative | Neovascularization, vitreous/preretinal hemorrhage |

## Dataset

**APTOS 2019 Blindness Detection** (Kaggle)
- ~3,600 training images
- Images from multiple clinical sites in India
- Varying image quality and acquisition conditions

Source: [Kaggle Competition](https://www.kaggle.com/c/aptos2019-blindness-detection)

## Methodology

### Preprocessing Pipeline

1. **Circle Cropping**: Removes black borders by detecting the retinal region using contour detection
2. **Resizing**: Standardizes all images to 512×512 pixels
3. **Ben Graham Preprocessing**: Local contrast enhancement using Gaussian blur subtraction—a technique proven effective for fundus images in the 2015 Kaggle DR competition.
4.   Reduces variation due to different cameras, exposure, and lighting, making the dataset more homogeneous and improving model robustness.
5.   Enhances fine retinal details (vessels, microaneurysms, exudates), which can improve CNN performance for diabetic retinopathy grading and other fundus tasks

```python
def apply_ben_graham_preprocessing(image, sigmaX=30):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX)
    processed = cv2.addWeighted(image, 4, blurred, -4, 128)
    return processed
```

### Model Architecture

- **Backbone**: EfficientNet-B6 (pretrained)
- **Modification**: BatchNorm layers replaced with GroupNorm for stability with small batch sizes
- **Head**: Single linear layer outputting continuous severity score
- **Approach**: Regression (MSE loss) treating grades as ordinal, then threshold optimization for final classification

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 512×512 |
| Batch Size | 4 |
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Scheduler | Cosine Annealing |
| Weight Decay | 1e-5 |
| Early Stopping | 5 epochs patience |

### Data Augmentation

- Random rotation (±40°)
- Horizontal flip
- Color jitter (brightness, contrast, saturation, hue)
- ImageNet normalization

### Threshold Optimization

Rather than rounding regression outputs directly, we optimize decision thresholds using Nelder-Mead to maximize quadratic weighted kappa on the validation set:

```
prediction < t0       → Grade 0
t0 ≤ prediction < t1  → Grade 1
t1 ≤ prediction < t2  → Grade 2
t2 ≤ prediction < t3  → Grade 3
prediction ≥ t3       → Grade 4
```

### Evaluation Metric

**Quadratic Weighted Kappa (QWK)**: Measures agreement between predicted and actual grades, accounting for the ordinal nature of the task. Penalizes predictions further from the true grade more heavily.

## Explainability

### GradCAM Visualization

To support clinical interpretability, we generate Gradient-weighted Class Activation Maps (GradCAM) showing which image regions influence the model's predictions.

This addresses a key requirement for clinical AI: the ability for clinicians to understand and verify the basis for automated diagnoses.

**Implementation**: Uses the `pytorch-grad-cam` library targeting the final convolutional layer of EfficientNet.

## Results

| Metric | Value |
|--------|-------|
| Validation Kappa (rounded) | 0.8916 |
| Validation Kappa (optimized thresholds) | 0.9140 |
| Optimized Thresholds | [0.55, 1.51, 2.50, 3.17] |

### GradCAM Examples by Grade

*To be added: visualization grid showing original images and attention maps for each severity level*

## Project Structure

```
├── dr_classification.py    # Main training and evaluation script
├── README.md               # This file
├── gradcam_by_grade.png    # Explainability visualizations (after training)
├── best_model_kappa_512.pth # Trained model weights (after training)
└── submission.csv          # Test set predictions (after training)
```

## Usage

### Requirements

```
torch
timm
albumentations
opencv-python
pandas
numpy
scikit-learn
scipy
tqdm
pytorch-grad-cam
matplotlib
```

### Running on Kaggle

1. Create a new Kaggle notebook
2. Add the APTOS 2019 dataset as input
3. Copy `dr_classification.py` contents into a code cell
4. Run all cells

### Local Execution

```bash
# Adjust paths in the script first
python dr_classification.py
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Regression + threshold optimization | Better handles ordinal nature of DR grades than pure classification |
| GroupNorm over BatchNorm | More stable with small batch sizes required for large images |
| Ben Graham preprocessing | Proven effective for fundus images; enhances vessel visibility |
| EfficientNet-B6 | Strong accuracy/efficiency tradeoff; pretrained weights transfer well to medical imaging |
| GradCAM explainability | Essential for clinical trust; validates model attends to relevant pathology |

## Limitations

- Trained on single dataset (APTOS 2019); domain shift expected on other populations
- No external validation on separate clinical datasets
- Image quality variation in source data may affect reliability
- Binary quality assessment not implemented

## Future Work

- Cross-dataset validation (MESSIDOR, IDRiD, EyePACS)
- Domain adaptation techniques for multi-center deployment
- Image quality filtering pipeline
- Multi-task learning for concurrent pathology detection (glaucoma, AMD, cataract)

## References

1. Graham, B. (2015). Kaggle Diabetic Retinopathy Detection Competition Report.
2. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
3. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.

## Author

Joana Owusu-Appiah

MSc Medical Imaging and Applications (Erasmus Mundus)

## License

MIT License - Free for academic and research use.
