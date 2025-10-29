## FMDementia
### Foundation AI Models for Reliable MRI-Based Dementia Detection

This project implements a reliable and interpretable MRI-based Dementia detection framework using pre-trained foundation AI models such as Vision Transformers (ViT, 3D-Swin Transformer) and large-scale medical imaging encoders (Uni-Encoder, MedSAM). It supports single- and multi-site MRI datasets, cross-domain evaluation, and interpretable predictions using attention maps and saliency masks.

### Features

- Preprocessing and standardized MRI pipeline (skull stripping, intensity normalization, affine registration)
- Diagnosis labeling for OASIS-1 and OASIS-2 datasets
- Support for binary classification: Dementia (DM) vs Healthy Control (HC)
- Multimodal feature extraction using foundation AI models: ViT, 3D-Swin Transformer, Uni-Encoder, MedSAM
- Stratified train-test splitting and 5-fold cross-validation
- Automatic evaluation of Accuracy, Precision, Recall, F1-score, and AUC-ROC
- Cross-domain adaptation for generalizability across heterogeneous datasets
- Interpretability via attention maps and Grad-CAM visualizations

### Dependencies

- Python 3.11+
- PyTorch
- Transformers
- scikit-learn
- pandas
- numpy
- tqdm
- openpyxl

### Quick Start

#### Navigate to project folder

```bash
cd /Users/srihith/src/FMDementia/src
```

#### Create a virtual environment

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv /Users/srihith/src/FMDementia/src/venv
```

#### Activate the virtual environment

```bash
source /Users/srihith/src/FMDementia/src/venv/bin/activate
```

#### Check Python version

```bash
python --version
```

#### Install required packages

```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn torch tqdm transformers datasets accelerate openpyxl
```

#### Run the model

```bash
python model.py
```

#### Deactivate the environment when done

```bash
deactivate
```

### Sample Output

#### Label Summary

-------------
Label |	Count
-------------
HC	  | 100
DM	  | 64

#### 5-Fold Cross-Validation Performance (OASIS-1 / OASIS-2)

#### Dataset |	Accuracy |	Precision |	Recall    |	F1-score |	AUC-ROC |
|------------|-----------|------------|-----------|----------|----------|
| OASIS-1	   |  92.5%	   |  91.8%	    |  91.0%	  |  91.4%	 |  95.0%   |
| OASIS-2	   |  90.5%	   |  89.8%	    |  89.2%	  |  89.4%	 |  89.0%   |
| **Average**|  91.5%	   |  90.8%	    |  90.1%	  |  90.4%	 |  92.0%   |

#### Cross-Domain Evaluation

#### Train / Test   |	 Accuracy |	Precision |	Recall    |	F1-score   |	AUC-ROC |
|-------------------|-----------|-----------|-----------|------------|----------|
| OASIS-1 / OASIS-2 |  89.2%	  |  88.5%	  |  87.8%	  |  88.1%	   |  92.5%   |
| OASIS-2 / OASIS-1	|  90.1%	  |  89.4%	  |  88.7%	  |  89.0%	   |  91.3%   |

### Adding New Datasets

#### Add MRI scans to a new dataset folder.

Update get_diagnosis_label() in model.py with rules for the new dataset.

Ensure labels match supported classes: DM, HC.

Run python model.py to include the dataset in training and evaluation.

### License

MIT License

### Author

Srihith Chennareddy

