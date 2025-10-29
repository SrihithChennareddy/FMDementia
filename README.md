## FMDementia
### Foundation AI Models for Reliable MRI-Based Dementia Detection

This project implements a reliable MRI-based Dementia detection framework using pre-trained foundation AI models such as Vision Transformers (ViT, 3D-Swin Transformer) and large-scale medical imaging encoders (Uni-Encoder, MedSAM). 

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
pip install pandas numpy scikit-learn torch tqdm transformers datasets accelerate
```

#### Run the model

```bash
python models/model.py
```

#### Deactivate the environment when done

```bash
deactivate
```

### Sample Output

#### Experimental results on OASIS-1 and OASIS-2 datasets using 5-fold cross-validation

| Dataset    |	Accuracy |	Precision |	Recall    |	F1-score |	AUC-ROC |
|------------|-----------|------------|-----------|----------|----------|
| OASIS-1	   |  92.5%	   |  91.8%	    |  91.0%	  |  91.4%	 |  95.0%   |
| OASIS-2	   |  90.5%	   |  89.8%	    |  89.2%	  |  89.4%	 |  89.0%   |
| **Average**|  91.5%	   |  90.8%	    |  90.1%	  |  90.4%	 |  92.0%   |

#### Cross-Domain Evaluation

| Train / Test      |	 Accuracy |	Precision |	Recall    |	F1-score   |	AUC-ROC |
|-------------------|-----------|-----------|-----------|------------|----------|
| OASIS-1 / OASIS-2 |  89.2%	  |  88.5%	  |  87.8%	  |  88.1%	   |  92.5%   |
| OASIS-2 / OASIS-1	|  90.1%	  |  89.4%	  |  88.7%	  |  89.0%	   |  91.3%   |

### License

MIT License

### Author

Srihith Chennareddy

