# Quick Start Guide

Get started with the Retina-Based Heart Disease Predictor in 5 steps!

## Step 1: Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Prepare Dataset

Organize your images:
```
data/raw/
├── normal/
│   └── [put normal images here]
└── disease/
    └── [put disease images here]
```

## Step 3: Train Model

```bash
python src/train.py
```

Wait for training to complete. Model will be saved to `models/retina_heart_disease_model.h5`

## Step 4: Evaluate Model (Optional)

```bash
python src/evaluate.py
```

## Step 5: Run Web App

```bash
python app.py
```

Open browser: http://localhost:5000

Upload an image and get predictions!

---

**That's it!** You now have a working heart disease predictor from retinal images.

For detailed documentation, see README.md

