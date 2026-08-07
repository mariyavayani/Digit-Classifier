# 🔢 Handwritten Digit Classifier

A machine learning web app that classifies handwritten digits (0–9) using 
a Logistic Regression model trained on the scikit-learn digits dataset.

**Live demo:** 

## How It Works
- Trains on 1,437 labeled 8x8 pixel images of handwritten digits
- Evaluates on 360 unseen test images
- Achieves ~97.5% test accuracy
- Visualizes model performance with a confusion matrix

## Tech Stack
- Python
- scikit-learn — model training and evaluation
- Streamlit — web app / UI
- Matplotlib — visualizations

## How to Run Locally
\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`
