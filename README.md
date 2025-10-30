# MushroomClassifier

Machine learning project for classifying mushroom edibility based on the [Mushroom Edibility Classification dataset](https://www.kaggle.com/datasets/devzohaib/mushroom-edibility-classification).

## Project Setup

1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle and place `secondary_data.csv` in the project root (already included here).
4. Train the model:
   ```powershell
   python src/train_model.py --data-path secondary_data.csv --output-dir models
   ```
5. Run CLI inference on a JSON payload:
   ```powershell
   python src/predict.py --input-json examples/sample_input.json
   ```
6. Launch the Streamlit web app locally:
   ```powershell
   streamlit run app/streamlit_app.py
   ```

## Project Structure

```
MushroomClassifier/
├── app/                # Streamlit web application
├── models/             # Serialized models and preprocessing artifacts
├── notebooks/          # Exploratory notebooks
├── src/                # Training and utility scripts
├── secondary_data.csv  # Dataset from Kaggle (semicolon separated)
├── requirements.txt
└── README.md
```

## Workflow

1. Exploratory data analysis and preprocessing in `notebooks/`.
2. Train the predictive model with scripts under `src/`.
3. Export the trained model and preprocessing pipeline to `models/`.
4. Deploy the Streamlit app located in `app/` for interactive predictions.

## Reporting Checklist

- Summarize dataset provenance (cite Kaggle source and metadata file).
- Document preprocessing decisions (imputation, encoding, scaling).
- Describe model selection process and hyper-parameters.
- Include evaluation metrics (accuracy, recall, F1, ROC-AUC) and confusion matrix.
- Reflect on limitations and ethical considerations when classifying mushrooms.

Refer to the course report template to document methodology and findings.
