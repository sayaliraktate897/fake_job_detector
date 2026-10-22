# Fake Job Posting Detector

Machine learning project that predicts whether a job posting is **real or fake**, with a Streamlit web app.

## Dataset
Kaggle: "Real or Fake Job Posting Prediction" (EMSCAD), about 18,000 postings, around 5% fake.
Download `fake_job_postings.csv` and place it in the `data/` folder.

## Tech
Python, Pandas, NumPy, Scikit-learn (TF-IDF, Logistic Regression, Random Forest), Plotly, Streamlit.

## How it works
1. Clean data and join title, company profile, description, requirements and benefits into one text column.
2. Convert text to features with TF-IDF (1-2 word n-grams).
3. Handle class imbalance with balanced class weights.
4. Train Logistic Regression and Random Forest on an 80/20 stratified split.
5. Compare precision, recall, F1 and ROC-AUC on the fake class, save the best model.
6. Streamlit app: paste a posting to get a real/fake prediction with probability.

## Run
```
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Results
(Fill in after training: accuracy, precision, recall, F1 of the best model.)

Developed by Sayali Dattatray Raktate
