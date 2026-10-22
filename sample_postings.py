"""
Prints real examples (3 fake + 3 real) from the TEST part of your Kaggle dataset,
with the model's prediction, and saves them to sample_postings.txt.
Run after train_model.py:   python sample_postings.py
"""
import joblib
from sklearn.model_selection import train_test_split

from train_model import BASE, DATA_PATH, load_and_clean

df = load_and_clean(DATA_PATH)
# same split as training (random_state=42), so these postings were NOT used to train the model
_, X_test, _, y_test = train_test_split(
    df["text"], df["fraudulent"], test_size=0.2, stratify=df["fraudulent"], random_state=42)

model = joblib.load(BASE / "model" / "best_model.joblib")

lines = []
for label, name in [(1, "FAKE"), (0, "REAL")]:
    for i, text in enumerate(X_test[y_test == label].sample(3, random_state=7), 1):
        prob = model.predict_proba([text])[0][1]
        lines.append(f"===== Actual: {name} #{i} | model fake probability: {prob * 100:.1f}% =====\n{text}\n")

out = "\n".join(lines)
(BASE / "sample_postings.txt").write_text(out, encoding="utf-8")
print(out)
print("Saved sample_postings.txt (open it and copy any posting into the app)")
