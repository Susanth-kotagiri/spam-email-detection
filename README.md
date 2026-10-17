# spam-email-detection
Spam Email Detection System: classifies messages as Spam or Not Spam using TF-IDF features and a scikit-learn classifier. Reaches 96.23% test accuracy on the 5,572-message SMS dataset.



# Spam Email Detection System

A machine learning project that classifies messages as **Spam** or **Not Spam (Ham)**. Text is converted into numeric features with TF-IDF and fed to a classifier, which is evaluated on held-out test data and can predict the label of any new message.

![Python](https://img.shields.io/badge/Python-3.x-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![Pandas](https://img.shields.io/badge/Pandas-data-lightgrey)

## Results

| Metric | Value |
|---|---|
| Training accuracy | 96.61% |
| Test accuracy | 96.23% |
| Dataset size | 5,572 messages |
| Train / test split | 4,457 / 1,115 (80% / 20%) |
| TF-IDF features | 7,510 |

Accuracy is measured on messages the model never saw during training. Spam is the minority class in this dataset, so accuracy alone can overstate performance; precision, recall and a confusion matrix are listed under *Future work*.

## How it works

1. **Load** the labelled SMS dataset (`spam.csv`) with Pandas.
2. **Prepare** the data: rename the columns to `Category` and `Message`, and encode the labels (spam = 0, ham = 1).
3. **Split** into training and test sets (80% / 20%, `random_state=3`).
4. **Vectorize** the text with `TfidfVectorizer` (English stop words removed, lowercased). The vectorizer is fitted on the training data only, then applied to the test data to avoid data leakage.
5. **Train** a Logistic Regression classifier.
6. **Evaluate** accuracy on the training and test sets.
7. **Predict** the label of a new message with the same fitted vectorizer.

## Example

```python
mail = ["Congratulations! You have won a FREE $1000 gift card, click here now to claim your prize before it expires!"]
features = feature_extraction.transform(mail)
prediction = model.predict(features)
print("HAM MAIL" if prediction[0] == 1 else "SPAM MAIL")
# SPAM MAIL
```

## Tech stack

- Python
- Pandas, NumPy
- scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `train_test_split`, `accuracy_score`)
- Jupyter Notebook

## Getting started

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install pandas numpy scikit-learn jupyter
jupyter notebook email_spam_classifier.ipynb
```

Place `spam.csv` (the SMS Spam Collection dataset, with columns `v1` for the label and `v2` for the text) in the same folder as the notebook. Load it with `encoding="latin-1"`.

## Project structure

```
├── email_spam_classifier.ipynb   # full pipeline with outputs
├── spam.csv                      # dataset (add your own copy)
└── README.md
```

## Limitations

- TF-IDF ignores word order, and spammers change their wording over time, so the model needs retraining on fresh data.
- Results apply to this dataset and split; other kinds of email may behave differently.
- Short, hand-written test messages may be misclassified, since the model learned from SMS-style text.

## Future work

- Add a confusion matrix with precision, recall and F1-score.
- Compare Logistic Regression with Multinomial Naive Bayes.
- Compare CountVectorizer with TF-IDF, and unigrams with bigrams.
- Build a small interface where a user pastes a message and sees the prediction.
