# %% [markdown]
# ### Mengimport Library

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import joblib
from urllib.parse import urlparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# %% [markdown]
# ### Loading Data

# %%
df = pd.read_csv("phishingURL_detection_dataset.csv")

# %%
print("Preview data:")
print(df.head())

# %%
print("\nDataset info:")
print(df.info())

# %% [markdown]
# Output menunjukkan bahwa dataset memiliki 58645 entri dan 20 kolom. Semua kolom memiliki tipe data int64 dan tidak ada nilai null (58645 non-null entries untuk semua kolom).

# %%
print("\nDescriptive statistics:")
print(df.describe())

# %% [markdown]
# ### Exploratory Data Analysis (EDA)

# %% [markdown]
# #### Memeriksa dan menampilkan jumlah nilai yang hilang (missing values) untuk setiap kolom dalam DataFrame.

# %%
print("\nMissing values:")
print(df.isnull().sum())

# %% [markdown]
# Tidak ada nilai yang hilang di semua kolom.

# %% [markdown]
# #### Menghitung dan menampilkan jumlah baris duplikat dalam DataFrame.

# %%
print("\nNumber of duplicates:", df.duplicated().sum())

# %% [markdown]
# Terdapat 44797 baris duplikat. Ini adalah jumlah yang sangat signifikan. Duplikat tidak akan ditanganin karena variasi fitur numerik yang memungkinkan adalanya link URL dengan jumlah elemen yang sama.

# %% [markdown]
# #### Memeriksa distribusi fitur numerik

# %%
df.hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Memeriksa distribusi label phishing dan legimate URL

# %%
# Target distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='phishing', data=df)
plt.title('Distribution of Phishing vs Legitimate URLs')
plt.xlabel('Phishing (1) vs Legitimate (0)')
plt.ylabel('Count')
plt.show()

# Percentage distribution
target_dist = df['phishing'].value_counts(normalize=True) * 100
print("\nTarget distribution (%):")
print(target_dist)

# %% [markdown]
# #### Memeriksa korelasi antar fitur dengan label

# %%
plt.figure(figsize=(15, 12))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# %% [markdown]
# #### Mengidentifikasi 10 fitur teratas

# %%
top_features = corr_matrix['phishing'].abs().sort_values(ascending=False).index[1:11]
print("\n Features most correlated with target:")
print(top_features)

# %% [markdown]
# #### Memeriksa distribusi fitur dengan kelas target

# %%
plt.figure(figsize=(15, 10))
for i, feature in enumerate(top_features, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(x='phishing', y=feature, data=df)
    plt.title(f'{feature} distribution')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Data Preprocessing

# %% [markdown]
# #### Feature Engineering Consideration

# %%
df['special_char_ratio'] = (df['qty_dot_url'] + df['qty_hyphen_url'] + df['qty_underline_url'] + 
                           df['qty_slash_url'] + df['qty_questionmark_url'] + df['qty_equal_url'] + 
                           df['qty_at_url'] + df['qty_and_url'] + df['qty_exclamation_url'] + 
                           df['qty_space_url'] + df['qty_tilde_url'] + df['qty_comma_url'] + 
                           df['qty_plus_url'] + df['qty_asterisk_url'] + df['qty_hashtag_url'] + 
                           df['qty_dollar_url'] + df['qty_percent_url']) / df['length_url']

print("\nCorrelation of new feature with target:", df['special_char_ratio'].corr(df['phishing']))

# %%
X = df.drop('phishing', axis=1)
y = df['phishing']

# %%
selector = SelectKBest(f_classif, k=15)
X_selected = selector.fit_transform(X, y)

# %%
selected_features = X.columns[selector.get_support()]
print("\nSelected features:")
print(selected_features)

X = X[selected_features]

# %% [markdown]
# ### Splitting Data

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTrain set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# %% [markdown]
# #### Handling Imbalance Class

# %%
print("\nTraining set class distribution:")
print(y_train.value_counts(normalize=True))

# %%
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# %%
print("\nAfter SMOTE class distribution:")
print(pd.Series(y_train_res).value_counts(normalize=True))

# %% [markdown]
# Output menunjukkan bahwa kelas-kelas pada set pelatihan sekarang seimbang (50% untuk kelas 0 dan 50% untuk kelas 1). Ini akan membantu model belajar dari kedua kelas secara lebih merata.

# %% [markdown]
# #### Feature Scaling

# %%
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train_res)

X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ### Modeling

# %% [markdown]
# #### 1. Logistic Regression Model

# %%
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train_res)

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

# %% [markdown]
# #### Evaluasi hasil training Logistic Regression Model

# %%
print("\nLogistic Regression Performance:")
print(classification_report(y_test, y_pred_lr))
print("Accuracy:", accuracy_score(y_test, y_pred_lr))

# %%
cm = confusion_matrix(y_test, y_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

plt.title('Logistic Regression Confusion Matrix')
plt.show()

# %%
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

RocCurveDisplay.from_estimator(lr, X_test_scaled, y_test)
plt.title('Logistic Regression ROC Curve')
plt.show()


# %% [markdown]
# #### 2. Support Vector Machine Model

# %%
svm = SVC(probability=True, random_state=42)
svm.fit(X_train_scaled, y_train_res)

y_pred_svm = svm.predict(X_test_scaled)
y_prob_svm = svm.predict_proba(X_test_scaled)[:, 1]

# %% [markdown]
# #### Evaluasi hasil training SVM Model

# %%
print("\nSVM Performance:")
print(classification_report(y_test, y_pred_svm))
print("Accuracy:", accuracy_score(y_test, y_pred_svm))

# %%
ConfusionMatrixDisplay.from_estimator(svm, X_test_scaled, y_test, cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.show()

# %%
RocCurveDisplay.from_estimator(svm, X_test_scaled, y_test)
plt.title('SVM ROC Curve')
plt.show()

# %% [markdown]
# #### 3. Random Forest Model

# %%
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False]
}

# %%
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                          cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train_res)

# %%
print("\nBest parameters:", grid_search.best_params_)

# %%
best_rf = grid_search.best_estimator_

y_pred_rf = best_rf.predict(X_test_scaled)
y_prob_rf = best_rf.predict_proba(X_test_scaled)[:, 1]

# %% [markdown]
# #### Evaluasi hasil training Random Forest Model

# %%
print("\nRandom Forest Performance:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# %%
ConfusionMatrixDisplay.from_estimator(best_rf, X_test_scaled, y_test, cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.show()

# %%
RocCurveDisplay.from_estimator(best_rf, X_test_scaled, y_test)
plt.title('Random Forest ROC Curve')
plt.show()

# %% [markdown]
# #### Menganalisis hasil evaluasi fitur yang paling berpengaruh dalam proses training model

# %%
feature_imp = pd.Series(best_rf.feature_importances_, index=selected_features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# %% [markdown]
# ### Best Model Selection

# %%
models = {
    'Logistic Regression': y_prob_lr,
    'SVM': y_prob_svm,
    'Random Forest': y_prob_rf
}

# %%
plt.figure(figsize=(8, 6))
for name, y_prob in models.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.show()

# %%
plt.figure(figsize=(8, 6))
for name, y_prob in models.items():
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label=f'{name}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend(loc="lower left")
plt.show()

# %%
best_model = best_rf
joblib.dump(best_model, 'phishing_detection_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'feature_selector.pkl')

# %% [markdown]
# ### Hasil Inference

# %%
scaler = joblib.load('scaler.pkl')
selector = joblib.load('feature_selector.pkl')
model = joblib.load('phishing_detection_rf_model.pkl')

# %%
def extract_features(url):
    parsed = urlparse(url)
    
    features = {
        'qty_dot_url': url.count('.'),
        'qty_hyphen_url': url.count('-'),
        'qty_underline_url': url.count('_'),
        'qty_slash_url': url.count('/'),
        'qty_questionmark_url': url.count('?'),
        'qty_equal_url': url.count('='),
        'qty_at_url': url.count('@'),
        'qty_and_url': url.count('&'),
        'qty_exclamation_url': url.count('!'),
        'qty_space_url': url.count(' '),
        'qty_tilde_url': url.count('~'),
        'qty_comma_url': url.count(','),
        'qty_plus_url': url.count('+'),
        'qty_asterisk_url': url.count('*'),
        'qty_hashtag_url': url.count('#'),
        'qty_dollar_url': url.count('$'),
        'qty_percent_url': url.count('%'),
        'qty_tld_url': len(parsed.netloc.split('.')[-1]) if parsed.netloc else 0,
        'length_url': len(url),
        'special_char_ratio': (url.count('.') + url.count('-') + url.count('_') + 
                              url.count('/') + url.count('?') + url.count('=') + 
                              url.count('@') + url.count('&') + url.count('!') + 
                              url.count(' ') + url.count('~') + url.count(',') + 
                              url.count('+') + url.count('*') + url.count('#') + 
                              url.count('$') + url.count('%')) / len(url) if len(url) > 0 else 0
    }
    
    return pd.DataFrame([features])

# %%
def predict_url(url):
    try:
        features_df = extract_features(url)
        
        selected_features = selector.transform(features_df)
        
        scaled_features = scaler.transform(selected_features)
        
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)
        
        result = {
            'url': url,
            'prediction': 'Phishing' if prediction[0] == 1 else 'Legitimate',
            'phishing_probability': f"{probability[0][1]*100:.2f}%",
            'legitimate_probability': f"{probability[0][0]*100:.2f}%"
        }
        
        return result
    
    except Exception as e:
        return {'error': str(e)}

# %%
if __name__ == "__main__":
    test_urls = [
        "www.google.com",
        "http://paypal-verify-account.com/login.php",
        "hhttps://link.dana.id/kaget?c=snqw25mpx&r=b7NAEX",
        "https://linkdanaa-kaget.webssit3.my.id/int.html"
        "https://1xlite-9231274.top/id/registration?type=email&bonus=SPORT"
    ]
    
    print("Phishing URL Detection Results:")
    print("="*50)
    
    for url in test_urls:
        result = predict_url(url)
        if 'error' not in result:
            print(f"\nURL: {result['url']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Phishing Probability: {result['phishing_probability']}")
            print(f"Legitimate Probability: {result['legitimate_probability']}")
            print("-"*50)
        else:
            print(f"Error processing URL: {url}")
            print(f"Error: {result['error']}")


