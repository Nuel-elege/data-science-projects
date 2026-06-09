# ============================================================
# COM737 - Political Bias Detection in News Articles
# Using NLP and Deep Learning (LSTM)
# Author: Emmanuel Nonso Elege | University of Ulster, Magee Campus
# Dataset: Political_Bias.csv (~3,458 news articles from multiple sources)
# Task: Multi-class classification of political bias (Left, Lean Left, Center,
#       Lean Right, Right)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from wordcloud import WordCloud
import nltk
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ---- Load Dataset ----
df = pd.read_csv("Political_Bias.csv")
print(df.info())

# ---- Handle Missing Values ----
print(df.isnull().sum())
df = df.dropna(subset=['Text'])

# ---- EDA: Bias Distribution ----
plt.figure(figsize=(8, 5))
sns.countplot(y=df['Bias'], order=df['Bias'].value_counts().index, palette="coolwarm")
plt.title("Distribution of Political Bias Labels", fontsize=14)
plt.xlabel("Count")
plt.ylabel("Bias Category")
plt.show()

print(df['Bias'].value_counts())

# Bias distribution by news source
source_bias_counts = df.groupby(['Source', 'Bias']).size().reset_index(name='count')
top_sources = df['Source'].value_counts().nlargest(30).index
source_bias_counts = source_bias_counts[source_bias_counts['Source'].isin(top_sources)]

plt.figure(figsize=(12, 7))
sns.set(style="whitegrid")
sns.barplot(data=source_bias_counts, x='Source', y='count', hue='Bias', palette='coolwarm')
plt.title("Political Bias by News Source", fontsize=14)
plt.xlabel("News Source")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Bias')
plt.tight_layout()
plt.show()

# ---- Word Clouds per Bias Category ----
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

for bias in df['Bias'].unique():
    text = " ".join(df[df['Bias'] == bias]['Text'])
    wordcloud = WordCloud(stopwords=stopwords, background_color="white",
                          width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(f"Word Cloud for {bias}")
    plt.axis("off")
    plt.show()

# Most common words
def get_most_common_words(texts, n=20):
    words = " ".join(texts).split()
    return Counter(words).most_common(n)

common_words = get_most_common_words(df['Text'])
print("Most Common Words:", common_words)

# ---- Handle Data Imbalance (Oversampling) ----
df['Bias'] = df['Bias'].astype('category').cat.codes
X, y = df['Text'], df['Bias']

oversample = RandomOverSampler(sampling_strategy='auto')
X_resampled, y_resampled = oversample.fit_resample(X.values.reshape(-1, 1), y)

df_balanced = pd.DataFrame({'Text': X_resampled.flatten(), 'Bias': y_resampled})

plt.figure(figsize=(8, 5))
sns.countplot(y=df_balanced['Bias'],
              order=df_balanced['Bias'].value_counts().index,
              palette="coolwarm")
plt.title("Balanced Distribution of Political Bias Labels", fontsize=14)
plt.xlabel("Count")
plt.ylabel("Bias Category (Encoded)")
plt.show()

# ---- Text Cleaning & TF-IDF ----
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)                         # Remove special chars
    text = re.sub(r'\d+', '', text)                         # Remove numbers
    text = text.lower()
    return text

df_balanced['cleaned_text'] = df_balanced['Text'].apply(clean_text)

# TF-IDF transformation (for feature analysis)
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df_balanced['cleaned_text'])
print("TF-IDF Transformation Complete. Shape:", X_tfidf.shape)

# ---- Model Data Preparation (Tokenization & Padding) ----
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df_balanced['cleaned_text'])

X_seq = tokenizer.texts_to_sequences(df_balanced['cleaned_text'])
X_pad = pad_sequences(X_seq, maxlen=300)

X_train, X_test, y_train, y_test = train_test_split(
    X_pad, df_balanced['Bias'], test_size=0.2, random_state=42
)

# ---- LSTM Model Definition ----
num_classes = len(df_balanced['Bias'].unique())

model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=300),
    LSTM(100, dropout=0.3, recurrent_dropout=0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax'),
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# ---- Model Training ----
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

r = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# ---- Evaluate ----
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# ---- Training Curves ----
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(r.history['loss'], label='Train Loss')
plt.plot(r.history['val_loss'], label='Val Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(r.history['accuracy'], label='Train Accuracy')
plt.plot(r.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# ---- Confusion Matrix & Classification Report ----
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=df_balanced['Bias'].unique(),
            yticklabels=df_balanced['Bias'].unique())
plt.title('Confusion Matrix - Political Bias Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print(classification_report(y_test, y_pred_classes))
