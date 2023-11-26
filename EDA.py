import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob

# Step 2: Load Data
with open('cleaned_data.jsonl', 'r') as file:
    lines = file.readlines()

data = [json.loads(line) for line in lines]

# Step 3: Explore the Data
df = pd.DataFrame(data)
print("Basic Statistics:")
print(df.describe())

# Step 4: Visualizations
# Example: Bar chart of output values
plt.figure(figsize=(8, 5))
sns.countplot(x='output', data=df)
plt.title('Distribution of Output Values')
plt.show()

# Step 5: Text Analysis
# Example: Word cloud for the most frequent words in questions
questions_text = ' '.join(df['input'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(questions_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Questions')
plt.show()

# Example: Sentiment analysis for answers
df['sentiment'] = df['output'].apply(lambda x: TextBlob(x).sentiment.polarity)
plt.figure(figsize=(8, 5))
sns.histplot(df['sentiment'], kde=True)
plt.title('Distribution of Sentiment in Answers')
plt.show()

# Step 6: Statistical Analysis
# Example: Correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(8, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Step 7: Machine Learning (Example: Text Classification with Dummy Data)
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, classification_report

# # Dummy data for text classification
# df['label'] = (df['output'] != '-').astype(int)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df['input'], df['label'], test_size=0.2, random_state=42)

# # Convert text to numerical features using CountVectorizer
# vectorizer = CountVectorizer()
# X_train_vectorized = vectorizer.fit_transform(X_train)
# X_test_vectorized = vectorizer.transform(X_test)

# # Train a simple Naive Bayes classifier
# classifier = MultinomialNB()
# classifier.fit(X_train_vectorized, y_train)

# # Predict on the test set
# y_pred = classifier.predict(X_test_vectorized)

# # Evaluate the classifier
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))
