import streamlit as st
import pandas as pd
import numpy as np
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load spaCy model
nlp = spacy.load('en_core_web_lg')

# Read data
df_review = pd.read_excel("E:\\G2 project\\reviews_l_h.xlsx")

review_name_col = 'review'  

# Start aspect-based sentiment analysis
df_review = pd.DataFrame({
    'review': df_review[review_name_col].astype(str)  # Convert to string type to handle non-string data
})

# Settings to match your data
app_name = "Puneet"
features = 20  # Number of features to extract
review_name_col = 'review'
language_of_review = 'english'

# Convert the text to lowercase and remove punctuation and white space
df_review['processed_review'] = df_review['review'].str.lower().str.replace("'", '', regex=True).str.replace('[^\w\s]', ' ', regex=True).str.replace(" \d+", " ", regex=True).str.replace(' +', ' ', regex=True).str.strip()

# Tokenize string
df_review['tokenized'] = df_review['processed_review'].apply(lambda review: nltk.word_tokenize(review))

# Read custom stop words from file
with open('E:\\G2 project\\stopwords.txt', 'r') as file:
    custom_stopwords = file.readlines()
    custom_stopwords = [word.strip() for word in custom_stopwords]

# Remove stopwords
stop_words = stopwords.words(language_of_review)
stop_words.extend(custom_stopwords)
df_review['remove_stopwords'] = df_review['tokenized'].apply(lambda tokens: [token for token in tokens if token not in stop_words])

# Lemmatize words
wordnet_lemmatizer = WordNetLemmatizer()
df_review['lemmatized'] = df_review['remove_stopwords'].apply(lambda tokens: [wordnet_lemmatizer.lemmatize(token) for token in tokens])

# Initialize the count vectorizer and join the processed data to be a vectorized
vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
vectors = []
for index, row in df_review.iterrows():
    vectors.append(", ".join(row['lemmatized']))
vectorized = vectorizer.fit_transform(vectors)

# Initialize LDA model
lda_model = LatentDirichletAllocation(n_components=features, random_state=10, evaluate_every=-1, n_jobs=-1)
lda_output = lda_model.fit_transform(vectorized)

# Get dominant topic for each document
dominant_topic = (np.argmax(lda_output, axis=1) + 1)
df_review['dominant_topic'] = dominant_topic

# Get keywords the LDA extracted from review
topic_keywords = lda_model.components_
topic_names = []
for topic_idx, topic in enumerate(topic_keywords):
    topic_names.append(f"Topic {topic_idx+1}: {', '.join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-features - 1:-1]])}")

df_review_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=[f"Topic {i+1}" for i in range(lda_model.n_components)])

# Combine dominant topic with original dataframe
df_review = df_review.reset_index()
df_review = pd.merge(df_review, df_review_document_topic, left_index=True, right_index=True, how='outer')

# Get the most used features (top keywords for each topic)
feature_names = vectorizer.get_feature_names_out()
top_features_per_topic = []
for topic_index in range(lda_model.n_components):
    top_features = [feature_names[i] for i in np.argsort(lda_model.components_[topic_index])[::-1][:features]]
    top_features_per_topic.append(", ".join(top_features))

# Create a new DataFrame with the top features
most_used_features = pd.DataFrame({'Topic': topic_names, 'Top Features': top_features_per_topic})

# Layout using Streamlit
st.title("User Reviews Detailed Analysis and Future Predictions")

# 1. Dominant Topic Distribution
st.header("1. Dominant Topic Distribution")
topic_counts = df_review['dominant_topic'].value_counts()
fig1 = plt.figure(figsize=(8, 6))
topic_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Dominant Topics')
st.pyplot(fig1)

# 2. Top Features per Topic
st.header("2. Top Features per Topic")
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.barh(topic_names, [len(topic.split(', ')) for topic in most_used_features['Top Features']])
ax2.set_xlabel('Number of Top Features')
ax2.set_ylabel('Topic')
ax2.set_title('Top Features per Topic')
st.pyplot(fig2)

# 3. Sentiment Score Distribution
st.header("3. Sentiment Score Distribution")
fig3 = plt.figure(figsize=(8, 6))
df_review['dominant_topic'].hist(bins=20)
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Count')
st.pyplot(fig3)

# 4. Sentiment Trends over Time
st.header("4. Sentiment Trends over Time")
if 'timestamp' in df_review.columns:
    df_review['timestamp'] = pd.to_datetime(df_review['timestamp'])
    fig4 = plt.figure(figsize=(12, 6))
    df_review.groupby(pd.Grouper(key='timestamp', freq='W'))['dominant_topic'].mean().plot()
    plt.title('Sentiment Trends over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    st.pyplot(fig4)

# 5. Word Cloud
st.header("5. Word Cloud")
text = ' '.join(df_review['processed_review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
fig5, ax5 = plt.subplots(figsize=(12, 6))
ax5.imshow(wordcloud)
ax5.axis('off')
plt.title('Word Cloud of Review Text')
st.pyplot(fig5)

# 6. Correlation Matrix
st.header("6. Correlation Matrix")
corr_matrix = df_review_document_topic.corr()
fig6, ax6 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd')
plt.title('Correlation Matrix of Topics')
st.pyplot(fig6)

# 7. Confusion Matrix
st.header("7. Confusion Matrix")
y_true = df_review['dominant_topic']
y_pred = df_review['dominant_topic']  # Replace with your predicted values
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=topic_names)
fig7, ax7 = plt.subplots(figsize=(10, 8))
disp.plot(cmap='Blues', ax=ax7)
plt.title('Confusion Matrix')
st.pyplot(fig7)


# 8. Topic Composition per Review
st.header("8. Topic Composition per Review")
fig8, ax8 = plt.subplots(figsize=(12, 6))
df_review_document_topic.iloc[:, :5].plot(kind='bar', ax=ax8)
ax8.set_xlabel('Review')
ax8.set_ylabel('Topic Composition')
ax8.set_title('Topic Composition per Review')
st.pyplot(fig8)

# 9. Sentiment Score by Dominant Topic
st.header("9. Sentiment Score by Dominant Topic")
fig9, ax9 = plt.subplots(figsize=(10, 6))
df_review.groupby('dominant_topic')['dominant_topic'].mean().plot(kind='bar', ax=ax9)
ax9.set_xlabel('Dominant Topic')
ax9.set_ylabel('Average Sentiment Score')
ax9.set_title('Sentiment Score by Dominant Topic')
st.pyplot(fig9)

# 10. Review Length Distribution
st.header("10. Review Length Distribution")
fig10, ax10 = plt.subplots(figsize=(8, 6))
df_review['processed_review'].str.len().hist(bins=20, ax=ax10)
ax10.set_xlabel('Review Length')
ax10.set_ylabel('Count')
ax10.set_title('Distribution of Review Lengths')
st.pyplot(fig10)

# 11. Top Features Treemap
st.header("11. Top Features Treemap")

# Create a dictionary to map topic names to top features
topic_feature_map = dict(zip(topic_names, most_used_features['Top Features']))

# Function to display top features on hover
def hover_text(label):
    return f"{label}<br>Top Features: {topic_feature_map[label]}"

fig11 = go.Figure(go.Treemap(
    labels=most_used_features['Topic'],
    parents=[''] * len(most_used_features),
    values=[len(topic.split(', ')) for topic in most_used_features['Top Features']],
    textinfo="label+value",
    hovertemplate="%{label}<br>Top Features: %{customdata}",
    customdata=[topic_feature_map[label] for label in most_used_features['Topic']],
    marker_colorscale='Blues'
))
fig11.update_layout(
    title='Top Features Treemap',
    margin=dict(t=50, l=25, r=25, b=25)
)
st.plotly_chart(fig11)


# Load the pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def generate_example_sentence(feature):
    # Prepare the input text for the model
    prompt = f"The {feature} feature would be really helpful for our users because"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate the example sentence
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, num_beams=1)[0]
    example_sentence = tokenizer.decode(output, skip_special_tokens=True)

    return example_sentence

# 12. Top Features and Example Sentences
st.title("Top Features and Example Sentences")

# Create a dropdown menu to select the topic
topic_options = [f"Topic {i+1}" for i in range(len(topic_names))]
selected_topic = st.selectbox("Select a topic:", topic_options)

# Generate example sentences for the selected topic
topic_index = int(selected_topic.split(" ")[1]) - 1
top_features = most_used_features.loc[topic_index, 'Top Features'].split(', ')

st.subheader(selected_topic)
for feature in top_features:
    example_sentence = generate_example_sentence(feature)
    st.write(f"- {feature}: {example_sentence}")

