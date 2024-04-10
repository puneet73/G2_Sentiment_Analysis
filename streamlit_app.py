import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import random
import streamlit as st
import plotly.graph_objects as go

# Load the CSV file
@st.cache
def load_data():
    return pd.read_csv('E:\\G2 project\\sentences_by_category.csv')

# Preprocess the data
def preprocess_data(df):
    for category in ['love', 'hate', 'recommendations', 'benefits']:
        df[category] = df[category].astype(str)
        df[category] = df[category].str.lower()
        df[category] = df[category].str.replace('[^\w\s]', '', regex=True)
        df[category] = df[category].str.replace('\s+', ' ', regex=True)
    return df

# Sentiment analysis
def get_sentiment_score(text, stop_words):
    analyzer = SentimentIntensityAnalyzer()
    doc = nlp(text)
    noun_verbs = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB'] and token.text not in stop_words]
    if noun_verbs:
        text_to_analyze = ' '.join(noun_verbs)
        return analyzer.polarity_scores(text_to_analyze)['compound']
    else:
        return 0

# Topic modeling
def topic_modeling(text_data):
    vectorizer = CountVectorizer(stop_words=all_stop_words)
    X = vectorizer.fit_transform(text_data)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)
    feature_names = vectorizer.get_feature_names_out()

    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words_indices = topic.argsort()[:-6:-1]  # Top 5 words
        top_words = [feature_names[idx] for idx in top_words_indices]
        topics[f"Topic {topic_idx+1}"] = top_words

    return topics

# Visualization functions
def visualize_word_cloud(text_data, category):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=all_stop_words).generate(' '.join(text_data))
    st.image(wordcloud.to_array(), caption=f'Most Frequent Words in {category}')

def visualize_sentiment_distribution(sentiment_scores, category):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(sentiment_scores, ax=ax)
    ax.set_title(f'Sentiment Distribution in {category}')
    st.pyplot(fig)

# Generate sentences based on sentiment score and analysis results
def generate_sentence(df, category):
    sentiment_score = df[f'{category}_sentiment_score'].mean()  # You may use actual sentiment score instead of mean
    
    user_needs = {
        'love': ['ease of use', 'performance'],
        'hate': ['lack of customization options', 'poor customer support'],
        'recommendations': ['better integration with other software', 'comprehensive reporting features'],
        'benefits': ['time-saving capabilities', 'improved productivity']
    }

    user_need = random.choice(user_needs[category])
    
    if sentiment_score >= 0.5:
        positive_sentences = [
            f"I absolutely love the {user_need}. It's fantastic!",
            f"The {user_need} are amazing, couldn't be happier!",
            f"I'm really impressed by the {user_need}. It's been a game-changer for me."
        ]
        return random.choice(positive_sentences)
    elif sentiment_score <= -0.5:
        negative_sentences = [
            f"I really dislike the {user_need}. It's frustrating.",
            f"The {user_need} are a major pain point for me.",
            f"The {user_need} are terrible, really disappointed."
        ]
        return random.choice(negative_sentences)
    else:
        neutral_sentences = [
            f"The {user_need} are okay, but there's room for improvement.",
            f"The {user_need} are decent, but not exceptional.",
            f"I'm indifferent towards the {user_need}."
        ]
        return random.choice(neutral_sentences)

# Main function
def main():
    st.title('G2 Project Analysis')

    df = load_data()
    df = preprocess_data(df)

    # Load the spaCy English language model
    nlp = spacy.load("en_core_web_sm")

    # Define a list of additional stop words to remove
    extra_stop_words = ['to', 'that', 'your', 'and', 'for', 'is', 'on', 'the', 'can', 'are', 'we', 'our', 'of', 'with', 'g2']
    all_stop_words = ENGLISH_STOP_WORDS.union(extra_stop_words)
    all_stop_words = list(all_stop_words)

    categories = ['love', 'hate', 'recommendations', 'benefits']
    for category in categories:
        st.header(category.capitalize())

        # Sentiment analysis
        df[f'{category}_sentiment_score'] = df[category].apply(lambda x: get_sentiment_score(x, all_stop_words))

        # Topic modeling
        topics = topic_modeling(df[category])
        st.write('Topic modeling results:')
        for topic, words in topics.items():
            st.write(f"{topic}: {' '.join(words)}")

        # Visualizations
        visualize_word_cloud(df[category], category)
        visualize_sentiment_distribution(df[f'{category}_sentiment_score'], category)

        # Generate and display sentence
        sentence = generate_sentence(df, category)
        st.write('Generated Sentence:', sentence)

if __name__ == "__main__":
    main()
