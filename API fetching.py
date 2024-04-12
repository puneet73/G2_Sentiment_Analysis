import requests
import csv
from collections import Counter
import matplotlib.pyplot as plt

def fetch_reviews(token):
    url = "https://data.g2.com/api/v1/survey-responses"
    headers = {
        "Authorization": f"Token token={token}",
        "Content-Type": "application/vnd.api+json"
    }
    all_reviews = []
    page = 1
    while True:
        params = {"page[size]": 100, "page[number]": page}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()['data']
            if not data:
                break
            all_reviews.extend(data)
            page += 1
        else:
            print("Error fetching reviews:", response.text)
            break
    return all_reviews

def save_reviews_to_csv(reviews):
    with open('E:\\G2 project\\reviews.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'product_name', 'title', 'comment_answers']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            writer.writerow({
                'id': review['id'],
                'product_name': review['attributes']['product_name'],
                'title': review['attributes']['title'],
                'comment_answers': review['attributes']['comment_answers']
            })

def filter_reviews_by_product_name(reviews, product_name):
    filtered_reviews = []
    for review in reviews:
        if 'attributes' in review and 'product_name' in review['attributes'] and review['attributes']['product_name'].strip().lower() == product_name.strip().lower():
            filtered_reviews.append(review)
    return filtered_reviews

def extract_features(reviews):
    feature_counts = Counter()
    for review in reviews:
        if 'attributes' in review and 'comment_answers' in review['attributes']:
            answers = review['attributes']['comment_answers']
            for key in answers:
                feature_counts[answers[key]['value']] += 1
    return feature_counts

def rank_features(feature_counts):
    return feature_counts.most_common()

def visualize_features(ranked_features):
    features, counts = zip(*ranked_features[:10])  # Top 10 features
    
    plt.figure(figsize=(10, 6))  # Adjust figure size
    
    plt.barh(range(len(features)), counts, tick_label=features, height=0.5)  # Adjust height of bars
    plt.xlabel('Frequency')
    plt.ylabel('Feature')
    plt.title('Top 10 Feature Requests for G2 Marketing Solutions')
    plt.gca().invert_yaxis()  # Invert y-axis to have highest count at the top
    
    # Rotate feature labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()  # Adjust layout to prevent clipping
    
    plt.show()

def main():
    # Replace 'token' with your actual token
    token = "5d5839189bd0fecbbe25669cd0d89ec0e03c370a11f21d298f005f199ae2008b"

    # Fetch reviews for all products
    reviews = fetch_reviews(token)
    if reviews:
        # Save reviews to CSV file
        save_reviews_to_csv(reviews)
        print("Reviews saved to 'reviews.csv'")
        
        # Filter reviews for a specific product name
        product_name = "G2 Marketing Solutions"
        filtered_reviews = filter_reviews_by_product_name(reviews, product_name)
        if filtered_reviews:
            print("Number of reviews for", product_name, ":", len(filtered_reviews))
            feature_counts = extract_features(filtered_reviews)
            ranked_features = rank_features(feature_counts)
            visualize_features(ranked_features)
        else:
            print("No reviews found for", product_name)

if __name__ == "__main__":
    main()
