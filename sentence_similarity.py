from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_sentence_similarity(sentence1, sentence2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

def main():
    sentence1 = input("请输入第一个句子: ")
    sentence2 = input("请输入第二个句子: ")
    
    similarity = calculate_sentence_similarity(sentence1, sentence2)
    
    print(f"\n句子1: {sentence1}")
    print(f"句子2: {sentence2}")
    print(f"\n两个句子的相似度: {similarity:.4f}")

if __name__ == "__main__":
    main()