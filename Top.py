from top2vec import Top2Vec
import pandas as pd
df = pd.read_csv("Heart_disease.csv")
titles = df['Title']
titles = titles.tolist()
model = Top2Vec(titles, speed="fast-learn")
topic_num = model.get_num_topics()
print(topic_num)
topic_words, word_scores, topic_nums = model.get_topics(topic_num)
print(topic_words)
print(word_scores)
print(topic_nums)
model.save("trained")
print("hello")


