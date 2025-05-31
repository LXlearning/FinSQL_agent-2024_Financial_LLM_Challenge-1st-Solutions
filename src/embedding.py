import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine


def find_similar_docs(doc_embeddings_list, text_embeddings_list, score_threshold=0.5, topk=10):
    doc_embeddings = np.array([x[1] for x in doc_embeddings_list])
    doc_list = [x[0] for x in doc_embeddings_list]
    text_list = [x[0] for x in text_embeddings_list]
    text_embeddings = np.array([x[1] for x in text_embeddings_list])

    predict_scores = doc_embeddings @ text_embeddings.T

    df = pd.DataFrame()
    num_docs = len(doc_embeddings_list)
    # 每个doc设置一个最大召回量，避免都被一个doc全部召回了
    num_recall_per_doc = round(topk // num_docs * 1.2)
    for i in range(num_docs):
        doc_scores = predict_scores[i]
        valid_indices = np.where((doc_scores >= score_threshold))[0]

        doc_similar_texts = [text_list[x] for x in valid_indices]
        doc_similar_scores = doc_scores[valid_indices]
        df_tmp = pd.DataFrame({'doc': doc_similar_texts, 'score': doc_similar_scores})
        df_tmp = df_tmp.sort_values(by='score', ascending=False).head(num_recall_per_doc)
        df_tmp['原文本'] = doc_list[i]
        df = pd.concat([df_tmp, df])
    df = df.sort_values(by='score', ascending=False)
    df = df.drop_duplicates(subset='doc').head(topk)

    return df


def cal_score(embeddings_1, embeddings_2):
    similarity = 1 - cosine(embeddings_1, embeddings_2)
    return round(similarity, 3)


def get_top_k(score_list, score, k):
    filtered_numbers_with_index = [(index, num) for index, num in enumerate(score_list) if num >= score]
    top_two_with_index = sorted(filtered_numbers_with_index, key=lambda x: x[1], reverse=True)[:k]
    return top_two_with_index
