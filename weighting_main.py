import numpy as np
import pandas as pd
from main import *
from main import initialize

def sim_query_weight(input_query: [str], feature_vector_1, feature_vector_2=None, feature_function_mode=0):
    first_weight = 0.5
    second_weight = 1.0 - first_weight
    if feature_function_mode == 1:
        similarity = ((cosine_similarity(feature_vector_1.loc[input_query], feature_vector_1) + 1) * 0.5) * first_weight + \
                     (1 / (1 + euclidean_distances(feature_vector_2.loc[input_query], feature_vector_2,
                                                   squared=True))) * second_weight
    elif feature_function_mode == 2 or feature_function_mode == 4:
        similarity = ((cosine_similarity(feature_vector_1.loc[input_query], feature_vector_1) * first_weight +
                       cosine_similarity(feature_vector_2.loc[input_query], feature_vector_2) * second_weight) + 1) * 0.5
    elif feature_function_mode == 3:
        similarity = (cosine_similarity(feature_vector_1.loc[input_query], feature_vector_1) + 1)
    else:
        raise ValueError("Feature function mode is invalid!")
    return similarity

def main_weight(tfidf_df, bert_df, genres, video_features_resnet_max, video_features_resnet_mean,
                               mfcc_bow, spectral, spectral_contrast, spotify_data):
    all_all_results = list()
    for i in range(4):
        all_results = pd.DataFrame(index=genres.index, columns=(["results"]))
        all_results.index.name = "id"
        count_df = pd.DataFrame(index=genres.index, columns=(["count"])).fillna(0)
        all_results.index.name = "id"
        all_all_results.append(all_results)

    splitted_dataset = np.array_split(genres, 1)
    iter_counter = 0
    for dataset in splitted_dataset:
        query_songs = dataset.index
        results_lyrics = sim_query_weight(query_songs, tfidf_df, bert_df, 1)
        results_video = sim_query_weight(query_songs, video_features_resnet_max, video_features_resnet_mean, 2)
        results_audio_bow = sim_query_weight(query_songs, mfcc_bow, None, 3)
        results_audio_spectral = sim_query_weight(query_songs, spectral, spectral_contrast, 4)

        results = [results_lyrics, results_video, results_audio_bow, results_audio_spectral]
        for idx, result in enumerate(results):
            for i in range(len(query_songs)):
                result[i, i+iter_counter] = 0
            res = pd.DataFrame(result.T, index=genres.index, columns=query_songs)
            res.index.name = "id"

            for col in res.columns:
                res_largest = res[col].nlargest(100)
                all_all_results[idx].loc[col]["results"] = res_largest.index.tolist()


        iter_counter += len(dataset)

    for idx, all_results in enumerate(all_all_results):

        index_loop = 0
        delta_mean_array = np.zeros(len(genres))
        precision_sum = 0
        mrr_sum = 0
        ndcg_sum_10 = 0
        ndcg_sum_100 = 0

        precision_arr_sum = 0
        recall_arr_sum = 0

        all_results.dropna(inplace=True)

        for index, result in all_results.iterrows():
            if not result.empty or not pd.isna(result):
                delta_mean_array[index_loop] = percent_delta_mean_2(spotify_data, index, result["results"])

                precision_sum += precision_s_2(genres, index, result["results"])
                mrr_sum += mrr_s_2(genres, index, result["results"])
                ndcg_sum_10 += nDCG_ms_2(result["results"], genres, index, 10)
                ndcg_sum_100 += nDCG_ms_2(result["results"], genres, index, 100)
                index_loop += 1



        print(f"Performan Metrics for {idx} {len(genres)} songs")
        print(f"Precision: {precision_sum / len(genres)}\n")
        print(f"MRR: {mrr_sum / len(genres)}\n")
        print(f"NDCG10: {ndcg_sum_10 / len(genres)}\n")
        print(f"NDCG100: {ndcg_sum_100 / len(genres)}\n")
        print(f"Median Delta Mean: {np.median(delta_mean_array)}")

if __name__ == "__main__":
    tfidf_df, bert_df, genres, video_features_resnet_max, video_features_resnet_mean, mfcc_bow, spectral, spectral_contrast, spotify_data = initialize()
    main_weight(tfidf_df, bert_df, genres, video_features_resnet_max, video_features_resnet_mean, mfcc_bow, spectral, spectral_contrast, spotify_data)