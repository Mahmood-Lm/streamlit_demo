import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")


def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


def knn_recommendations(user_id, n_neighbors, sim_threshold):
    """
    KNN-based course recommendation using user-item collaborative filtering
    """
    ratings_df = load_ratings()
    
    # Create user-item matrix
    user_item_matrix = ratings_df.pivot_table(
        index='user', 
        columns='item', 
        values='rating', 
        fill_value=0
    )
    
    # Get the current user's ratings
    if user_id not in user_item_matrix.index:
        return {}
    
    user_ratings = user_item_matrix.loc[user_id].values.reshape(1, -1)
    
    # Remove the current user from the matrix for finding neighbors
    other_users_matrix = user_item_matrix.drop(user_id)
    
    if len(other_users_matrix) == 0:
        return {}
    
    # Use KNN to find similar users
    knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(other_users_matrix)), metric='cosine')
    knn.fit(other_users_matrix.values)
    
    # Find k nearest neighbors
    distances, neighbor_indices = knn.kneighbors(user_ratings)
    
    # Get neighbor user IDs
    neighbor_users = other_users_matrix.iloc[neighbor_indices[0]]
    
    # Calculate weighted average ratings for items not rated by current user
    user_rated_items = set(user_item_matrix.columns[user_item_matrix.loc[user_id] > 0])
    all_items = set(user_item_matrix.columns)
    unrated_items = all_items - user_rated_items
    
    recommendations = {}
    
    for item in unrated_items:
        weighted_sum = 0
        weight_sum = 0
        
        for i, neighbor_user_id in enumerate(neighbor_users.index):
            neighbor_rating = user_item_matrix.loc[neighbor_user_id, item]
            if neighbor_rating > 0:  # Only consider items that neighbor has rated
                # Convert distance to similarity (1 - cosine_distance)
                similarity = 1 - distances[0][i]
                weighted_sum += similarity * neighbor_rating
                weight_sum += similarity
        
        if weight_sum > 0:
            predicted_rating = weighted_sum / weight_sum
            # Normalize to similarity score (0-1)
            similarity_score = predicted_rating / 5.0  # Assuming ratings are 1-5
            
            if similarity_score >= sim_threshold:
                recommendations[item] = similarity_score
    
    # Sort recommendations by score
    recommendations = {k: v for k, v in sorted(recommendations.items(), key=lambda item: item[1], reverse=True)}
    return recommendations


# Model training
def train(model_name, params):
    # TODO: Add model training code here
    pass


# Prediction
def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == models[0]:
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)
        # KNN model
        elif model_name == models[4]:  # KNN is at index 4
            n_neighbors = params.get('n_neighbors', 5)
            knn_sim_threshold = params.get('knn_sim_threshold', 60) / 100.0
            res = knn_recommendations(user_id, n_neighbors, knn_sim_threshold)
            for key, score in res.items():
                users.append(user_id)
                courses.append(key)
                scores.append(score)
        # TODO: Add other prediction model code here

    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df
