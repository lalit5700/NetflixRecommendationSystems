import numpy as np
from collections import defaultdict
from surprise import accuracy

def calculate_map_at_k(predictions, k=10, relevance_threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
        
    user_aps = []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        num_relevant = 0
        score_sum = 0.0
        for i, (est, true_r) in enumerate(top_k):
            if true_r >= relevance_threshold:
                num_relevant += 1
                score_sum += num_relevant / (i + 1)
        actual_relevant = sum(1 for _, true_r in user_ratings if true_r >= relevance_threshold)
        if actual_relevant > 0:
            user_aps.append(score_sum / min(k, actual_relevant))
    return np.mean(user_aps) if user_aps else 0.0

def evaluate_recommenders(model_svd, model_base, testset):
    preds_svd = model_svd.test(testset)
    preds_base = model_base.test(testset)
    return {
        "SVD": {"RMSE": accuracy.rmse(preds_svd, verbose=False), "MAP@10": calculate_map_at_k(preds_svd, k=10)},
        "BaselineOnly": {"RMSE": accuracy.rmse(preds_base, verbose=False), "MAP@10": calculate_map_at_k(preds_base, k=10)}
    }, preds_svd
