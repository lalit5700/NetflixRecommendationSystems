from surprise import Dataset, Reader, SVD, BaselineOnly

def prepare_surprise_data(df):
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(df[['UserID', 'MovieID', 'Rating']], reader)

def train_models(trainset):
    print("🏋️‍♂️ Training Model A: SVD Matrix Factorization...")
    model_svd = SVD(n_factors=50, random_state=42)
    model_svd.fit(trainset)
    
    print("🏋️‍♂️ Training Model B: BaselineOnly ALS...")
    model_base = BaselineOnly(bsl_options={'method': 'als'})
    model_base.fit(trainset)
    
    return model_svd, model_base
