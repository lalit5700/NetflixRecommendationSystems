import pandas as pd
from src.data_pipeline import download_and_parse_data
from src.model_pipeline import prepare_surprise_data, train_models
from src.evaluation import evaluate_recommenders
from src.inference import generate_and_analyze_user_recs
from surprise.model_selection import train_test_split

def main():
    print("🚀 Starting Production Recommendation Execution Pipeline 🚀\n")
    df = download_and_parse_data()
    surprise_dataset = prepare_surprise_data(df)
    trainset, testset = train_test_split(surprise_dataset, test_size=0.2, random_state=42)
    
    model_svd, model_base = train_models(trainset)
    metrics, preds_svd = evaluate_recommenders(model_svd, model_base, testset)
    
    print("\n=========================================\n🏆 FINAL PERFORMANCE RESULTS 🏆\n=========================================")
    print(pd.DataFrame(metrics).T.to_string())
    
    generate_and_analyze_user_recs(preds_svd, preds_svd[0].uid, k=5)

if __name__ == "__main__":
    main()
