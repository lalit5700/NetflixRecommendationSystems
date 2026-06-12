def generate_and_analyze_user_recs(predictions, target_user_id, k=5):
    user_preds = [p for p in predictions if p.uid == target_user_id]
    user_preds.sort(key=lambda x: x.est, reverse=True)
    
    print(f"\n=========================================\n📋 INFERENCE REPORT FOR USER: {target_user_id}\n=========================================")
    for p in user_preds[:k]:
        print(f" 🎬 MovieID: {p.iid:<5} | Predicted: {p.est:.2f} | Actual: {p.r_ui}")
        
    success = [p for p in user_preds if p.est >= 3.8 and p.r_ui >= 4.0]
    if success:
        print(f"\n✅ SUCCESS CASE: Movie {success[0].iid} | Predicted: {success[0].est:.2f} | Actual: {success[0].r_ui}")
