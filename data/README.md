# **Recommendation Systems for Personalized Content Discovery**
**Dataset:** Netflix Prize Dataset (Subset)  
**Platform Architecture:** Machine Learning Pipeline for Streaming Discovery  

## **Project Overview**
This project focuses on building a scalable, memory-efficient recommendation engine capable of predicting user preferences and ranking personalized content. We address the core challenge of extreme data sparsity characteristic of large-scale streaming platforms.

### **Pipeline Architecture**
1. **Exploratory Data Analysis (EDA):** Quantifying sparsity and analyzing global rating trends.
2. **Data Engineering:** Filtering and downsampling to maintain core latent structures within Colab memory limits.
3. **Model Development:** Implementing and optimization of Latent Factor Modeling (SVD) and Biased Baselines (BaselineOnly).
4. **Evaluation:** Assessing models using absolute error metrics ($RMSE$) and ranking performance ($MAP@10$).

