# EMP
Enhanced multi-modal prediction for fashion sales using Fourier Mapping and ERP-based contrastive learning

This repository implements the full two-stage EMP pipeline used in our study.  
EMP predicts future sales for unseen items(zero-shot) by combining:

- Stage 1: ERP-aware multimodal contrastive learning 
- Stage 2: Transformer forecasting with Fourier Mapping and contextual signals  

The implementation supports the Visuelle 2.0 benchmark dataset.

---

# Overview of the 2-Stage System

Stage 1: ERP-Aware Contrastive Learning
    test_pred_dist_argsort.npy  
Stage 2: EMP Zero-shot Forecasting  
    Uses Stage 1 outputs + Fourier mapping + Transformer

Stage 1 learns the embedding space.  
Stage 2 performs forecasting using retrieved neighbor sales.

---

# Repository Structure

stage_1/    – ERP-Aware contrastive learning  
stage_2/    – EMP forecasting model (Fourier + Transformer)  
requirements.txt  
README.md  

---

# 🚀 Stage 1 — ERP-Aware Contrastive Learning

### Objective  
Learn multimodal item embeddings where ERP-distance–close items become neighbors.

### Outputs  
- test_pred_dist_argsort.npy  

### Run
python main.py 

---

# 🔮 Stage 2 — EMP Zero-shot Forecasting

### Objective  
Predict 12-week sales for items without sales history using:

### Run
python main.py --use_gtrends 0 --use_pop 0 --use_kitem 1 --fourier_B 10000

---

# 📦 Installation

pip install -r requirements.txt

---

# 📝 Citation
If you use this code, please cite the EMP study.

---

# 👤 Contact
Sanguk Park – Yonsei University
GitHub: https://github.com/sanguk-ys
E-mail: lostmywatch@yonsei.ac.kr
