# Pri-MCCD Technical Validation

This repository provides baseline model evaluation and feature analysis code for the **Pri-MCCD** dataset, a multimodal classroom climate dataset collected from real-world primary school lessons. The code supports pilot studies, ablation experiments, and acoustic feature analysis using standard multimodal pipelines.

---

## 📂 Repository Structure

| File / Folder         | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `main.py`             | Main training script for baseline and ablation experiments                  |
| `IS09_analysis.py`    | Statistical analysis of key IS09 acoustic features                          |
| `models/`             | Encoders, fusion modules, and classification heads                          |


---

## ✅ Baseline Model Validation 

We provide a reproducible baseline using standard GRU encoders and transformer-based fusion. This is intended to establish initial accuracy and robustness on Pri-MCCD's multimodal emotion/climate recognition task.

---

### 🔧 Encoder Configuration and Ablation

You can adjust `--visual_encoder_type` and `--acoustic_encoder_type` to test different encoder combinations, such as:

- `gru`
- `lstm`
- `tcn`



You can also optionally disable the fusion module (e.g., transformer-based fusion)  by omitting `--use_transformer_fusion`, to perform ablation studies on modality interaction mechanisms.

### ✅ Example usage:

```bash
python main.py \
  --visual_encoder_type gru \
  --acoustic_encoder_type gru \
  --save_name gru \
  --use_transformer_fusion
````

---

## 🔍 IS09 Acoustic Feature Analysis

The script `IS09_analysis.py` performs statistical inspection of the acoustic feature distributions across labeled classroom climate conditions. It includes:

* **Mean Pitch Distribution**
* **Energy Standard Deviation**
* **F0 Standard Deviation**


These indicators reflect underlying affective cues and help interpret the contribution of acoustic modality in multimodal fusion.

### ✅ Run the script:

```bash
python IS09_analysis.py
```

The analysis outputs visualizations and aggregated metrics.

---

## 📊 Experimental Goals

* Evaluate basic multimodal architectures on Pri-MCCD
* Assess the role of audio-visual encoders and fusion mechanisms
* Interpret key acoustic indicators aligned with classroom climate theory
* Provide reproducible experimental baselines for future research

---

## 📫 Contact

For dataset access or questions, please contact:
clbian@ouc.edu.cn

---





