# Deep Learning-Based Signature Forgery Detection: A Learnable Distance Approach with YOLOv10, ResNet-34, and Triplet Siamese Similarity Network

## Introduction

Handwritten signatures remain a cornerstone of identity verification in critical sectors like banking, law, and finance. However, traditional verification systems struggle with two fundamental challenges: the natural variability in a person's signature and the increasing sophistication of skilled forgeries. Furthermore, deploying a robust system often requires a large number of signature samples per user, which is impractical in many real-world scenarios.

To address these limitations, this project presents a **Few-Shot Adaptive Metric Learning framework** for offline signature forgery detection. Instead of relying on a single, fixed similarity metric for all users, our approach leverages **meta-learning** to learn a unique, writer-specific distance metric from just a handful of genuine signature samples. This allows the system to adapt to the unique characteristics of any individual's signature, providing a more personalized and accurate verification.

Our framework integrates three key components:
- **YOLOv10**: For high-efficiency signature localization from documents.
- **Pre-trained ResNet-34**: As a robust feature extractor to generate powerful signature embeddings.
- **Adaptive Metric Learner**: A meta-trained network that generates a unique Mahalanobis distance metric for each user, trained using an **Online Hard Triplet Mining** strategy.

Experimental results demonstrate the state-of-the-art performance of our approach. Using a rigorous **5-fold cross-validation** on the **CEDAR dataset**, the model achieves a near-perfect mean accuracy of **99.94% ± 0.13%**. More importantly, to prove its true generalization capability, the model, trained exclusively on CEDAR, achieves an impressive **82.04% accuracy** on the completely unseen **BHSig-260 (Bengali & Hindi)** dataset, showcasing its robustness across different languages and writing styles.

## Key Features
- **Few-Shot Learning**: Accurately verifies signatures using only a small number (`k-shot`) of genuine samples for new users.
- **Adaptive Metric Learning**: Utilizes a meta-learning approach to generate a writer-specific Mahalanobis distance metric, providing personalized verification.
- **Advanced Training Strategy**: Employs a pre-trained feature extractor and an Online Hard Triplet Mining strategy for robust and efficient training.
- **State-of-the-Art Performance**: Achieves **99.94%** mean accuracy on CEDAR with 5-fold cross-validation.
- **Proven Generalization**: Demonstrates strong cross-dataset performance, achieving **82.04%** accuracy on the BHSig-260 dataset without any retraining.
- **End-to-End Pipeline**: Includes YOLOv10 for automated signature localization from documents.

---

## Project Structure
```plaintext
├── configs/
│   ├── __init__.py
│   └── config_tSSN.yaml                # Configuration for baseline models
│
├── dataloader/
│   ├── __init__.py
│   ├── meta_dataloader.py              # Dataloader for meta-learning episodes
│
├── losses/
│   ├── __init__.py
│   └── triplet_loss.py                 # Triplet loss implementations, including Mahalanobis distance
│
├── models/
│   ├── __init__.py
│   ├── feature_extractor.py            # ResNet-34 backbone
│   └── meta_learner.py                 # The core MetricGenerator network
│
├── notebooks/
│   ├── pretraining.ipynb               # Notebook for pre-training the feature extractor
│   ├── meta_training_kfold.ipynb       # Main notebook for K-fold CV meta-learning on CEDAR
│   ├── cross_dataset_evaluation.ipynb  # Notebook for evaluating on BHSig-260
│   └── yolov10_bcsd_trainning.ipynb    # Notebook for training the YOLOv10 localizer
│
├── scripts/
│   ├── __init__.py
│   ├── prepare_kfold_splits.py         # Script to generate 5-fold data splits for CEDAR
│   └── restructure_bhsig.py            # Script to restructure the BHSig-260 dataset
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py
│   └── model_evaluation.py             # Evaluation functions for meta-learning
│
├── README.md
├── requirements.txt
├── setup.py                            # Installation setup (optional for packaging)
├── signature_verification.egg-info/    # Build metadata (auto-generated)
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   ├── requires.txt
│   └── top_level.txt
```

---

## Installation
1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/trongjhuongwr/Deep-Learning-Based-Signature-Forgery-Detection-for-Personal-Identity-Authentication-Update.git](https://github.com/trongjhuongwr/Deep-Learning-Based-Signature-Forgery-Detection-for-Personal-Identity-Authentication-Update.git)
    cd Deep-Learning-Based-Signature-Forgery-Detection-for-Personal-Identity-Authentication-Update
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## **Kaggle API Token Setup**

To access and download datasets directly from Kaggle within this project, follow these steps to set up your Kaggle API token:

1. Go to your [Kaggle account settings](https://www.kaggle.com/account).
2. Scroll down to the **API** section.
3. Click on **"Create New API Token"** – a file named `kaggle.json` will be downloaded.
4. Place the `kaggle.json` file in the root directory of this project **or** in your system's default path:  
   - Linux/macOS: `~/.kaggle/kaggle.json`  
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
5. Make sure the file has appropriate permissions:  
   ```bash
   chmod 600 ~/.kaggle/kaggle.json

---

---

## Usage & Replication of Results

To replicate the state-of-the-art results of our **Few-Shot Adaptive Metric Learning** framework, follow these steps. It is highly recommended to use a GPU-accelerated environment like Kaggle or Google Colab.

**Step 0: Data Preparation & Signature Localization**
- Download the **CEDAR** and **BHSig-260** datasets.
- The `notebooks/yolov10_bcsd_trainning.ipynb` notebook can be used to train a YOLOv10 model for automated signature localization from raw documents. The experiments below assume pre-cropped signature images are available.

**Step 1: Pre-train the Feature Extractor**
- **Objective**: To provide the model with a strong initial understanding of signature features before meta-learning.
- **Action**: Run the `notebooks/pretraining.ipynb` notebook.
- **Outcome**: This trains the ResNet-34 backbone on a standard triplet loss task and saves the weights (`pretrained_feature_extractor.pth`), which will be used as a starting point in the next step.

**Step 2: Meta-Train and Evaluate on CEDAR (K-Fold Cross-Validation)**
- **Objective**: To train the adaptive metric learner and rigorously validate its performance and reliability on the CEDAR dataset.
- **Action**:
    1.  Run the `scripts/prepare_kfold_splits.py` script to generate the 5 JSON files required for cross-validation.
    2.  Run the `notebooks/meta_training_kfold.ipynb` notebook.
- **Outcome**: This notebook performs the full 5-fold cross-validation, training and evaluating the model five times on different subsets of the data. The final output confirms the near-perfect result of **99.94% ± 0.13% mean accuracy**. The best model from this process should be saved for the next step.

**Step 3: Evaluate Cross-Dataset Generalization on BHSig-260**
- **Objective**: To perform the ultimate test of the model's generalization capabilities by evaluating it on a completely unseen dataset with different languages and writing styles.
- **Action**:
    1.  If using the mixed `cedarbhsig-260` dataset, first run `scripts/restructure_bhsig.py` to create a clean, structured split file.
    2.  Run the `notebooks/cross_dataset_evaluation.ipynb` notebook.
- **Outcome**: This notebook loads the model trained **only on CEDAR** and evaluates its few-shot performance on the BHSig-260 dataset. The result of **82.04% accuracy** demonstrates the model's powerful and robust generalization capabilities.

---

## Results

### Key Findings:
Our Few-Shot Adaptive Metric Learning approach yielded state-of-the-art results, successfully addressing the core limitations identified in the initial review.

1.  **Near-Perfect Intra-Dataset Performance & Reliability (CEDAR):**
    -   Using a rigorous 5-fold cross-validation, the model achieved a mean accuracy of **99.94% ± 0.13%**.
    -   This near-perfect score, validated across multiple data splits, demonstrates the model's exceptional effectiveness and reliability on a standard benchmark dataset.

2.  **Strong Cross-Dataset Generalization (BHSig-260):**
    -   When the model trained exclusively on CEDAR was tested on the unseen BHSig-260 dataset, it achieved **82.04% accuracy**.
    -   This is a critical finding, proving that the model did not simply overfit to the CEDAR dataset. It learned a true, generalizable ability to adapt to new users, even those with entirely different writing styles and languages (English vs. Bengali/Hindi).

3.  **Methodological Superiority:**
    -   The combination of a pre-trained feature extractor, a meta-learned adaptive metric (Mahalanobis distance), and an Online Hard Triplet Mining strategy proved to be highly effective.
    -   Visualizations of the embedding space show a clear and consistent separation between genuine and forged signatures, even for unseen users.

---

## Datasets & Notebooks

### Datasets Used:
-   **BCSD**: [Used for training the YOLOv10 model](https://www.kaggle.com/datasets/saifkhichi96/bank-checks-signatures-segmentation-dataset)
-   **CEDAR**: [Used for meta-training and K-fold validation](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)
-   **BHSig-260**: [Used for cross-dataset generalization evaluation](https://www.kaggle.com/datasets/nth2165/bhsig260-hindi-bengali) (Recommended Version) or [the mixed version](https://www.kaggle.com/datasets/ankita22053139/cedarbhsig-260) (requires restructuring).

### Replication Notebooks:
The complete and reproducible workflow is documented within the `notebooks/` directory of this repository. These notebooks are designed to be run sequentially to achieve the final results.

-   `pretraining.ipynb`: Pre-trains the feature extractor.
-   `meta_training_kfold.ipynb`: The main notebook for training and achieving the 99.94% result on CEDAR.
-   `cross_dataset_evaluation.ipynb`: The notebook for achieving the 82.04% generalization result on BHSig-260.

---

## Contributions

-   **Designed and implemented a novel Few-Shot Adaptive Metric Learning framework** for signature verification, moving beyond generic similarity metrics.
-   **Successfully integrated a meta-learning paradigm** (`MetricGenerator`) to create personalized, writer-specific Mahalanobis distance metrics.
-   **Implemented an advanced training pipeline**, including a pre-trained feature extractor and an Online Hard Triplet Mining strategy to handle challenging cases.
-   **Conducted rigorous, state-of-the-art validation** using 5-fold cross-validation, achieving near-perfect **99.94% accuracy** on the CEDAR dataset.
-   **Provided a definitive answer to the question of generalization**, performing cross-dataset evaluation and achieving an impressive **82.04% accuracy** on the BHSig-260 dataset without any re-training.
-   **Structured the entire project for full reproducibility**, with modular code, data preparation scripts, and a clear sequence of experimental notebooks.
-   **Addressed every major point of criticism from the initial peer review**, transforming weaknesses into the core strengths of the research.

## Future Work
- **Domain Adaptation**: Explore unsupervised domain adaptation techniques to close the performance gap between different datasets (e.g., from 99% on CEDAR to 82% on BHSig-260).
- **Explainable AI (XAI)**: Integrate techniques like Grad-CAM to visualize which parts of a signature the model focuses on, increasing transparency and trust.
- **Lighter Architectures**: Experiment with more lightweight backbones (e.g., MobileNetV3, EfficientNet) for deployment on resource-constrained devices.

---

## **Acknowledgments**
Special thanks to the contributors and open-source community for providing tools and resources.

--- 


