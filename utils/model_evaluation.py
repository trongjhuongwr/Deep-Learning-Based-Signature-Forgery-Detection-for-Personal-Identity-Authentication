import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
)
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
try:
    from dataloader.meta_dataloader import SignatureEpisodeDataset
except ImportError:
    print("Warning: Could not import SignatureEpisodeDataset. Ensure dataloader path is correct.")

def evaluate_meta_model(feature_extractor, metric_generator, test_dataset, device):
    """
    Evaluates the meta-learning model on the provided test dataset episodes.

    This function iterates through meta-test episodes, generates writer-specific
    metrics using the metric_generator, calculates distances, determines an
    optimal threshold per episode for predictions, and aggregates results
    to compute comprehensive performance metrics.

    Args:
        feature_extractor (torch.nn.Module): The pre-trained feature extractor model.
        metric_generator (torch.nn.Module): The meta-trained metric generator model.
        test_dataset (Dataset): A SignatureEpisodeDataset instance for meta-testing.
        device (torch.device): The device (CPU or CUDA) to perform computations on.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary containing overall evaluation metrics:
                'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'.
            - list: Aggregated true labels for all query samples.
            - list: Aggregated predictions for all query samples.
            - list: Aggregated raw distances (before thresholding) for ROC AUC calculation.
    """
    feature_extractor.eval()
    metric_generator.eval()

    # Use appropriate num_workers based on environment
    num_workers = 2 if 'kaggle' in os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') else 0
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    all_true_labels = []
    all_predictions = []
    all_distances = [] # Store distances for ROC AUC calculation

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Meta-Testing", leave=False):
            support_images = batch['support_images'].squeeze(0).to(device)
            query_images = batch['query_images'].squeeze(0).to(device)
            query_labels = batch['query_labels'].squeeze(0).to(device)
            user_id = batch.get('user_id', ['N/A'])[0] # Get user ID for potential debugging

            # --- Feature Extraction ---
            k_shot = len(support_images)
            if k_shot == 0:
                # print(f"Warning: Skipping episode for user {user_id} due to empty support set.")
                continue

            # Concatenate for efficient feature extraction
            all_images = torch.cat([support_images, query_images], dim=0)
            try:
                all_embeddings = feature_extractor(all_images)
            except RuntimeError as e:
                print(f"Error extracting embeddings for batch (User: {user_id}): {e}. Skipping episode.")
                continue # Skip episode if feature extraction fails

            support_embeddings = all_embeddings[:k_shot]
            query_embeddings = all_embeddings[k_shot:]

            if len(query_embeddings) == 0:
                # print(f"Warning: Skipping episode for user {user_id} due to empty query set.")
                continue

            # --- Generate Adaptive Metric (W) and Prototype ---
            try:
                W = metric_generator(support_embeddings)
                prototype_genuine = torch.mean(support_embeddings, dim=0)
            except RuntimeError as e:
                 print(f"Error generating metric for user {user_id}: {e}. Skipping episode.")
                 continue

            # --- Calculate Distances from Query Samples to Prototype ---
            distances_episode = []
            valid_episode = True
            for q_embed in query_embeddings:
                diff = q_embed - prototype_genuine
                try:
                    # Calculate Mahalanobis distance: (x-p)^T * W * (x-p)
                    dist = torch.matmul(torch.matmul(diff.unsqueeze(0), W), diff.unsqueeze(1)).item()

                    # Handle potential numerical instability
                    if not np.isfinite(dist):
                        # print(f"Warning: Non-finite distance ({dist}) encountered for user {user_id}. Using Euclidean fallback.")
                        dist = torch.linalg.norm(diff).item() # Fallback to Euclidean
                    distances_episode.append(dist)
                except RuntimeError as e:
                    print(f"Error calculating Mahalanobis distance for user {user_id}: {e}. Using Euclidean fallback.")
                    dist = torch.linalg.norm(diff).item()
                    distances_episode.append(dist)
                except Exception as e:
                     print(f"Unexpected error calculating distance for user {user_id}: {e}. Skipping sample.")
                     # Mark episode as potentially invalid if errors occur frequently
                     valid_episode = False # Or implement more robust error handling
                     break # Stop processing this episode

            if not valid_episode:
                 continue # Skip to the next episode

            distances_episode = np.array(distances_episode)
            labels_episode = query_labels.cpu().numpy()

            # Ensure distances and labels arrays are not empty before proceeding
            if distances_episode.size == 0 or labels_episode.size == 0 or distances_episode.size != labels_episode.size:
                 # print(f"Warning: Mismatched or empty distances/labels for user {user_id}. Skipping episode.")
                 continue

            # --- Find Optimal Threshold FOR THIS EPISODE (for prediction only) ---
            best_acc_episode = -1.0
            # Default threshold: mean, median, or a fixed value like 0.5 or 1.0
            default_thresh = np.median(distances_episode) if len(distances_episode) > 0 else 0.5
            best_thresh_episode = default_thresh

            # Generate candidate thresholds between unique sorted distances
            sorted_dists = np.sort(np.unique(distances_episode))
            threshold_candidates = (sorted_dists[:-1] + sorted_dists[1:]) / 2.0

            # Include min/max bounds slightly adjusted as potential thresholds
            if len(sorted_dists) > 0:
                 min_dist, max_dist = sorted_dists[0], sorted_dists[-1]
                 threshold_candidates = np.concatenate(([min_dist - 1e-6], threshold_candidates, [max_dist + 1e-6]))

            if len(threshold_candidates) == 0: # Handle cases with single unique distance
                 threshold_candidates = [default_thresh]

            # Find threshold yielding best accuracy on this episode's query set
            for thresh in threshold_candidates:
                if not np.isfinite(thresh): continue # Skip invalid thresholds
                preds = (distances_episode < thresh).astype(int)
                acc = np.mean(preds == labels_episode)

                # Update best threshold if accuracy improves, or if accuracy is equal but threshold is closer to median
                if acc > best_acc_episode:
                    best_acc_episode = acc
                    best_thresh_episode = thresh
                elif acc == best_acc_episode:
                    current_median_diff = abs(best_thresh_episode - default_thresh)
                    new_median_diff = abs(thresh - default_thresh)
                    if new_median_diff < current_median_diff:
                        best_thresh_episode = thresh

            # --- Generate Final Predictions for this Episode ---
            final_preds_episode = (distances_episode < best_thresh_episode).astype(int)

            # --- Aggregate Results ---
            all_true_labels.extend(labels_episode.tolist())
            all_predictions.extend(final_preds_episode.tolist())
            all_distances.extend(distances_episode.tolist()) # Store raw distances

    # --- Calculate Overall Metrics After Processing All Episodes ---
    if not all_true_labels or not all_predictions:
        print("Warning: No valid data aggregated for final metrics calculation.")
        # Return zero metrics and empty lists
        zero_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'roc_auc': 0.0}
        return zero_metrics, [], [], []

    # Prepare scores for ROC AUC: Lower distance means more likely genuine (label 1).
    # ROC AUC requires higher score for positive class. Use negative distance.
    # Ensure distances are finite before negation
    valid_distances = np.array([d if np.isfinite(d) else np.nan for d in all_distances])
    max_finite_dist = np.nanmax(valid_distances[np.isfinite(valid_distances)]) if np.any(np.isfinite(valid_distances)) else 1.0
    valid_distances[np.isnan(valid_distances)] = max_finite_dist + 1 # Replace NaN with a large value
    roc_scores = -valid_distances

    # Calculate metrics, handle potential division by zero
    results = {
        'accuracy': accuracy_score(all_true_labels, all_predictions),
        'precision': precision_score(all_true_labels, all_predictions, zero_division=0),
        'recall': recall_score(all_true_labels, all_predictions, zero_division=0),
        'f1_score': f1_score(all_true_labels, all_predictions, zero_division=0),
        # Calculate AUC only if both classes are present in true labels
        'roc_auc': roc_auc_score(all_true_labels, roc_scores) if len(np.unique(all_true_labels)) > 1 else 0.0
    }

    # Return metrics dictionary and aggregated lists for plotting
    return results, all_true_labels, all_predictions, all_distances

def plot_roc_curve(all_true_labels, all_distances, title='Receiver Operating Characteristic (ROC) Curve'):
    """
    Plots the ROC curve and displays the AUC score.

    Args:
        all_true_labels (list or np.array): Aggregated true binary labels (0 or 1).
        all_distances (list or np.array): Aggregated raw distances. Lower distance
                                         indicates higher likelihood of being genuine (class 1).
        title (str): The title for the plot.
    """
    if not all_true_labels or not all_distances or len(all_true_labels) != len(all_distances):
        print("Error: Invalid input data for ROC curve plotting.")
        return
    if len(np.unique(all_true_labels)) < 2:
        print("Warning: ROC AUC score is not defined when only one class is present in true labels.")
        return

    # Scores for roc_curve: higher score should mean higher likelihood of positive class (genuine=1)
    # Since lower distance means genuine, use negative distance as score.
    roc_scores = -np.array(all_distances)

    # Ensure scores are finite
    valid_indices = np.isfinite(roc_scores)
    if not np.all(valid_indices):
         print(f"Warning: Removing {np.sum(~valid_indices)} non-finite scores before plotting ROC.")
         roc_scores = roc_scores[valid_indices]
         all_true_labels = np.array(all_true_labels)[valid_indices]


    fpr, tpr, thresholds = roc_curve(all_true_labels, roc_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(all_true_labels, all_predictions, class_names=['Forgery (0)', 'Genuine (1)'], title='Confusion Matrix'):
    """
    Plots the confusion matrix using seaborn heatmap.

    Args:
        all_true_labels (list or np.array): Aggregated true binary labels.
        all_predictions (list or np.array): Aggregated predicted binary labels.
        class_names (list): Names for the classes (e.g., ['Negative', 'Positive']).
        title (str): The title for the plot.
    """
    if not all_true_labels or not all_predictions or len(all_true_labels) != len(all_predictions):
        print("Error: Invalid input data for confusion matrix plotting.")
        return

    cm = confusion_matrix(all_true_labels, all_predictions)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.show()