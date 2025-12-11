import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from utils import Animal10Dataset, CaltechDataset

def get_descriptors(X_values):
    all_descriptors = []

    # Initialize SIFT
    sift = cv2.SIFT_create()

    print("Extracting SIFT features...")

    no_descriptors = []
    for i, img_path in enumerate(X_values):
        # Progress Ticker
        if i % 1000 == 0:
            print(f"Image: [{i}/{X_values.shape[0]}]")

        # Read Image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        # Keypoint Detection (SIFT)
        keypoints, descriptors = sift.detectAndCompute(img, None)

        # If features are found, store them
        if descriptors is None:
            no_descriptors.append(i)
        else:
            all_descriptors.append(descriptors)

    
    print(f"Image: [{i}/{X_values.shape[0]}]")

    return all_descriptors, no_descriptors

def train_kmeans_model(all_descriptors, K):
    # Stack all descriptors vertically for clustering
    all_descriptors_stacked = np.vstack(all_descriptors)

    print(f"Starting K-Means clustering with K = {K}...")

    # Initialize the K-Means model
    kmeans = MiniBatchKMeans(n_clusters=K, init="k-means++", n_init="auto", random_state=42, verbose=0)
    # Fit the model to the stacked descriptors
    kmeans.fit(all_descriptors_stacked)

    print("Clustering complete.")

    return kmeans
    
def generate_features(all_descriptors, kmeans, K):

    features = [] # List to hold the K-dimensional histograms (the features)

    for descriptors in all_descriptors:
        # Map descriptors to the K visual words
        visual_word_indices = kmeans.predict(descriptors)

        # Count the occurrences of each visual word
        histogram, _ = np.histogram(visual_word_indices, bins=range(K + 1))
        
        # Normalize the histogram
        histogram = histogram.astype("float")
        histogram /= (histogram.sum() + 1e-7) # L1 Normalization
        
        # Store the results
        features.append(histogram)

    
    return features

def main(dataset):
    X = dataset.filepaths
    y = dataset.labels


    X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y # Use stratify if you want balanced classes across splits
        )
    
    train_descriptors, to_remove = get_descriptors(X_train)
    # Remove corresponding y label
    for idx in to_remove:
        print(f"removing {idx} from train set")
        y_train = np.delete(y_train, idx)

    test_descriptors, to_remove = get_descriptors(X_test)
    for idx in to_remove:
        print(f"removing {idx} from test set")
        y_test = np.delete(y_test, idx)

    Ks = [500, 750, 1000, 1500, 2000]
    for K in Ks:
        kmeans = train_kmeans_model(train_descriptors, K)
        
        train_features = generate_features(train_descriptors, kmeans, K)
        test_features = generate_features(test_descriptors, kmeans, K)

        print("Starting SVM training...")

        # Initialize the SVM classifier
        svm = SVC(C=1.0, kernel="rbf", decision_function_shape="ovr", random_state=42)

        # Train the SVM on the BoVW histograms
        svm.fit(train_features, y_train)

        print("SVM training complete.")

        y_pred = svm.predict(test_features)
        # You can then use metrics like classification_report, accuracy_score, etc. to evaluate performance.
        report = classification_report(y_test, y_pred, target_names=dataset.encoder.classes_, output_dict=True)

        df_report = pd.DataFrame(report).T

        # Save DataFrame to CSV
        csv_filename = f'{dataset.name}_report_metrics_K_{K}.csv'
        df_report.to_csv(csv_filename, index=True)

        print(f"Classification report successfully saved to {csv_filename}")

if __name__ == "__main__":
    main(Animal10Dataset())
    main(CaltechDataset())