import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from utils import Animal10Dataset, CaltechDataset

def get_descriptors(X_values):
    all_descriptors = []  # The "Soup" of all features

    # Initialize SIFT
    sift = cv2.SIFT_create()

    counter = 0
    print("Extracting SIFT features...")

    for img_path in X_values:
        if counter % 200 == 0:
            print(f"Image: [{counter}/{X_values.shape[0]}]")
        # Read Image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        # Keypoint Detection (SIFT)
        keypoints, descriptors = sift.detectAndCompute(img, None)

        # If features are found, store them
        if descriptors is not None:
            all_descriptors.append(descriptors)

        counter += 1
    
    print(f"Image: [{counter}/{X_values.shape[0]}]")

    return all_descriptors

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
    
def generate_features(all_descriptors, kmeans, n):

    features = [] # List to hold the K-dimensional histograms (the features)

    for descriptors in all_descriptors:
        # Quantization: Map descriptors to the K visual words
        # This returns an array of size (Number of descriptors) containing cluster indices (0 to K-1)
        visual_word_indices = kmeans.predict(descriptors)

        # Histogram Creation: Count the occurrences of each visual word
        # The result is a K-dimensional vector representing the image
        histogram, _ = np.histogram(visual_word_indices, bins=range(n + 1))
        
        # Normalize the histogram (Optional but recommended for robust SVM training)
        histogram = histogram.astype("float")
        histogram /= (histogram.sum() + 1e-7) # L1 Normalization
        
        # Store the results
        features.append(histogram)

    
    return features

def main(dataset, Ks, Ns):
    X = dataset.filepaths
    y = dataset.labels

    X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y # Use stratify if you want balanced classes across splits
        )

    train_descriptors = get_descriptors(X_train)
    test_descriptors = get_descriptors(X_test)

    for k in Ks:
        kmeans = train_kmeans_model(train_descriptors, k)

        for n in Ns:
            train_features = generate_features(train_descriptors, kmeans, n)
            test_features = generate_features(test_descriptors, kmeans, n)

            print("Starting SVM training...")

            # Initialize the SVM classifier
            svm = SVC(C=1.0, kernel="rbf", decision_function_shape="ovr", random_state=42)

            # Train the SVM on the BoVW histograms
            svm.fit(train_features, y_train)
            print("SVM training complete.")
            y_pred = svm.predict(test_features)

            # Generate report
            report = classification_report(y_test, y_pred, target_names=dataset.encoder.classes_, output_dict=True)
            df_report = pd.DataFrame(report).T

            # Save Report to CSV
            outdir=f"Local_Feature_Results/{dataset.name}"

            import os
            os.makedirs(outdir, exist_ok=True)

            csv_filename = f'{outdir}/report_metrics_k_{k}_n_{n}.csv'
            df_report.to_csv(csv_filename, index=True)

            print(f"Classification report successfully saved to {csv_filename}")

if __name__ == "__main__":
    main(CaltechDataset(), Ks=[500, 1000, 2000], Ns=[500, 1000, 2000])
    main(Animal10Dataset(), Ks=[500, 1000, 2000], Ns=[500, 1000, 2000])