
# Audio Feature Clustering

## Introduction
This project applies machine learning techniques to cluster audio files based on their features. It extracts features from audio samples, performs k-means clustering, and visualizes the results, offering insights into the similarities among audio samples.

## Installation
To run this project, you need to have Python installed on your machine, along with the following libraries:
- `librosa` for audio processing,
- `numpy` for numerical operations,
- `matplotlib` for plotting,
- `scikit-learn` for clustering and metrics.

You can install the required libraries using pip:
```
pip install librosa numpy matplotlib scikit-learn pywt
```

## Usage
To use this script, place your audio files in the `audio` folder and run the script from the command line:
```
python audio_clustering.py
```

## Features
- Audio feature extraction using MFCC, chroma, spectral contrast, and additional features such as Harmonic-to-Noise Ratio (HNR) and wavelet features.
- Clustering audio files into groups based on their features using KMeans algorithm.
- Visualization of clusters using PCA for dimensionality reduction.
- Calculation of clustering metrics such as silhouette coefficient, Calinski-Harabasz index, and Davies-Bouldin index.

## Clustering Metrics and Optimal Values
The silhouette coefficient measures how similar an object is to its own cluster compared to other clusters. The higher the value, the better the object fits into its own cluster. Optimal value is close to 1.

The Calinski-Harabasz index is higher when clusters are dense and well separated, which relates to a model with better defined clusters. The higher the value, the better the clustering.

The Davies-Bouldin index indicates the average 'similarity' between clusters, where lower values mean the clusters are more distinct from each other. An optimal value is close to 0.

## Experimentation and Findings
During the development, various feature sets were experimented with. Incorporating Harmonic-to-Noise Ratio (HNR) and wavelet features alongside traditional audio features like MFCCs and chroma showed a mixed impact on the clustering performance. Adjusting these feature sets demonstrated the delicate balance between feature relevance and clustering effectiveness.

Additionally, the Gaussian Mixture Model (GMM) algorithm was evaluated as an alternative to KMeans for clustering. However, KMeans was ultimately chosen for its simplicity, efficiency, and the quality of clustering achieved in the context of this project.

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request for review.
