import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# Audio Files
audio_files = [
    'audio/cat_1.wav', 'audio/cat_2.wav', 'audio/cat_3.wav', 'audio/cat_4.wav', 
    'audio/dog_1.wav', 'audio/dog_2.wav', 'audio/dog_3.wav', 'audio/dog_4.wav', 
    'audio/cow_1.wav'
]

# Function to extract features.
def extract_features(file_name, default_n_fft=512):
    """
    Parameters:
    - file_name: Path to the audio file.
    - default_n_fft: Number of FFT components.
    """
    audio, sample_rate = librosa.load(file_name)
    
    # Check if the audio is shorter than the default_n_fft and adjust if necessary
    n_fft_value = min(default_n_fft, len(audio))
    
    # Ensure audio is at least n_fft_value in length by padding
    if len(audio) < n_fft_value:
        audio = librosa.util.fix_length(audio, size=n_fft_value)
    
    hop_length_value = n_fft_value // 4
    
    # Proceed with adjusted feature extraction
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_fft=n_fft_value, hop_length=hop_length_value)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_fft=n_fft_value, hop_length=hop_length_value)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_fft=n_fft_value, hop_length=hop_length_value)
    
    # Corrected approach for harmonic component: Compute the STFT first, then derive the harmonic component
    D = librosa.stft(audio, n_fft=n_fft_value, hop_length=hop_length_value)
    harmonic, percussive = librosa.decompose.hpss(D)
    
    # Since tonnetz expects a harmonic time-domain signal, we need to convert back
    harmonic_audio = librosa.istft(harmonic)
    tonnetz = librosa.feature.tonnetz(y=harmonic_audio, sr=sample_rate)
    
    # Combine all features into one array
    features = np.hstack([np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(contrast, axis=1), np.mean(tonnetz, axis=1)])
    
    return features

#Plot the clustered features in 2D using PCA for dimensionality reduction.
def plot_clusters(features):

    # Reduce dimensions to 2D for visualization using PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # Plot the clustered features
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', edgecolor='k', s=70)
    plt.title("Clustered Audio Features with Enhanced Feature Set")
    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    plt.colorbar(label='Cluster')
    plt.show()


if __name__ == "__main__":
    # Extract features for each audio file using the function that we created.
    features = np.array([extract_features(file) for file in audio_files])

    # Perform k-means clustering.
    kmeans = KMeans(n_clusters=3, random_state=42).fit(features)
    labels = kmeans.labels_

    # Print out the cluster assignments.
    for i, label in enumerate(labels):
        print(f'File: {audio_files[i]} - Cluster: {label}')

    #This line is creating the plot.
    plot_clusters(features)

    # Evaluation metrics.
    print("\nMetric Results")
    silhouette_avg = silhouette_score(features, kmeans.labels_)
    print(f"Silhouette Coefficient: {silhouette_avg:.3f}")

    calinski_harabasz_index = calinski_harabasz_score(features, kmeans.labels_)
    print(f"Calinski-Harabasz Index: {calinski_harabasz_index:.3f}")

    davies_bouldin_index = davies_bouldin_score(features, kmeans.labels_)
    print(f"Davies-Bouldin Index: {davies_bouldin_index:.3f}")
