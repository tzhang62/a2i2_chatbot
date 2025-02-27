import numpy as np
from sklearn.cluster import KMeans

def normalize_data(data):
    counts = np.array([
        data['after initial exchange'],
        data['first turn, no greeting'],
        data['first turn, with greeting'],
        data['no proposal']
    ])

    # Calculate the total number of dialogues for each individual
    total_dialogues = np.sum(counts, axis=1)

    # Normalize the counts by dividing by the total number of dialogues
    normalized_counts = counts / total_dialogues[:, np.newaxis]

    data['after initial exchange'] = normalized_counts[0].tolist()
    data['first turn, no greeting'] = normalized_counts[1].tolist()
    data['first turn, with greeting'] = normalized_counts[2].tolist()
    data['no proposal'] = normalized_counts[3].tolist()

    return data

def main():
    data = {
        'participant': list(range(1, 32)),
        'after initial exchange': [0, 4, 4, 0, 0, 0, 0, 0, 4, 4, 2, 0, 1, 0, 4, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 5, 0, 6, 5, 0],
        'first turn, no greeting': [1, 3, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
        'first turn, with greeting': [5, 0, 1, 0, 0, 3, 1, 6, 0, 2, 0, 1, 0, 0, 0, 1, 0, 0, 3, 3, 0, 2, 1, 0, 0, 1, 0, 0, 4, 1, 3],
        'no proposal': [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'saved': [9, 6, 6, 2, 7, 4, 2, 8, 6, 3, 3, 6, 4, 7, 7, 6, 4, 0, 7, 8, 3, 6, 5, 0, 7, 3, 8, 0, 5, 9, 5]
    }

    normalized_data = normalize_data(data)

    X = np.array([
        normalized_data['after initial exchange'],
        normalized_data['first turn, no greeting'],
        normalized_data['first turn, with greeting'],
        normalized_data['no proposal']
    ])

    # Perform clustering with k-means
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X.T)

    normalized_data['cluster'] = kmeans.labels_.tolist()

    print(normalized_data['participant'])
    print(normalized_data['cluster'])


if __name__ == "__main__":
    main()
