import numpy as np
from sklearn.cluster import KMeans


def normalize_data(data):
    counts = np.array([
        data['after_initial_exchange'],
        data['first_turn_no_greeting'],
        data['first_turn_w_greeting'],
        data['no_proposal']
    ])

    total_dialogues = np.sum(counts, axis=1)

    normalized_counts = counts / total_dialogues[:, np.newaxis]

    data['after_initial_exchange'] = normalized_counts[0].tolist()
    data['first_turn_no_greeting'] = normalized_counts[1].tolist()
    data['first_turn_with_greeting'] = normalized_counts[2].tolist()
    data['no_proposal'] = normalized_counts[3].tolist()

    return data


def main():
    data = {
        'individual': ['bob', 'lindsay', 'michelle', 'niki', 'ross'],
        'after_initial_exchange': [11, 5, 11, 4, 13],
        'first_turn_no_greeting': [4, 1, 2, 2, 5],
        'first_turn_with_greeting': [10, 5, 13, 3, 9],
        'no_proposal': [1, 1, 2, 0, 2]
    }

    normalized_data = normalize_data(data)

    X = np.array([
        normalized_data['after_initial_exchange'],
        normalized_data['first_turn_no_greeting'],
        normalized_data['first_turn_with_greeting'],
        normalized_data['no_proposal']
    ])

    # Perform clustering with k-means
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X.T)

    normalized_data['cluster'] = kmeans.labels_.tolist()

    print(normalized_data)


if __name__ == "__main__":
    main()
