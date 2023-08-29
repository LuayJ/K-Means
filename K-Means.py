import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import adjusted_rand_score

# Actual dataset
df = pd.read_csv('fashion-mnist_train.csv')
df_data = df.iloc[:, 1:]  # Unlabeled data
df_labels = df.iloc[:, 0]  # Data labels to  be used for RAND later
df_labels = np.array(df_labels)

# Low Dimension (edited) dataset used for testing if algorithm worked
# df = pd.read_csv('heart.csv')
# df_data = df.iloc[:, :2]
# df_labels = df.iloc[:, 2]
# df_labels = np.array(df_labels)


# k should be an integer (number of clusters)
def K_Means(df, k):
    k_means = pd.DataFrame(df.sample(k, replace=False))  # Stores current iteration cluster means and chooses k random points to act as cluster centers
    prev_k_means = pd.DataFrame()  # Stores previous iteration cluster means
    cluster_dist = pd.DataFrame()  # Stores cluster distance

    while not k_means.equals(prev_k_means):
        k = 0
        for i, mean in k_means.iterrows():
            cluster_dist[k] = (df[k_means.columns] - np.array(mean)).pow(2).sum(1).pow(0.5)  # Distance of each point from each cluster
            k += 1

        df['Cluster'] = cluster_dist.idxmin(axis=1)  # Assigns the point to the nearest cluster and writes to the df

        prev_k_means = k_means  # Puts the current means / centers in the previous df
        k_means = pd.DataFrame()  # Clears current data
        k_means = df.groupby('Cluster').agg(np.mean)  # Calculates new means

        # For plotting when data dimensionality is low
        # print('Plotting...')
        # [plt.scatter(
        #     x=df['V1'],
        #     y=df['V2'].where(df['Cluster'] == c)
        # ) for c in range(k)]
        #
        # plt.scatter(x=k_means['V1'], y=k_means['V2'], color='#000000')
        # plt.show()
        # print(k)

    return df


# Segment for testing different k-values
# for k in range(5, 16):
#     pred = K_Means(df_data, k)
#     pred = pred['Cluster']
#     pred = np.array(pred)
#
#     score_rand = rand_score(df_labels, pred)
#     score_adjustedRand = adjusted_rand_score(df_labels, pred)
#
#     print('k:', k)
#     print(score_rand)
#     print(score_adjustedRand)

# Using best k
k = 3  # Number of clusters to be used
pred = K_Means(df_data, k)
pred = pred['Cluster']
pred = np.array(pred)

score_rand = rand_score(df_labels, pred)
score_adjustedRand = adjusted_rand_score(df_labels, pred)

print('k:', k)
print(score_rand)
print(score_adjustedRand)
