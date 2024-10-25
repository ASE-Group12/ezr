from ezr import the, DATA, csv
import os
import random

def kmeans(data: DATA, k: int, max_iters: int = 100):
    # Step 1: Randomly initialize centroids by selecting k rows from data
    centroids = random.sample(data.rows, k)
    
    for iteration in range(max_iters):
        # Step 2: Assign each row to the nearest centroid
        clusters = [DATA().add(data.cols.names) for _ in centroids]
        for row in data.rows:
            distances = [data.dist(row, centroid) for centroid in centroids]
            closest_centroid = distances.index(min(distances))
            clusters[closest_centroid].add(row)

        # Step 3: Update centroids
        new_centroids = [clusters[i].mid() for i in range(k)]

        # Check for convergence (if centroids haven't changed, stop)
        if all(data.dist(centroids[i], new_centroids[i]) < 1e-4 for i in range(k)):
            break

        # Update centroids for the next iteration
        centroids = new_centroids

    return clusters, centroids


def myfun():
    d = DATA().adds(csv('data/optimize/misc/auto93.csv'))
    clusters, centroids = kmeans(d, 10)
    print(centroids)

#   for folder in os.listdir('data/optimize'):
#     for file in os.listdir(f'data/optimize/{folder}'):
#       if file.endswith('csv'):
#         d = DATA().adds(csv(f'data/optimize/{folder}/{file}'))
#         clusters, centroids = kmeans(d, 10)
#         print(centroids, clusters)
#         break


myfun()
