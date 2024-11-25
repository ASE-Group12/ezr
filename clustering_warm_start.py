from ezr import the, DATA, csv
import sys, random, stats, time, math


def dbscan(data: DATA, eps: float = 0.5, min_samples: int = 5, max_iterations=300):
    """
    Perform DBSCAN clustering on the given data.
    
    :param data: DATA object containing the rows to be clustered
    :param eps: The maximum distance between two samples for them to be considered as in the same neighborhood
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point
    :return: List of representative points for each cluster
    """
    def point_to_hashable(point):
        """Convert a point to a hashable type."""
        if isinstance(point, list):
            return tuple(point_to_hashable(x) for x in point)
        return point

    def get_neighbors(point, eps):
        return [row for row in data.rows if data.dist(point, row) <= eps]

    def expand_cluster(point, neighbors, cluster, visited, iteration_count):
        cluster.append(point)
        for neighbor in neighbors:
            if iteration_count[0] >= max_iterations:
                return
            iteration_count[0] += 1
            neighbor_hash = point_to_hashable(neighbor)
            if neighbor_hash not in visited:
                visited.add(neighbor_hash)
                new_neighbors = get_neighbors(neighbor, eps)
                if len(new_neighbors) >= min_samples:
                    neighbors.extend([n for n in new_neighbors if point_to_hashable(n) not in visited])
            if neighbor not in cluster:
                cluster.append(neighbor)

    visited = set()
    clusters = []
    noise = []
    iteration_count = [0]  # Use a list to allow modification inside nested functions

    for point in data.rows:
        if iteration_count[0] >= max_iterations:
            break
        point_hash = point_to_hashable(point)
        if point_hash in visited:
            continue
        visited.add(point_hash)
        neighbors = get_neighbors(point, eps)
        if len(neighbors) < min_samples:
            noise.append(point)
        else:
            cluster = []
            expand_cluster(point, neighbors, cluster, visited, iteration_count)
            clusters.append(cluster)

    representatives = []
    for cluster in clusters:
        centroid = data.mid()
        representative = min(cluster, key=lambda x: data.dist(x, centroid))
        representatives.append(representative)

    return representatives



def kmeans(data: DATA, k: int = 5, max_iters: int = 200):
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
        new_centroids = []
        for i in range(k):
            if len(clusters[i].rows) == 0:
                new_centroids.append(random.choice(data.rows))
            else:
                new_centroids.append(clusters[i].mid())

        # Check for convergence (if centroids haven't changed, stop)
        if all(data.dist(centroids[i], new_centroids[i]) < 1e-4 for i in range(k)):
            break

        # Update centroids for the next iteration
        centroids = new_centroids

    representatives = []
    for i, cluster in enumerate(clusters):
        centroid = centroids[i]
        min_distance = 1E32
        representative = None
        for row in cluster.rows:
            distance = cluster.dist(row, centroid)
            if distance < min_distance:
                min_distance = distance
                representative = row
        representatives.append(representative)

    return representatives

def twoFarClustering(data: DATA):
    c = data.cluster()
    representatives = []

    for i, j in c.nodes():
        if j == True:
            min_distance = 1E32
            representative = None
            for row in i.data.rows:
                distance = i.data.dist(row, i.data.mid())
                if distance < min_distance:
                    min_distance = distance
                    representative = row
            representatives.append(representative)
    
    return representatives

def cold(data: DATA):
    return []

d = DATA().adds(csv(sys.argv[1]))

r = dbscan(d)
print(r)

b4 = [d.chebyshev(row) for row in d.rows]
somes = [stats.SOME(b4,f"asIs,{len(d.rows)}")]
rnd = lambda z: z
scoring_policies = [('exploit', lambda B, R,: B - R), ('explore', lambda B, R :  (math.exp(B) + math.exp(R))/ (1E-30 + abs(math.exp(B) - math.exp(R))))]
cluster_algs = {cold: 'cold', dbscan: 'dbscan', kmeans: 'kmeans', twoFarClustering: 'twoFar'}

the.label=5
representatives = kmeans(d, the.label)

for cluster_alg, name in cluster_algs.items():
    representatives = cluster_alg(d)
    for what,how in scoring_policies:
        for the.Last in [20, 30, 40]:
            for the.branch in [False, True]:
                start = time.time()
                result = []
                warm_result = []
                runs = 0
                for _ in range(20):
                    d = d.shuffle()
                    d.rows = [row for row in d.rows if row not in representatives]
                    d.rows = representatives + d.rows
                    w=d.activeLearning(score=how)
                    runs += len(w)
                    warm_result += [rnd(d.chebyshev(w[0]))]
                    result += [rnd(d.chebyshev(w[0]))]

                pre=f"{name} {what}/b={the.branch}" if the.Last >0 else "rrp"
                tag = f"{pre},{int(runs/20)}"
                print(tag, f": {(time.time() - start) /20:.2f} secs")
                somes +=   [stats.SOME(result,    tag)]

stats.report(somes, 0.01)