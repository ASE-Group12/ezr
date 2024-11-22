from ezr import the, DATA, csv
import sys, random, stats, time, math

def dbscan(data: DATA, eps: float, min_samples: int):
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

    def expand_cluster(point, neighbors, cluster, visited):
        cluster.append(point)
        for neighbor in neighbors:
            neighbor_hash = point_to_hashable(neighbor)
            if neighbor_hash not in visited:
                visited.add(neighbor_hash)
                new_neighbors = get_neighbors(neighbor, eps)
                if len(new_neighbors) >= min_samples:
                    neighbors.extend([n for n in new_neighbors if point_to_hashable(n) not in set(map(point_to_hashable, neighbors))])
            if neighbor not in [c for clust in clusters for c in clust]:
                cluster.append(neighbor)

    visited = set()
    clusters = []
    noise = []

    for point in data.rows:
        point_hash = point_to_hashable(point)
        if point_hash in visited:
            continue
        visited.add(point_hash)
        neighbors = get_neighbors(point, eps)
        if len(neighbors) < min_samples:
            noise.append(point)
        else:
            cluster = []
            expand_cluster(point, neighbors, cluster, visited)
            clusters.append(cluster)

    representatives = []
    for cluster in clusters:
        centroid = data.mid()
        representative = min(cluster, key=lambda x: data.dist(x, centroid))
        representatives.append(representative)

    return representatives



d = DATA().adds(csv(sys.argv[1]))
b4 = [d.chebyshev(row) for row in d.rows]
somes = [stats.SOME(b4,f"asIs,{len(d.rows)}")]
rnd = lambda z: z
scoring_policies = [
('exploit', lambda B, R,: B - R),
('explore', lambda B, R :  (math.exp(B) + math.exp(R))/ (1E-30 + abs(math.exp(B) - math.exp(R))))]
the.label = 10
eps = 0.5  # Adjust based on your data
min_samples = 5  # Adjust based on your data

for what,how in scoring_policies:
    for the.Last in [20, 30, 40]:
        for the.branch in [False, True]:
            start = time.time()
            result = []
            warm_result = []
            runs = 0
            for _ in range(20):
                d = d.shuffle()
                c = d.activeLearning(score=how)
                representatives = dbscan(d, eps, min_samples)
                d.rows = [row for row in d.rows if row not in representatives]
                d.rows = representatives + d.rows
                w = d.activeLearning(score=how)
                runs += len(c)
                warm_result += [rnd(d.chebyshev(w[0]))]
                result += [rnd(d.chebyshev(c[0]))]

            pre = f"cold {what}/b={the.branch}" if the.Last > 0 else "cold rrp"
            tag = f"{pre},{int(runs/20)}"
            print(tag, f": {(time.time() - start) /20:.2f} secs")
            somes += [stats.SOME(result, tag)]

            pre = f"warm {what}/b={the.branch}" if the.Last > 0 else "warm rrp"
            tag = f"{pre},{int(runs/20)}"
            print(tag, f": {(time.time() - start) /20:.2f} secs")
            somes += [stats.SOME(warm_result, tag)]

stats.report(somes, 0.01)