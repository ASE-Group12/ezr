from ezr import the, DATA, csv
import sys, random, stats, time, math

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


d = DATA().adds(csv(sys.argv[1]))

b4 = [d.chebyshev(row) for row in d.rows]
somes = [stats.SOME(b4,f"asIs,{len(d.rows)}")]
rnd = lambda z: z
scoring_policies = [
('exploit', lambda B, R,: B - R),
('explore', lambda B, R :  (math.exp(B) + math.exp(R))/ (1E-30 + abs(math.exp(B) - math.exp(R))))]
the.label=10
representatives = kmeans(d, the.label)
for startstrat in ['warm', 'cold']:
    for what,how in scoring_policies:
        for the.Last in [20, 30, 40]:
            for the.branch in [False, True]:
                start = time.time()
                result = []
                warm_result = []
                runs = 0
                for _ in range(20):
                    d = d.shuffle()
                    if startstrat == 'warm':
                        d.rows = [row for row in d.rows if row not in representatives]
                        d.rows = representatives + d.rows
                    w=d.activeLearning(score=how)
                    runs += len(w)
                    warm_result += [rnd(d.chebyshev(w[0]))]
                    result += [rnd(d.chebyshev(w[0]))]

                pre=f"{startstrat} {what}/b={the.branch}" if the.Last >0 else "rrp"
                tag = f"{pre},{int(runs/20)}"
                print(tag, f": {(time.time() - start) /20:.2f} secs")
                somes +=   [stats.SOME(result,    tag)]

stats.report(somes, 0.01)