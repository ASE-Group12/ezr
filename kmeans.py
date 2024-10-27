from ezr import the, DATA, csv
import os, random, stats, time, math

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


def myfun():
    d = DATA().adds(csv('data/optimize/misc/auto93.csv'))
    d = d.shuffle()
    representatives = kmeans(d, 10)
    d.rows = [row for row in d.rows if row not in representatives]
    d.rows = representatives + d.rows
    b4 = [d.chebyshev(row) for row in d.rows]
    somes = [stats.SOME(b4,f"asIs,{len(d.rows)}")]
    rnd = lambda z: z
    scoring_policies = [
    ('exploit', lambda B, R,: B - R),
    ('explore', lambda B, R :  (math.exp(B) + math.exp(R))/ (1E-30 + abs(math.exp(B) - math.exp(R))))]
    the.label=10
    for what,how in scoring_policies:
        for the.Last in [0, 10, 20, 30, 40]:
            for the.branch in [False, True]:
                start = time.time()
                result = []
                warm_result = []
                runs = 0
                for _ in range(20):
                    tmp=d.activeLearning(score=how)
                    tmp1=d.shuffle().activeLearning(score=how)
                    runs += len(tmp1)
                    warm_result += [rnd(d.chebyshev(tmp[0]))]
                    result += [rnd(d.chebyshev(tmp1[0]))]

                pre=f"cold {what}/b={the.branch}" if the.Last >0 else "cold rrp"
                tag = f"{pre},{int(runs/20)}"
                print(tag, f": {(time.time() - start) /20:.2f} secs")
                somes +=   [stats.SOME(result,    tag)]

                pre=f"warm {what}/b={the.branch}" if the.Last >0 else "warm rrp"
                tag = f"{pre},{int(runs/20)}"
                print(tag, f": {(time.time() - start) /20:.2f} secs")
                somes +=   [stats.SOME(warm_result,    tag)]

    stats.report(somes, 0.01)    

#   for folder in os.listdir('data/optimize'):
#     for file in os.listdir(f'data/optimize/{folder}'):
#       if file.endswith('csv'):
#         d = DATA().adds(csv(f'data/optimize/{folder}/{file}'))
#         clusters, centroids = kmeans(d, 10)
#         print(centroids, clusters)
#         break


myfun()
