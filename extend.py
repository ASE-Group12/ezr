import sys, time, math, stats, os
from ezr import the, DATA, csv

def myfun():
  cmds = ''
  for folder in os.listdir('data/optimize'):
    for file in os.listdir(f'data/optimize/{folder}'):
      if file.endswith('csv'):
        d = DATA().adds(csv(file))
        if len(d.cols.x) > 6:
          cmds += f'python3.13 extend.py data/optimize/{folder}/{file} > output_highX/{file} &\n'
        else:
          cmds += f'python3.13 extend.py data/optimize/{folder}/{file} > output_LowX/{file} &\n'

  print(cmds)     


d = DATA().adds(csv(sys.argv[1]))
b4 = [d.chebyshev(row) for row in d.rows]
somes = [stats.SOME(b4,f"asIs,{len(d.rows)}")]
rnd = lambda z: z
scoring_policies = [
  ('exploit', lambda B, R,: B - R),
  ('explore', lambda B, R :  (math.exp(B) + math.exp(R))/ (1E-30 + abs(math.exp(B) - math.exp(R))))]

for what,how in scoring_policies:
  for the.Last in [0,20, 30, 40]:
    for the.branch in [False, True]:
      start = time.time()
      result = []
      runs = 0
      for _ in range(20):
         tmp=d.shuffle().activeLearning(score=how)
         runs += len(tmp)
         result += [rnd(d.chebyshev(tmp[0]))]

      pre=f"{what}/b={the.branch}" if the.Last >0 else "rrp"
      tag = f"{pre},{int(runs/20)}"
      print(tag, f": {(time.time() - start) /20:.2f} secs")
      somes +=   [stats.SOME(result,    tag)]

stats.report(somes, 0.01)

