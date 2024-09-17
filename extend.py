# on my machine, i ran this with:  
#   python3.13 -B extend.py ../moot/optimize/[comp]*/*.csv

import sys,random, time, math, stats, os
from ezr import the, DATA, csv, dot

# def show(lst):
#   return print(*[f"{word:6}" for word in lst], sep="\t")

# def myfun(train):
#   d    = DATA().adds(csv(train))
#   x    = len(d.cols.x)
#   size = len(d.rows)
#   dim  = "small" if x <= 5 else ("med" if x < 12 else "hi")
#   size = "small" if size< 500 else ("med" if size<5000 else "hi")
#   return [dim, size, x,len(d.cols.y), len(d.rows), train[17:]]

# random.seed(the.seed) #  not needed here, but good practice to always take care of seeds
# show(["dim", "size","xcols","ycols","rows","file"])
# show(["------"] * 6)
# [show(myfun(arg)) for arg in sys.argv if arg[-4:] == ".csv"]

# cmds = ''
# for folder in os.listdir('data/optimize'):
#   for file in os.listdir(f'data/optimize/{folder}'):
#     if file.endswith('csv'):
#       cmds += f'python3.13 extend.py data/optimize/{folder}/{file} > output/{file} &\n'

# print(cmds)
      

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

