# vim : set et ts=2 sw=2 :
"""
ez: ai can be easy, let me show you how   
(c) 2023, Tim Menzies, <timm@ieee.org>, BSD-2  
  
USAGE:    
1. download ez.py, etc.py, eg.py    
2. python3 eg.py [OPTIONS]   
  
OPTIONS:  

     -a --adequate    good if over adequate*best = .1
     -A --ample       only want this many rules  = 20
     -b --budget0     initial evals              = 4  
     -B --Budget      subsequent evals           = 5   
     -c --cohen       small effect size          = .35  
     -C --confidence  statistical confidence     = .05
     -d --discretizationRange    number of bins when discretizing numerical data for SNEAK = 8
     -e --effectSize  non-parametric small delta = 0.2385
     -E --Experiments number of Bootstraps       = 256
     -f --file        csv data file name         = '../data/healthCloseIsses12mths0011-easy.csv'  
     -F --Far         far search outlier control = .95 
     -h --help        print help                 = false
     -H --Half        #items for far search      = 256
     -k --k           rare class  kludge         = 1  
     -m --m           rare attribute  kludge     = 2  
     -M --Min         min size is N**Min         = .5
     -p --p           distance coefficient       = 2
     -r --ranges      max number of bins         = 16
     -R --Repeats     max number of bins         = 20
     -s --seed        random number seed         = 31210 
     -S --score       support exponent for range = 2  
     -t --todo        start up action            = 'help'   
     -T --Top         best section               = .5   
"""

import re,sys,math,random
from collections import Counter
from stats import sk
from etc import o,isa,struct,merges
import etc
from math import pi
import numpy as np




the = etc.THE(__doc__)
tiny= sys.float_info.min 

#          _   _   |   _ 
#         (_  (_)  |  _>                       

def isGoal(s):   return s[-1] in "+-!"
def isHeaven(s): return 0 if s[-1] == "-" else 1
def isNum(s):    return s[0].isupper() 

class SYM(Counter):
  "Adds `add` to Counter"
  def add(self,x,n=1): self[x] += n

  def entropy(self): 
    n = sum(self.values()) 
    return -sum(v/n*math.log(v/n,2) for _,v in self.items() if v>0)
  
  def __add__(i,j):
    k=SYM()
    [k.add(x,n) for old in [i,j] for x,n in old.items()]
    return k
  
class NUM(struct):
  def __init__(self,lst=[],txt=" "):
   self.n, self.mu, self.m2, self.sd, self.txt = 0,0,0,0,txt 
   self.lo, self.hi = sys.maxsize, -sys.maxsize
   self.heaven = isHeaven(txt)
   [self.add(x) for x in lst]

  def add(self,x):
    self.lo = min(x,self.lo)
    self.hi = max(x,self.hi)
    self.n += 1
    delta = x - self.mu
    self.mu += delta / self.n
    self.m2 += delta * (x -  self.mu)
    self.sd = 0 if self.n < 2 else (self.m2 / (self.n - 1))**.5

  def norm(self,x):
    return x=="?" and x or (x - self.lo) / (self.hi - self.lo + tiny)
 
#         ._   _          _ 
#         |   (_)  \/\/  _> 

class DATA(struct):
  def __init__(self, lsts=[], order=False):
    self.names,*rows = list(lsts) 
    self.rows = []
    self.cols = [(NUM(txt=s) if isNum(s) else SYM()) for s in self.names] 
    self.ys   = {n:c for n,(s,c) in enumerate(zip(self.names,self.cols)) if isGoal(s)}
    self.xs   = {n:c for n,(s,c) in enumerate(zip(self.names,self.cols)) 
                 if not isGoal(s) and s[-1] != "X"}
    [self.add(row) for row in rows] 
    if order: self.ordered()

  def ordered(self):
    self.rows = sorted(self.rows, key = self.d2h)

  def add(self,row): 
    self.rows += [row] 
    [col.add(x) for x,col in zip(row,self.cols) if x != "?"]

  def clone(self, rows=[], order=False):
    return DATA([self.names] + rows, order=order)

  def d2h(self,lst):
    nom = sum(abs(col.heaven - col.norm(lst[n]))**2 for n,col in self.ys.items())
    return (nom / len(self.ys))**.5

  def mid(self): 
    return [max(col,key=col.get) if isa(col,SYM) else col.mu for col in self.cols]

  def div(self):
    return [col.entropy() if isa(col,SYM) else col.sd for col in self.cols]
  
  # return all the rows binarizing all columns
  def binXRows(self):
    # for each column, obtain the number of bins and their respective ranges if the column is not a goal
    bins = []
    for col in self.cols:
      # check if the column is not a goal
      if col not in self.ys.values():
        # obtain the number of bins and their respective ranges
        bins.append(self.binCol(col))

    # using the number of bins create a vector of zeros with the same length as the number of bins
    vecSize = 0
    for b in bins:
      vecSize += b[0]
    binVec = [0] * vecSize

    # build a matrix of zeros and ones with the same number of rows as the data set and the same number of columns as the number of bins
    binRows = [[0] * vecSize for _ in range(len(self.rows))]
    # for each row in the data set convert transform it into a binary row
    for i, row in enumerate(self.rows):
      # create a vector of zeros with the appropriate length for the size of each bin applied to each column
      
      binVec = [0]  * vecSize
      # for each column in the row
      curbin = 0
      offset = 0
      for j, col in enumerate(row):
        #print(row)
        # check if the column is not a goal using the index of the column
        if j not in self.ys:
          val = row[j]
          # if the value is not missing:
          if val != "?":
            # find the bin value of the column
            binVal = self.binColVal(self.cols[j], val, bins[curbin])
            # set the value of the bin vector to 1

            binVec[offset + binVal] = 1
            offset += bins[curbin][0]
            curbin += 1
      binRows[i] = (i, binVec)
    return binRows
  
  # return the converted value of a column into a bin vector
  def binColVal(self, col, val, bn):
    # if the column is a symbol, find the index of the value in the range
    if isa(col, SYM):
      #print("SYM", val, bn, bn[1].index(val))
      return bn[1].index(val)
    else:
      # if the column is numerical, find the bin value of the value
      for i, (lo, hi) in enumerate(bn[1]):
        # if checking the first bin, only check if the value is less than the upper bound
        if i == 0 and val < hi:
          #print(col.txt, val, bn, i)
          return i
        # if checking the last bin, only check if the value is greater than the lower bound
        elif i == len(bn[1]) - 1 and val >= lo:
          #print(col.txt, val, bn, i)
          return i
        if lo <= val < hi:
          #print(col.txt, val, bn, i)
          return i
# return the number of bins for each column with their respective ranges
  def binCol(self, col):
    if isa(col, SYM):
      #find the number of unique values
      nBins = len(col)
      #find all the unique values and return them as self contained ranges
      ranges = [x for x in col]
      return nBins, ranges
    else:
      #find the number of bins
      nBins = the.discretizationRange
      #find the range of values
      ran = col.hi - col.lo
      #find the size of each bin
      binSize = ran / nBins
      #find the ranges for each bin
      ranges = [(col.lo + i * binSize, col.lo + (i + 1) * binSize) for i in range(nBins)]
      return nBins, ranges



  
#                                  _     
#          _  |   _.   _   _  o  _|_     
#         (_  |  (_|  _>  _>  |   |   \/ 
#                                     /  

  def loglike(self,row,nall,nh,m=1,k=2):
    def num(col,x):
      v     = col.sd**2 + tiny
      nom   = math.e**(-1*(x - col.mu)**2/(2*v)) + tiny
      denom = (2*math.pi*v)**.5
      return min(1, nom/(denom + tiny))
    def sym(col,x):
      return (col.get(x, 0) + m*prior) / (len(self.rows) + m)
    #------------------------------------------
    prior = (len(self.rows) + k) / (nall + k*nh)
    out   = math.log(prior)
    for c,x in etc.slots(row):
      if x != "?" and c not in self.ys:
        col  = self.cols[c]
        out += math.log((sym if isa(col,SYM) else num)(col, x))
    return out
            
#          _   ._   _|_  o  ._ _   o  _    _   /| 
#         (_)  |_)   |_  |  | | |  |  /_  (/_   | 
#              |                                  

  def smo(self, score=lambda B,R: 2*B-R, fun=None):
    def acquire(i, best, rest, rows):
      out,most = 0,-sys.maxsize
      for k,row in enumerate(rows):
        b = best.loglike(row, len(self.rows), 2, the.m, the.k)
        r = rest.loglike(row, len(self.rows), 2, the.m, the.k)
        tmp = score(b,r)
        if tmp > most: out,most = k,tmp
      if fun: fun(i, best.rows[0])
      return out
    #-----------
    random.shuffle(self.rows)
    done, todo = self.rows[:the.budget0], self.rows[the.budget0:]
    data1 = self.clone(done, order=True)
    for i in range(the.Budget):
      n = int(len(done)**the.Top + .5)
      done.append(
        todo.pop(
          acquire( i + 1 + the.budget0,
                   self.clone(data1.rows[:n]),
                   self.clone(data1.rows[n:]),
                   todo)))
      data1 = self.clone(done, order=True)
    return data1.rows[0]
                                     
#          _  |        _  _|_   _   ._ 
#         (_  |  |_|  _>   |_  (/_  |  
                                      
  def dist(self,row1,row2):
    def sym(_,x,y): 
      return 1 if x=="?" and y=="?" else (0 if x==y else 1)     
    def num(col,x,y):
      if x=="?" and y=="?" : return 1 
      x, y = col.norm(x), col.norm(y) 
      if x=="?" : x= 1 if y<.5 else 0
      if y=="?" : y= 1 if x<.5 else 0 
      return abs(x-y)  
    #-----------------
    d, n, p = 0, 0, the.p
    for c,col in  enumerate(self.cols):
      if c not in self.ys:
        n   = n + 1
        inc = (sym if isa(col,SYM) else num)(col, row1[c],row2[c])
        d   = d + inc**p 
    return (d/n)**(1/p) 

  def near(self, row1, rows=None):
    return sorted(rows or self.rows, key=lambda row2: self.dist(row1,row2))

  def far(self, rows, sortp=False, before=None):
    n     = int(len(rows) * the.Far)
    left  = before or self.near(random.choice(rows),rows)[n]
    right = self.near(left,rows)[n]
    if sortp and self.d2h(right) < self.d2h(left): left,right = right,left
    return left, right 
    
  def half(self, rows, sortp=False, before=None):
    def dist(r1,r2): return self.dist(r1, r2)
    def proj(row)  : return (dist(row,left)**2 + C**2 - dist(row,right)**2)/(2*C)
    left,right = self.far(random.choices(rows, k=min(the.Half, len(rows))),
                          sortp=sortp, before=before)
    lefts,rights,C = [],[], dist(left,right)
    for n,row in enumerate(sorted(rows, key=proj)):  
      (lefts if n < len(rows)/2 else rights).append(row)
    return lefts, rights, left
  
  def halfBin(self, items):
    total_group=10
    left = None
    lefts = []
    right = None
    rights = []
    rand = random.choice(items)
    max_r = -float('inf')
    min_r = float('inf')
    for item in items:
        item.r = sum(item.item)
        item.d = sum([a_i - b_i for a_i, b_i in zip(item.item, rand.item)])
        if item.r > max_r:
            max_r = item.r
        if item.r < min_r:
            min_r = item.r
    for item in items:
        item.r = (item.r - min_r) / (max_r - min_r + 10 ** (-32))
    R = {r.r for r in items}
    for k in R:
        group = [item for item in items if item.r == k]
        group.sort(key=lambda z: z.d, reverse=True)
        for i, value in enumerate(group):
            value.theta = (2 * pi * (i + 1)) / len(group)
    thk = max_r / total_group
    for g_value in range(total_group):
        group = [i for i in items if (g_value * thk) <= i.r <= ((g_value + 1) * thk)]
        group.sort(key=lambda x: x.theta)
        if len(group) > 0:
            left = group[0]
            right = group[len(group) - 1]
            for i in group:
                if i.theta <= pi:
                    lefts.append(i)
                else:
                    rights.append(i)
    return Node(None, None, None, None, None, None, lefts), left, Node(None, None, None, None, None, None, rights), right
#                                              _  
#          _   ._   _|_  o  ._ _   o  _    _    ) 
#         (_)  |_)   |_  |  | | |  |  /_  (/_  /_ 
#              |                                  

  def branch(self, rows=None, stop=None, rest=None, evals=1, before=None):
    rows = rows or self.rows
    stop = stop or 2*len(rows)**the.Min
    rest = rest or []
    if len(rows) > stop:
      lefts,rights,left  = self.half(rows, True, before)
      return self.branch(lefts, stop, rest+rights, evals+1, left)
    else:
      return rows,rest,evals,before
    
  def rtree(self, items=None, stop=None):

    stop = stop or 2*len(items)**the.Min
    if len(items) > stop:
      leftN, leftR, rightN, rightR = self.halfBin(items)
      leftN.left, leftN.lefts, leftN.leftR, leftN.right, leftN.rights, leftN.rightR = self.rtree(items = leftN.all, stop = stop)
      rightN.left, rightN.lefts, rightN.leftR, rightN.right, rightN.rights, rightN.rightR = self.rtree(items = rightN.all,stop =  stop)
      return leftN, leftN.all, leftR,  rightN, rightN.all, rightR
    return None, None, None, None, None, None
  
  def tree(self):
    # convert the rows into a binary list of rows
    binRows = self.binXRows()

    # create a list of items from the binary rows
    items = [Item(row) for row in binRows]
    left, lefts, leftR, right, rights, rightR, = self.rtree(items = items)
    root = Node(left, leftR, right, rightR, lefts, rights, items)

    return root

                                               
#          _|  o   _   _  ._   _   _|_  o  _    _  
#         (_|  |  _>  (_  |   (/_   |_  |  /_  (/_                                                
# 
# class RANGE(struct):
#   def __init__(self,txt="",at=0,lo=None,hi=None,ys=None):
#     self.txt, self.at = txt,at
#     self.lo = lo
#     self.hi = hi or lo
#     self.ys = ys  
#     self.scored = 0
# 
#   def add(self,x,y): 
#     self.lo  = min(self.lo, x) 
#     self.hi  = max(self.hi, x)
#     self.ys.add(y)
# 
#   def merge(i,j,small):
#     k = i + j 
#     ni,nj = sum(i.ys.values()), sum(j.ys.values())
#     if ni <= small: return k
#     if nj <= small: return k
#     if k.ys.entropy() <= (ni*i.ys.entropy() + nj*j.ys.entropy())/(ni+nj): return k
# 
#   def __add__(i,j):
#     return RANGE(txt=i.txt, at=i.at, lo=i.lo, hi=j.hi, ys=i.ys + j.ys)
#   
#   def __repr__(self): 
#     lo,hi,s = self.lo, self.hi,self.txt
#     if lo == -sys.maxsize: return f"{s} <  {hi}" 
#     if hi ==  sys.maxsize: return f"{s} >= {lo}" 
#     if lo ==  hi         : return f"{s} == {lo}" 
#     return f"{lo} <= {s} < {hi}" 
# 
# def discretize(c,txt,col,rowss):
#   def bin(col,x): 
#     if isa(col,SYM): return x
#     tmp = (col.hi - col.lo)/(the.ranges - 1)
#     out = col.hi==col.lo and 0 or int(.5 + (x-col.lo)/tmp)  
#     return out
#   #--------------------------
#   def fill(bins):
#     if bins==[]: return bins
#     for i in range(1,len(bins)): bins[i].lo = bins[i-1].hi  
#     bins[0].lo  = -sys.maxsize
#     bins[-1].hi =  sys.maxsize
#     return bins  
#   #------------
#   bins, n = {}, 0
#   for y,rows in rowss.items(): 
#     for row in rows:
#       x = row[c]
#       if x != "?": 
#         n += 1
#         b = bin(col,x) 
#         bins[b] = bins[b] if b in bins else RANGE(txt=txt, at=c,lo=x, hi=x, ys=SYM()) 
#         bins[b].add(x, y) 
#   bins = sorted(bins.values(), key=lambda bin:bin.lo)  
#   return bins if isa(col,SYM) else fill(merges(bins, lambda a,b: a.merge(b, n/the.ranges)))
#                                       
# #          _       ._   |   _.  o  ._  
# #         (/_  ><  |_)  |  (_|  |  | | 
# #                  |                   
# 
# class RULE(struct):
#   def __init__(self,ranges):
#     self.parts, self.scored = {},0
#     for range in ranges:
#       self.parts[range.txt] = self.parts.get(range.txt,[])
#       self.parts[range.txt] += [range]
# 
#   def _or(self,ranges,row):
#     x =  row[ranges[0].at]
#     if x== "?": return True
#     for range in ranges:
#       return range.lo==range.hi==x or range.lo <= x < range.hi 
#   
#   def _and(self,row):
#     for ranges in self.parts.values():
#       if not self._or(ranges,row): return False
#     return True
#   
#   def selects(self,rows):
#     return [row for row in rows if self._and(row)]
#   
#   def selectss(self,rowss): 
#     return [row for _,rows in rowss.items() for row in self.selects(rows)]
#   
#   def __repr__(self):
#     def merge(a,b):
#       if a.txt == b.txt and a.hi ==b.lo: return a + b
#     return ' and '.join(' or '.join(merges(sorted(ors,key=lambda r:r.lo),merge))
#                         for ors in self.parts.values())
#   
#   class RULES(struct):
#     def score(range,goal,LIKE,HATE):
#       like,hate=0,0
#       for klass,n in range.ys.items():
#         if klass==goal: like += n
#         else:           hate += n
#       like,hate = like/(LIKE + tiny), hate/(HATE + tiny) 
#       range.scored = 0 if hate>like else  like ** the.Score / (like + hate)
# 
#     def __init__(self,ranges,goal,rowss,scoring):
#       def count(what=None):
#         return sum(len(rows) for klass,rows in rowss.items() if klass==what)
#       fun= return lambda lst: scoring(lst,self.goal,like,hate)
#       like, hate = count(goal), count()
#       [fun(range,self.goal, like, hate) for range in ranges]
#       self.ordered = self.top(self.test(self.top(ranges)))
#     
#     def test(self,ranges):
#       RULE(subset) for subset in powerset(ranges) if len(subset) > 0 
# 
#     def top(self,lst):
#       tmp = sorted(lst,key=lambda z:z.scored)
#       return [x for x in tmp if x.scored > tmp[0].scored * the.adequate][the.ample]

class Node(struct):
  def __init__(self, left, leftR, right, rightR, lefts, rights, all):
    self.left = left
    self.leftR = leftR
    self.right = right
    self.rightR = rightR
    self.lefts = lefts
    self.rights = rights
    self.all = all
  def __repr__(self):
    return f"Node(children={len(self.all)}\nleft={self.left}\nright={self.right})"


class Item(struct):
  def __init__(self, item):
    self.item = item[1]
    self.ev = []
    self.r = -1
    self.d = -1
    self.theta = -1
    self.score = 0
    self.pos = item[0]
    self.features = sum(self.item)

