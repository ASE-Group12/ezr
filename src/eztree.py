# ezr.py : tiny ai teaching lab. sequential model optimization (using not-so-naive bayes)
# (c)2024, Tim Menzies, BSD2 license. Share and enjoy.
import random,math,ast,sys,re
from fileinput import FileInput as file_or_stdin
from collections import Counter

config = dict(  beam = .75,
                Bins = 16,
                commence=4,
                Cease=20,
                enough=0.5,
                file="../data/auto93.csv",
                k=1,
                m=2,
                seed=1234567891,
                todo="the")
#----------------------------------------------------------------------------------------
class OBJ:
  def __init__(i,**d): i.__dict__.update(d)
  def __repr__(i): return i.__class__.__name__+'{'+show(i.__dict__)+'}'

the  = OBJ(**config)
big  = 1E30
tiny = 1/big
isa  = isinstance
r    = random.random

def adds(x,lst=None): [x.add(y) for y in lst or []]; return x

def cli(d):
  for k,v in d.items():
    for c,arg in enumerate(sys.argv):
      after = "" if c >= len(sys.argv) - 1 else sys.argv[c+1]
      if arg in ["-"+k[0], "--"+k]:
        v = str(v)
        v = "False" if v==True else ("True" if v==False else after)
        d[k] = coerce(v)
  return d

def coerce(s):
  try: return ast.literal_eval(s)
  except Exception: return s

def csv(file=None):
  with file_or_stdin(file) as src:
    for line in src:
      line = re.sub(r'([\n\t\r"\’ ]|#.*)', '', line)
      if line: yield [coerce(s.strip()) for s in line.split(",")]

def show(x,n=2):
  if isa(x,(int,float)) : return x if int(x)==x else round(x,n)
  if isa(x,(list,tuple)): return [show(y,n) for y in x][:10]
  if isa(x,dict): 
    return ' '.join(f":{k} {show(v,n)}" for k,v in x.items() if k[0]!="_")
  return x
#----------------------------------------------------------------------------------------
class RANGE(obj):
  def __init__(i,at,lo):
    i.at, i.lo, i.hi, i.has = at,lo,lo,{} 

  def add(i,x,y):
    i.lo = min(x,i.lo)
    i.hi = max(x,i.hi)
    i.has= i.has.get(y,0) + 1

class COL(OBJ):
  def __init__(i,at=0,txt=" "):
    i.n,i.at,i.txt = 0,at,txt
    i.heaven = 0 if txt[-1]=="-" else 1
  
  def score(i,d,goal,BEST,REST):
    rest=0
    for k,v in d.items():
      if k==goal: best = v
      else: rest += v
    best, rest = best/(BEST + tiny), rest/(REST + tiny)
    return best**2/(rest + tiny)
  
class SYM(COL):
  def __init__(i,**d)  : super().__init__(**d); i.has={}
  def add(i,x)         : i.n += 1; i.has[x] = 1 + i.has.get(x,0)
  def bin1(i,x)         : return x 
  def like(i,x,m,prior): return (i.has.get(x, 0) + m*prior) / (i.n + m)
  def mid(i)           : return max(i.has, key=i.has.get)
  def div(i):
    return -sum(n/i.n * math.log(n/i.n,2) for n in i.has.values() if n > 0) 
  
  # yrows = {{y,row}..}
  def cuts(i,j):
    a = 1/(2*i.div()**2) - 1/(2*j.div()**2)
    b = j.mu/(j.div()**2) - i.mu/(i.div()**2)
    c = i.mu**2 /(2*i.div()**2) - j.mu**2 / (2*j.div()**2) - math.log(j.div()/i.div())
    r1 = b**2 - 4*
    (b**2 - 4*a*c)**.5    return np.roots([a,b,c])
(b**2 - 4*a*c)**.5  

  yrows,goal,BEST,REST):
    lhs, rhs = Counter(), Counter() 
    def X(row)  (b**2 - 4*a*c)**.5   : return row[i.at]
    def ORDER(yrow): return big if X(yrow[1]) == "?" else X(yrow[1])
    yrows = sorted(yrows, key = ORDER)
    for y,_ in yrows: rhs[y] += 1
    hi = 0
    for j,(y,row) in enumerate(yrows):
      if j > 0  and X(row) != "?":
        rhs[y] -= 1
        lhs[y] += 1
        if x != X(yrows[j-1][1]):
          s1 = i.score(lhs,goal,BEST,REST)
          s2 = i.score(rhs,goal,BEST,REST)
          if s1 > hi: hi,val,op = s1,x,lt 
          if s2 > hi: hi,val,op = s2,x,ge 
    return OBJ(fun = lambda row: op(X(row),val),
               show= f"{i.txt} <  {val}" if op==lt else f"{i.txt} >= {val}", 
               yes = yrows[:j], 
               no  = yrows[j:])
  
  def cuts(i,yrows,goal,BEST,REST): 
    def X(yrow): return yrow[1][i.at] 
    all={}
    for yrow in rows:
      x = X(yrow))
      all[x] = all[x] if X(row) in all else Counter()
      all[x][y] += 1
    _,val = max((i.score(counter,goal,BEST,REST),x) for x,counter in all.items())
    return OBJ(fun = lambda row: X(row) == val
               show= f"{i.txt} == {val}"
               yes = [yrow for yrow in yrows if X(yrow[1])==val], 
               no  = [yrow for yrow in yrows if X(yrow[1])!=val])
  
def lt(x,y): return x <  y 
def ge(x,y): return x >= y 

class NUM(COL):
  def __init__(i,**d): super().__init__(**d); i.lo,i.hi = big, -big
  def add(i,x): 
    if x != "?": i.lo, i.hi = min(i.lo,x), max(i.hi,x) 

  def bin1(i,x): 
    return max(the.Bins - 1, int((x - i.lo)/((i.hi - i.lo)/the.Bins)))
  
  def norm(i,n): return n=="?" and n or (n - i.lo) / (i.hi - i.lo + tiny)

  def cuts(i,ranges,goal,BEST,REST):
    def merge(ranges)L
      d=Counter()
      for range in ds: d += Counter(range.has)
      return score(d),d
    
    for j,range in enumerate(ranges):
      if j> 1:
        s1,left = merge(ranges[:j])
        right = merge(ranges[j:])
              
      in range(lo,hi):

  

  def add(i,n):
    i.n += 1
    i.lo = min(n,i.lo)
    i.hi = max(n,i.hi)
    delta = n - i.mu
    i.mu += delta / i.n
    i.m2 += delta * (n -  i.mu)

  def like(i,n,*_):
    v     = i.div()**2 + tiny
    nom   = math.e**(-1*(n - i.mid())**2/(2*v)) + tiny
    denom = (2*math.pi*v)**.5  
    return min(1, nom/(denom + tiny))   
#----------------------------------------------------------------------------------------
class COLS(OBJ):
  def __init__(i,names):
    i.x,i.y,i.all,i.names,i.klass = [],[],[],names,None
    for at,txt in enumerate(names):
      a,z = txt[0], txt[-1]
      col = (NUM if a.isupper() else SYM)(at=at,txt=txt)
      i.all.append(col)
      if z != "X":
        (i.y if z in "!+-" else i.x).append(col)
        if z == "!": i.klass= col

  def add(i,lst): 
    [col.add(lst[col.at]) for col in i.all if lst[col.at] != "?"]; return lst

class DATA(OBJ):
  def __init__(i,src=[],fun=None,ordered=False):
    i.rows, i.cols = [],[]
    [i.add(lst,fun) for lst in src]
    if ordered: i.ordered()

  def add(i,lst,fun=None):
    if i.cols:
      if fun: fun(i,lst)
      i.rows += [i.cols.add(lst)]
    else: i.cols = COLS(lst)

  def clone(i,lst=None,ordered=False): 
    tmp = adds(DATA([i.cols.names]), lst)
    if ordered: tmp.ordered()
    return tmp

  def d2h(i,row):
    d,n = 0,0
    for col in i.cols.y:
      d += abs(col.norm(row[col.at]) - col.heaven)**2
      n += 1
    return (d/n)**.5
  
  def loglike(i, lst, nall, nh, m,k):
    prior = (len(i.rows) + k) / (nall + k*nh)
    likes = [c.like(lst[c.at],m,prior) for c in i.cols.x if lst[c.at] != "?"]
    return sum(math.log(x) for x in likes + [prior] if x>0)

  def ordered(i): i.rows.sort(key=i.d2h); return i.rows

  def smo(i, score=lambda B,R: B - R ):
    def like(row,data): 
      return data.loglike(row,len(data.rows),2,the.m,the.m)
    def acquire(best, rest, rows): 
      chop=int(len(rows) * the.beam)
      return sorted(rows, key=lambda r: -score(like(r,best),like(r,rest)))[:chop]
    #---------------------
    random.shuffle(i.rows)
    done, todo = i.rows[:the.commence], i.rows[the.commence:]
    data1 = i.clone(done, ordered=True)  
    evals = 0
    for _ in range(the.Cease - the.commence):
      n = int(len(done)**the.enough + .5)
      top,*todo = acquire(i.clone(data1.rows[:n]),  
                          i.clone(data1.rows[n:]),
                          todo) 
      done.append(top)
      evals += 1
      data1 = i.clone(done, ordered=True)
      if len(todo) < 3: break
    return data1.rows[0],evals

class NB(OBJ):
  def __init__(i): i.correct,i.nall,i.datas = 0,0,{}

  def loglike(i,data,lst):
    return data.loglike(lst, i.nall, len(i.datas), the.m, the.k)

  def run(i,data,lst):
    klass = lst[data.cols.klass.at]
    i.nall += 1
    if i.nall > 10:
      guess = max((i.loglike(data,lst),klass1) for klass1,data in i.datas.items())
      i.correct += klass == guess[1] 
    if klass not in i.datas: i.datas[klass] =  data.clone()
    i.datas[klass].add(lst)

  def report(i): return OBJ(accuracy = i.correct / i.nall)
#----------------------------------------------------------------------------------------
class eg:
  def unknown(): print(f"W> unknown action [{the.todo}].")
  
  def the():  print(the)

  def sym():
    s = adds(SYM(),"aaaabbc")
    assert round(s.div(),2) == 1.38 and s.mid() == "a" 

  def one():
    w = OBJ(n=0)
    def inc(_,r): w.n += len(r)
    d = DATA(csv("../data/auto93.csv"), inc) 
    assert w.n == 3184

  def clone(): 
    d = DATA(csv(the.file))
    c =d.clone()
    print(d.cols.all[1])
    print(c.cols.all[1])

  def ordered():
    d = DATA(csv(the.file),ordered=True)
    for j,row in enumerate(d.rows):
      if j%50==0: print(j,row)

  def nb():
    out=[]
    for k in [1,2,3]:
      for m in [1,2,3]: 
        the.k, the.m = k,m
        nb = NB()
        DATA(csv("../data/soybean.csv"), nb.run)
        out += [OBJ(acc = nb.report().accuracy, k=k, m=m)]
    [print(show(x,3)) for x in sorted(out,key=lambda z: z.acc)]

  def smo():
    d=DATA(csv(the.file),ordered=True)
    b4   = d.rows[len(d.rows)//2]
    for _ in range(30):
      sys.stderr.write('.');  sys.stderr.flush()
      after,evals= d.smo()
      print("\n",show(dict(mid= d.d2h(b4),smo= d.d2h(after), evals=evals)),end="")
#----------------------------------------------------------------------------------------
if __name__=="__main__":
  the = OBJ(**cli(config))
  random.seed(the.seed)
  getattr(eg, the.todo, eg.unknown)()
