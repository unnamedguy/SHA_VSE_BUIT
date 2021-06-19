# !gdown https://drive.google.com/uc?id=1T49iBpXvdnXaKbWrLqnJB8tniYrdRcLM&export=download

# !gdown https://drive.google.com/uc?id=1zBa0k7L_CzGSZHa4-WCBad62v1HblJxY

import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')
test = pd.read_csv("test.csv")
print(test)
print(test.describe())

print(data)
print(data.describe())

#длительности

data.groupby('id').max('time')

#количество экземпляров без аномальных показателей

print(data.groupby('id').agg({'y':sum}))

class DecisionTree():
    def __init__(self, x, y, idxs = None, min_leaf=2):
        if idxs is None: idxs=np.arange(len(y))
        self.x,self.y,self.idxs,self.min_leaf = x,y,idxs,min_leaf
        self.n,self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()
        
    def find_varsplit(self):
        for i in range(self.c): self.find_better_split(i)
        if self.score == float('inf'): return
        x = self.split_col
        lhs = np.nonzero(x<=self.split)[0]
        rhs = np.nonzero(x>self.split)[0]
        self.lhs = DecisionTree(self.x, self.y, self.idxs[lhs])
        self.rhs = DecisionTree(self.x, self.y, self.idxs[rhs])

    def find_better_split(self, var_idx):
        x,y = self.x.values[self.idxs,var_idx], self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y,sort_x = y[sort_idx], x[sort_idx]
        rhs_cnt,rhs_sum,rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
        lhs_cnt,lhs_sum,lhs_sum2 = 0,0.,0.

        for i in range(0,self.n-self.min_leaf-1):
            xi,yi = sort_x[i],sort_y[i]
            lhs_cnt += 1; rhs_cnt -= 1
            lhs_sum += yi; rhs_sum -= yi
            lhs_sum2 += yi**2; rhs_sum2 -= yi**2
            if i<self.min_leaf or xi==sort_x[i+1]:
                continue

            lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_std*lhs_cnt + rhs_std*rhs_cnt
            if curr_score<self.score: 
                self.var_idx,self.score,self.split = var_idx,curr_score,xi

    @property
    def split_name(self): return self.x.columns[self.var_idx]
    
    @property
    def split_col(self): return self.x.values[self.idxs,self.var_idx]

    @property
    def is_leaf(self): return self.score == float('inf')
    
    def __repr__(self):
        s = f'n: {self.n}; val:{self.val}'
        if not self.is_leaf:
            s += f'; score:{self.score}; split:{self.split}; var:{self.split_name}'
        return s

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        t = self.lhs if xi[self.var_idx]<=self.split else self.rhs
        return t.predict_row(xi)
import math
def std_agg(cnt, s1, s2): return math.sqrt((s2/cnt) - (s1/cnt)**2)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pywt
 
i   = 0
sum = 0
t   = data.id[0]

pat = 1
rb  = 1
 
J = 1

maxs2d  = []
maxs_   = []
mins_   = []

ys    = []
xs    = []
times = []


while rb < data.id[len(data.id)-1]:
  y     = []
  x     = []
  time  = []
  sum   = 0

  Max = 0
  Min = 10000

  maxs = []
  while data.id[i] <= rb:
    # if (data.id[i] != pat):
    #   i += 1
    #   continue
    if (data.id[i] > t):
      t = data.id[i]
      # if (len(time)>0):
      #   sum = time[len(time)-1]
    y.append(data.y[i])
    if (data.y[i] and data.x[i] > Max):
      Max = data.x[i]
    if (data.y[i] and data.x[i] < Min):
      Min = data.x[i]
    if (data.y[i] == 0 and Max != 0 and Min != 10000):
      maxs.append(Max - Min)
      maxs_.append(Max)
      mins_.append(Min)
      Max = 0
      Min = 10000
    x.append(data.x[i])
    time.append(data.time[i] + sum)
    i += 1
  
  user_id = data.id[i-1]
  rb += 1
  J += 1
  if (len(y) > 0):
    ys.append(y)
    xs.append(x)
    times.append(time)
  #   maxs2d.append(maxs)
  #   plt.figure(J, figsize=(30, 15))
    
  #   y = np.array(y)
  #   x = np.array(x)
  #   time = np.array(time)

  # #   nrows = 2
  #   plt.subplot(nrows, 1, 1)
  #   plt.plot(time, x, '-')
  #   plt.grid()
  #   plt.fill_between(time, x, 0, where=(y == 1), facecolor='r', alpha=0.6)
  #   plt.title("Patient "+str(user_id))
    
  #   lvl = 1
    
  #   cfs, cfs1 = pywt.cwt(x, [1], "gaus1")
  #   # [0.5 0.5]
  #   # [[-1. -1.]
  #   # [-1. -1.]
  #   # [ 1.  1.]
  #   # [ 1.  1.]]
  #   # print(scaler.transform([[2, 2]]))
  #   # [[3. 3.]]
    
  #   plt.subplot(nrows, 1, 2)
  #   plt.plot(cfs[0],'b',linewidth=2, label='cA,level-'+str(lvl))
  #   plt.legend(loc='best')
  #   plt.grid()
  #   # plt.subplot(nrows, 1, 3)
  #   # plt.plot(cfs1,'r',linewidth=2, label='cD,level-'+str(lvl))
  #   # plt.legend(loc='best')
  #   # # print(cA)
  #   # # print(cD)
  #   # plt.grid()
    
    plt.show()

  #   break



MAX = max(maxs_)
MIN = min(mins_)

print("MAX:", MAX)
print("MIN:", MIN)

data1 = data.drop(data[data.x < MIN].index)
data1 = data.drop(data[data.x > MAX].index)

max_id = data.id.max()
print("max_id:", max_id)

for i in range(1, max_id+1):
  datat = data[data['id'] == i]
  dataA = datat
  # print(dataA)
  # break
  
  if (len(datat) <= 0):
    continue
  
  plt.figure(J, figsize=(25, 12))
  y = datat.y.to_numpy()
  x = datat.x.to_numpy()
  time = datat.time.to_numpy()

  nrows = 2
  # plt.subplot(nrows, 1, 1)

  # plt.subplot(nrows, 1, 1)
  # plt.plot(time, x, '-')
  # plt.grid()
  # plt.fill_between(time, x, 0, where=(y == 1), facecolor='r', alpha=0.6)
  # plt.title("Patient "+str(i))

  # lvl = 1
  n   = 1
  df  = np.zeros(n)
  df  = np.concatenate([df, np.diff(x, n=n)])

  dtime = np.zeros(n)
  dtime = np.concatenate([dtime, np.diff(time, n=n)])

  dataA["df"] = df
  dataA["dtime"] = dtime
  dataA = dataA[dataA['y'] == 1]
  print(dataA)
  print(dataA.describe())
  nrows = 2
  # cfs, cfs1 = pywt.cwt(df, [1], "gaus1")

  # plt.subplot(nrows, 1, 2)
  # plt.plot(time, cfs[0],'b',linewidth=2, label='dif')
  # plt.fill_between(time, cfs[0], -200, where=(y == 1), facecolor='r', alpha=0.6)
  # plt.legend(loc='best')
  # plt.grid()
  # plt.subplot(nrows, 1, 3)
  # plt.plot(cfs1,'r',linewidth=2, label='cD,level-'+str(lvl))
  # plt.legend(loc='best')
  # # print(cA)
  # # print(cD)
  # plt.grid()

  # plt.show()


max_id = test.id.max()
print("max_id:", max_id)
for i in range(1, max_id+1):
  datat = test[test['id'] == i]
  dataA = datat
  # print(dataA)
  # break
  
  if (len(datat) <= 0):
    continue
  
  plt.figure(J, figsize=(25, 12))
  # y = datat.y.to_numpy()
  x = datat.x.to_numpy()
  time = datat.time.to_numpy()

  nrows = 2
  # plt.subplot(nrows, 1, 1)

  # plt.subplot(nrows, 1, 1)
  # plt.plot(time, x, '-')
  # plt.grid()
  # plt.fill_between(time, x, 0, where=(y == 1), facecolor='r', alpha=0.6)
  # plt.title("Patient "+str(i))

  # lvl = 1
  n   = 1
  df  = np.zeros(n)
  df  = np.concatenate([df, np.diff(x, n=n)])

  dtime = np.zeros(n)
  dtime = np.concatenate([dtime, np.diff(time, n=n)])

  dataA["df"] = df
  dataA["dtime"] = dtime
  # dtime > 680 && dtime < 830

  # dataA = dataA[dataA['y'] == 1]
  print(dataA)
  print(dataA.describe())
  nrows = 2
  # cfs, cfs1 = pywt.cwt(df, [1], "gaus1")

  # plt.subplot(nrows, 1, 2)
  # plt.plot(time, cfs[0],'b',linewidth=2, label='dif')
  # plt.fill_between(time, cfs[0], -200, where=(y == 1), facecolor='r', alpha=0.6)
  # plt.legend(loc='best')
  # plt.grid()
  # plt.subplot(nrows, 1, 3)
  # plt.plot(cfs1,'r',linewidth=2, label='cD,level-'+str(lvl))
  # plt.legend(loc='best')
  # # print(cA)
  # # print(cD)
  # plt.grid()

  # plt.show()