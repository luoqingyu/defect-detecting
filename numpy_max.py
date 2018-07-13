import numpy  as np
a = np.array([[1,2,3],[4,5,1]])
max_scores=np.max(a[0,:])
print(np.where(max_scores))