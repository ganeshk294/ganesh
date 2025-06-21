import numpy as np
 np.array([2,4,56,422,32,1]) # 1D array
 Out[2]: array([  2,   4,  56, 422,  32,   1])
 In [3]:
 In [4]:
 a = np.array([2,4,56,422,32,1])  #Vector
 print(a)                       
[  2   4  56 422  32   1]
 type(a)
 Out[4]: numpy.ndarray
 1/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [5]:
 In [6]:
 # 2D Array ( Matrix)
 new = np.array([[45,34,22,2],[24,55,3,22]])
 print(new)
 [[45 34 22  2]
 [24 55  3 22]]
 # 3 D ----  # Tensor
 np.array ( [[2,3,33,4,45],[23,45,56,66,2],[357,523,32,24,2],[32,32,44,33,234]]
 Out[6]: array([[  2,   3,  33,   4,  45],
 [ 23,  45,  56,  66,   2],
 [357, 523,  32,  24,   2],
 [ 32,  32,  44,  33, 234]]
##dtype
 :
 np.array([11,23,44] , dtype =float)
 Out[7]: array([11., 23., 44.])
 In [8]:
 np.array([11,23,44] , dtype =bool) # Here True becoz , python treats Non -zero
 Out[8]: array([ True,  True,  True])
 In [9]:
 np.array([11,23,44] , dtype =complex)
 Out[9]: array([11.+0.j, 23.+0.j, 44.+0.j])

