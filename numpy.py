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
np.arange(1,25)   
# 1-included , 25 - last one got excluded
 Out[10]: array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
 18, 19, 20, 21, 22, 23, 24])
 In [11]:
 np.arange(1,25,2) #strides ---> Alternate numbers 
Out[11]: array([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23])
 np.arange(1,11).reshape(5,2) # converted 5 rows and 2 columns
 Out[12]: array([[ 1,  2],
 [ 3,  4],
 [ 5,  6],
 [ 7,  8],
 [ 9, 10]])
 In [13]:
 np.arange(1,11).reshape(2,5) # converted 2 rows and 5 columns
 Out[13]: array([[ 1,  2,  3,  4,  5],
 [ 6,  7,  8,  9, 10]])
 In [14]:
 np.arange(1,13).reshape(3,4)  # converted 3 rows and 4 columns 
Out[14]: array([[ 1,  2,  3,  4],
 [ 5,  6,  7,  8],
 [ 9, 10, 11, 12]])
np.ones((3,4)) # we have to mention iside tuple
 Out[15]: array([[1., 1., 1., 1.],
 [1., 1., 1., 1.],
 [1., 1., 1., 1.]])
 3/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [16]:
 np.zeros((3,4))
 Out[16]: array([[0., 0., 0., 0.],
 [0., 0., 0., 0.],
 [0., 0., 0., 0.]])
 In [17]:
 # Another Type ---> random()
 np.random.random((4,3))
 Out[17]: array([[0.36101914, 0.04882035, 0.23266312],
 [0.74023073, 0.01298753, 0.03403761],
 [0.80722213, 0.55568178, 0.94063313],
 [0.45455407, 0.06724469, 0.75013537]])
 linspace
 It is also called as Linearly space , Linearly separable,in a given range at equal distance it
 creates points.
 In [18]:
 np.linspace(-10,10,10) # here: lower range,upper range ,number of items to gen
 Out[18]: array([-10.        ,  -7.77777778,  -5.55555556,  -3.33333333,-1.11111111,   1.11111111,   3.33333333,   5.55555556,
 7.77777778,  10.        ])
 In [19]:
 np.linspace(-2,12,6) 
Out[19]: array([-2. ,  0.8,  3.6,  6.4,  9.2, 12. ])
 identity
 indentity matrix is that diagonal items will be ones and evrything will be zeros
 In [20]:
 # creating the indentity matrix
 np.identity(3)
 Out[20]: array([[1., 0., 0.],
 [0., 1., 0.],
 [0., 0., 1.]])
 4/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [21]:
 np.identity(6)
 Out[21]: array([[1., 0., 0., 0., 0., 0.],
 [0., 1., 0., 0., 0., 0.],
 [0., 0., 1., 0., 0., 0.],
 [0., 0., 0., 1., 0., 0.],
 [0., 0., 0., 0., 1., 0.],
 [0., 0., 0., 0., 0., 1.]])
 Array Attributes
 In [22]:
 a1 = np.arange(10) # 1D
 a1
 Out[22]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
 In [23]:
 a2 =np.arange(12, dtype =float).reshape(3,4) # Matrix
 a2
 Out[23]: array([[ 0.,  1.,  2.,  3.],
 [ 4.,  5.,  6.,  7.],
 [ 8.,  9., 10., 11.]])
 In [24]:
 a3 = np.arange(8).reshape(2,2,2) # 3D --> Tensor
 a3
 Out[24]: array([[[0, 1],
 [2, 3]],
 [[4, 5],
 [6, 7]]])
 ndim
 To findout given arrays number of dimensions
 In [25]:
 a1.ndim
 Out[25]: 1
 In [26]:
 a2.ndim
 Out[26]: 2
 In [27]:
 a3.ndim
 Out[27]: 3
 5/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 shape
 gives each item consist of no.of rows and np.of column
 In [28]:
 a1.shape # 1D array has 10 Items
 Out[28]: (10,)
 In [29]:
 a2.shape # 3 rows and 4 columns
 Out[29]: (3, 4)
 In [30]:
 a3.shape  # first ,2 says it consists of 2D arrays .2,2 gives no.of rows and c
 Out[30]: (2, 2, 2)
 size
 gives number of items
 In [31]:
 a3
 Out[31]: array([[[0, 1],
 [2, 3]],
 [[4, 5],
 [6, 7]]])
 In [32]:
 a3.size # it has 8 items . like shape :2,2,2 = 8
 Out[32]: 8
 In [33]:
 a2
 Out[33]: array([[ 0.,  1.,  2.,  3.],
 [ 4.,  5.,  6.,  7.],
 [ 8.,  9., 10., 11.]])
 In [34]:
 a2.size
 Out[34]: 12
 item size
 Memory occupied by the item
 6/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [35]:
 a1
 Out[35]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
 In [36]:
 a1.itemsize # bytes
 Out[36]: 4
 In [37]:
 a2.itemsize # integer 64 gives = 8 bytes
 Out[37]: 8
 In [38]:
 a3.itemsize  # integer 32 gives = 4 bytes
 Out[38]: 4
 dtype
 gives data type of the item
 In [39]:
 In [40]:
 print(a1.dtype)
 print(a2.dtype)
 print(a3.dtype)
 int32
 float64
 int32
 Changing Data Type
 #astype
 x = np.array([33, 22, 2.5])
 x
 Out[40]: array([33. , 22. ,  2.5])
 In [41]:
 x.astype(int)
 Out[41]: array([33, 22,  2])
 Array operations
 In [42]:
 z1 = np.arange(12).reshape(3,4)
 z2 = np.arange(12,24).reshape(3,4)
 7/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [43]:
 z1
 Out[43]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 In [44]:
 z2
 Out[44]: array([[12, 13, 14, 15],
 [16, 17, 18, 19],
 [20, 21, 22, 23]])
 scalar operations
 Scalar operations on Numpy arrays include performing addition or subtraction, or multiplication
 on each element of a Numpy array.
 In [45]:
 # arithmetic
 z1 + 2
 Out[45]: array([[ 2,  3,  4,  5],
 [ 6,  7,  8,  9],
 [10, 11, 12, 13]])
 In [46]:
 # Subtraction
 z1 - 2
 Out[46]: array([[-2, -1,  0,  1],
 [ 2,  3,  4,  5],
 [ 6,  7,  8,  9]])
 In [47]:
 # Multiplication
 z1 * 2
 Out[47]: array([[ 0,  2,  4,  6],
 [ 8, 10, 12, 14],
 [16, 18, 20, 22]])
 In [48]:
 # power
 z1 ** 2
 Out[48]: array([[  0,   1,   4,   9],
 [ 16,  25,  36,  49],
 [ 64,  81, 100, 121]], dtype=int32)
 In [49]:
 ## Modulo
 z1 % 2
 Out[49]: array([[0, 1, 0, 1],
 [0, 1, 0, 1],
 [0, 1, 0, 1]], dtype=int32)
 8/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 relational Operators
 The relational operators are also known as comparison operators, their main function is to
 return either a true or false based on the value of operands.
 In [50]:
 z2
 Out[50]: array([[12, 13, 14, 15],
 [16, 17, 18, 19],
 [20, 21, 22, 23]])
 In [51]:
 z2 > 2   # if 2 is greater than evrythig gives True
 Out[51]: array([[ True,  True,  True,  True],
 [ True,  True,  True,  True],
 [ True,  True,  True,  True]])
 In [52]:
 z2 > 20 
Out[52]: array([[False, False, False, False],
 [False, False, False, False],
 [False,  True,  True,  True]])
 Vector Operation
 We can apply on both numpy array
 In [53]:
 z1
 Out[53]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 In [54]:
 z2
 Out[54]: array([[12, 13, 14, 15],
 [16, 17, 18, 19],
 [20, 21, 22, 23]])
 In [55]:
 # Arthemetic 
z1 + z2  # both numpy array Shape is same , we can add item wise
 Out[55]: array([[12, 14, 16, 18],
 [20, 22, 24, 26],
 [28, 30, 32, 34]])
 9/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [56]:
 z1 * z2
 Out[56]: array([[  0,  13,  28,  45],
 [ 64,  85, 108, 133],
 [160, 189, 220, 253]])
 In [57]:
 z1 - z2
 Out[57]: array([[-12, -12, -12, -12],
 [-12, -12, -12, -12],
 [-12, -12, -12, -12]])
 In [58]:
 z1 / z2
 Out[58]: array([[0.        , 0.07692308, 0.14285714, 0.2       ],
 [0.25      , 0.29411765, 0.33333333, 0.36842105],
 [0.4       , 0.42857143, 0.45454545, 0.47826087]])
 Array Functions
 In [59]:
 k1 = np.random.random((3,3))
 k1 = np.round(k1*100)
 k1
 Out[59]: array([[44., 98., 47.],
 [56., 49., 30.],
 [60., 54., 24.]])
 In [60]:
 # Max
 np.max(k1)
 Out[60]: 98.0
 In [61]:
 # min
 np.min(k1)
 Out[61]: 24.0
 In [62]:
 # sum
 np.sum(k1)
 Out[62]: 462.0
 In [63]:
 # prod ----> Multiplication
 np.prod(k1)
 Out[63]: 1297293445324800.0
 In Numpy
 10/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 0 = column , 1 = row
 In [64]:
 # if we want maximum of every row
 np.max(k1, axis = 1)
 Out[64]: array([98., 56., 60.])
 In [65]:
 # maximum of every column
 np.max(k1, axis = 0)
 Out[65]: array([60., 98., 47.])
 In [66]:
 # product of every column
 np.prod(k1, axis = 0)
 Out[66]: array([147840., 259308.,  33840.])
 Statistics related fuctions
 In [67]:
 # mean
 k1
 Out[67]: array([[44., 98., 47.],
 [56., 49., 30.],
 [60., 54., 24.]])
 In [68]:
 np.mean(k1)
 Out[68]: 51.333333333333336
 In [69]:
 # mean of every column
 k1.mean(axis=0)
 Out[69]: array([53.33333333, 67.        , 33.66666667])
 In [70]:
 # median
 np.median(k1)
 Out[70]: 49.0
 In [71]:
 np.median(k1, axis = 1)
 Out[71]: array([47., 49., 54.])
 11/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [72]:
 # Standard deviation
 np.std(k1)
 Out[72]: 19.89416441516903
 In [73]:
 np.std(k1, axis =0)
 Out[73]: array([ 6.79869268, 22.0151463 ,  9.7410928 ])
 In [74]:
 # variance
 np.var(k1)
 Out[74]: 395.77777777777777
 Trignometry Functions
 In [75]:
 np.sin(k1) # sin 
Out[75]: array([[ 0.01770193, -0.57338187,  0.12357312],
 [-0.521551  , -0.95375265, -0.98803162],
 [-0.30481062, -0.55878905, -0.90557836]])
 In [76]:
 np.cos(k1)
 Out[76]: array([[ 0.99984331, -0.81928825, -0.99233547],
 [ 0.85322011,  0.30059254,  0.15425145],
 [-0.95241298, -0.82930983,  0.42417901]])
 In [77]:
 np.tan(k1)
 Out[77]: array([[ 0.0177047 ,  0.69985365, -0.12452757],
 [-0.61127369, -3.17290855, -6.4053312 ],
 [ 0.32004039,  0.6738001 , -2.1348967 ]])
 dot product
 The numpy module of Python provides a function to perform the dot product of two arrays.
 In [78]:
 In [79]:
 s2 = np.arange(12).reshape(3,4)
 s3 = np.arange(12,24).reshape(4,3)
 s2
 Out[79]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 12/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [80]:
 s3
 Out[80]: array([[12, 13, 14],
 [15, 16, 17],
 [18, 19, 20],
 [21, 22, 23]])
 In [81]:
 np.dot(s2,s3)  # dot product of s2 , s3
 Out[81]: array([[114, 120, 126],
 [378, 400, 422],
 [642, 680, 718]])
 Log and Exponents
 In [82]:
 np.exp(s2)
 Out[82]: array([[1.00000000e+00, 2.71828183e+00, 7.38905610e+00, 2.00855369e+01],
 [5.45981500e+01, 1.48413159e+02, 4.03428793e+02, 1.09663316e+03],
 [2.98095799e+03, 8.10308393e+03, 2.20264658e+04, 5.98741417e+04]])
 round / floor /ceil
 1. round
 The numpy.round() function rounds the elements of an array to the nearest integer or to the
 specified number of decimals.
 In [87]:
 In [88]:
 In [84]:
 # Round to the nearest integer
 arr = np.array([1.2, 2.7, 3.5, 4.9])
 rounded_arr = np.round(arr)
 print(rounded_arr) 
[1. 3. 4. 5.]
 # Round to two decimals
 arr = np.array([1.234, 2.567, 3.891])
 rounded_arr = np.round(arr, decimals=2)
 print(rounded_arr) 
[1.23 2.57 3.89]
 #randomly
 np.round(np.random.random((2,3))*100) 
Out[84]: array([[ 8., 36., 43.],
 [13., 90., 63.]])
 2. floor
 13/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 The numpy.floor() function returns the largest integer less than or equal to each element of an
 array.
 In [89]:
 In [85]:
 # Floor operation
 arr = np.array([1.2, 2.7, 3.5, 4.9])
 floored_arr = np.floor(arr)
 print(floored_arr)
 [1. 2. 3. 4.]
 np.floor(np.random.random((2,3))*100) # gives the smallest integer ex :6.8 = 
Out[85]: array([[58., 56., 89.],
 [10., 83., 34.]])
 3. Ceil
 The numpy.ceil() function returns the smallest integer greater than or equal to each element of
 an array.
 In [90]:
 In [86]:
 arr = np.array([1.2, 2.7, 3.5, 4.9])
 ceiled_arr = np.ceil(arr)
 print(ceiled_arr)
 [2. 3. 4. 5.]
 np.ceil(np.random.random((2,3))*100) # gives highest integer ex : 7.8 = 8
 Out[86]: array([[94.,  5., 46.],
 [84., 71., 41.]])
 Indexing and slicing
 In [91]:
 In [92]:
 p1 = np.arange(10)
 p2 = np.arange(12).reshape(3,4)
 p3 = np.arange(8).reshape(2,2,2)
 p1
 Out[92]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
 In [93]:
 p2
 Out[93]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 14/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [94]:
 p3
 Out[94]: array([[[0, 1],
 [2, 3]],
 [[4, 5],
 [6, 7]]])
 Indexing on 1D array
 In [95]:
 p1
 Out[95]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
 In [96]:
 # fetching last item
 p1[-1]
 Out[96]: 9
 In [97]:
 # fetchig first ietm
 p1[0]
 Out[97]: 0
 indexing on 2D array
 In [98]:
 p2
 Out[98]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 In [100]:
 # fetching desired element : 6
 p2[1,2] # here 1 = row(second) , 2= column(third) , becoz it starts from zero 
Out[100]: 6
 In [101]:
 # fetching desired element : 11
 p2[2,3] # row =2 , column =3
 Out[101]: 11
 15/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [102]:
 # fetching desired element : 4
 p2[1,0] # row =1 , column =0
 Out[102]: 4
 indexing on 3D ( Tensors)
 In [103]:
 p3
 Out[103]: array([[[0, 1],
 [2, 3]],
 [[4, 5],
 [6, 7]]])
 In [106]:
 # fetching desired element : 5
 p3[1,0,1]
 Out[106]: 5
 EXPLANATION :Here 3D is consists of 2 ,2D array , so Firstly we take 1 because our desired is
 5 is in second matrix which is 1 .and 1 row so 0 and second column so 1
 In [109]:
 # fetching desired element : 2
 p3[0,1,0]
 Out[109]: 2
 EXPLANATION :Here firstly we take 0 because our desired is 2, is in first matrix which is 0 .
 and 2 row so 1 and first column so 0
 In [110]:
 # fetching desired element : 0
 p3[0,0,0]
 Out[110]: 0
 Here first we take 0 because our desired is 0, is in first matrix which is 0 . and 1 row so 0 and
 first column so 0
 In [113]:
 # fetching desired element : 6
 p3[1,1,0]
 Out[113]: 6
 16/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 EXPLANATION : Here first we take because our desired is 6, is in second matrix which is 1 .
 and second row so 1 and first column so 0
 Slicing
 Fetching Multiple items
 Slicing on 1D
 In [114]:
 p1
 Out[114]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
 In [116]:
 # fetching desired elements are  : 2,3,4
 p1[2:5]
 Out[116]: array([2, 3, 4])
 EXPLANATION :Here First we take , whatever we need first item ,2 and up last(4) + 1 which 5
 .because last element is not included
 In [117]:
 # Alternate (same as python)
 p1[2:5:2]
 Out[117]: array([2, 4])
 Slicing on 2D
 In [121]:
 p2
 Out[121]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 In [122]:
 # fetching total First row
 p2[0, :]
 Out[122]: array([0, 1, 2, 3])
 EXPLANATION :Here 0 represents first row and (:) represnts Total column
 17/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [124]:
 # fetching total third column
 p2[:,2]
 Out[124]: array([ 2,  6, 10])
 EXPLANATION :Here we want all rows so (:) , and we want 3rd column so 2
 In [164]:
 # fetch 5,6 and 9,10
 p2
 Out[164]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 In [165]:
 p2[1:3] # for rows
 Out[165]: array([[ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 In [127]:
 p2[1:3 ,1:3]  # For columns
 Out[127]: array([[ 5,  6],
 [ 9, 10]])
 EXPLANATION :Here first [1:3] we slice 2 second row is to third row is not existed which is 2
 and Secondly , we take [1:3] which is same as first:we slice 2 second row is to third row is not
 included which is 3
 In [129]:
 # fetch 0,3 and 8,11
 p2
 Out[129]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 In [130]:
 p2[::2, ::3]
 Out[130]: array([[ 0,  3],
 [ 8, 11]])
 EXPLANATION : Here we take (:) because we want all rows , second(:2) for alternate value,
 and (:) for all columns and (:3) jump for two steps
 18/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [163]:
 # fetch 1,3 and 9,11
 p2  
Out[163]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 In [162]:
 p2[::2] # For rows
 Out[162]: array([[ 0,  1,  2,  3],
 [ 8,  9, 10, 11]])
 In [ ]:
 In [160]:
 p2[::2 ,1::2] # columns
 EXPLANATION : Here we take (:) because we want all rows , second(:2) for alternate value,
 and (1) for we want from second column and (:2) jump for two steps and ignore middle one
 # fetch only 4 ,7
 p2
 Out[160]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 In [161]:
 p2[1] # first rows
 Out[161]: array([4, 5, 6, 7])
 In [150]:
 p2[1,::3] # second columns
 Out[150]: array([4, 7])
 EXPLANATION : Here we take (1) because we want second row , second(:) for total column,
 (:3) jump for two steps and ignore middle ones
 In [157]:
 # fetch 1,2,3 and 5,6,7
 p2 
Out[157]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 In [159]:
 p2[0:2] # first fetched rows
 Out[159]: array([[0, 1, 2, 3],
 [4, 5, 6, 7]])
 19/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [156]:
 p2[0:2 ,1: ] # for column
 Out[156]: array([[1, 2, 3],
 [5, 6, 7]])
 In [166]:
 # fetch 1,3 and 5,7
 p2
 Out[166]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 In [167]:
 p2[0:2] # for rows
 Out[167]: array([[0, 1, 2, 3],
 [4, 5, 6, 7]])
 In [170]:
 p2[0:2 ,1::2]
 Out[170]: array([[1, 3],
 [5, 7]])
 EXPLANATION : 0:2 selects the rows from index 0 (inclusive) to index 2 (exclusive), which
 means it will select the first and second rows of the array. , is used to separate row and column
 selections. 1::2 selects the columns starting from index 1 and selects every second column. So
 it will select the second and fourth columns of the array.
 Slicing in 3D
 In [172]:
 p3 = np.arange(27).reshape(3,3,3)
 p3
 Out[172]: array([[[ 0,  1,  2],
 [ 3,  4,  5],
 [ 6,  7,  8]],
 [[ 9, 10, 11],
 [12, 13, 14],
 [15, 16, 17]],
 [[18, 19, 20],
 [21, 22, 23],
 [24, 25, 26]]])
 In [173]:
 # fetch second matrix
 p3[1]
 Out[173]: array([[ 9, 10, 11],
 [12, 13, 14],
 [15, 16, 17]])
 20/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [179]:
 # fetch first and last
 p3[::2]
 Out[179]: array([[[ 0,  1,  2],
 [ 3,  4,  5],
 [ 6,  7,  8]],
 [[18, 19, 20],
 [21, 22, 23],
 [24, 25, 26]]])
 EXPLANATION : Along the first axis, (::2) selects every second element. This means it will
 select the subarrays at indices 0 and 2
 In [180]:
 # Fetch 1 2d array's 2 row ---> 3,4,5
 p3
 Out[180]: array([[[ 0,  1,  2],
 [ 3,  4,  5],
 [ 6,  7,  8]],
 [[ 9, 10, 11],
 [12, 13, 14],
 [15, 16, 17]],
 [[18, 19, 20],
 [21, 22, 23],
 [24, 25, 26]]])
 In [185]:
 p3[0] # first numpy array
 Out[185]: array([[0, 1, 2],
 [3, 4, 5],
 [6, 7, 8]])
 In [186]:
 p3[0,1,:] 
Out[186]: array([3, 4, 5])
 EXPLANATION : 0 represnts first matrix , 1 represents second row , (:) means total
 21/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [187]:
 # Fetch 2 numpy array ,middle column ---> 10,13,16
 p3
 Out[187]: array([[[ 0,  1,  2],
 [ 3,  4,  5],
 [ 6,  7,  8]],
 [[ 9, 10, 11],
 [12, 13, 14],
 [15, 16, 17]],
 [[18, 19, 20],
 [21, 22, 23],
 [24, 25, 26]]])
 In [189]:
 p3[1] # middle Array
 Out[189]: array([[ 9, 10, 11],
 [12, 13, 14],
 [15, 16, 17]])
 In [191]:
 p3[1,:,1]
 Out[191]: array([10, 13, 16])
 EXPLANATION : 1 respresnts middle column , (:) all columns , 1 represnts middle column
 In [192]:
 # Fetch 3 array--->22,23,25,26
 p3
 Out[192]: array([[[ 0,  1,  2],
 [ 3,  4,  5],
 [ 6,  7,  8]],
 [[ 9, 10, 11],
 [12, 13, 14],
 [15, 16, 17]],
 [[18, 19, 20],
 [21, 22, 23],
 [24, 25, 26]]])
 In [194]:
 p3[2] # last row
 Out[194]: array([[18, 19, 20],
 [21, 22, 23],
 [24, 25, 26]])
 22/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [195]:
 p3[2, 1: ] # last two rows
 Out[195]: array([[21, 22, 23],
 [24, 25, 26]])
 In [196]:
 p3[2, 1: ,1:] # last two columns
 Out[196]: array([[22, 23],
 [25, 26]])
 EXPLANATION : Here we go through 3 stages , where 2 for last array , and (1:) from second
 row to total rows , and (1:) is for second column to total columns
 In [197]:
 # Fetch o, 2, 18 , 20
 p3
 Out[197]: array([[[ 0,  1,  2],
 [ 3,  4,  5],
 [ 6,  7,  8]],
 [[ 9, 10, 11],
 [12, 13, 14],
 [15, 16, 17]],
 [[18, 19, 20],
 [21, 22, 23],
 [24, 25, 26]]])
 In [201]:
 p3[0::2] # for  arrays
 Out[201]: array([[[ 0,  1,  2],
 [ 3,  4,  5],
 [ 6,  7,  8]],
 [[18, 19, 20],
 [21, 22, 23],
 [24, 25, 26]]])
 In [206]:
 p3[0::2 , 0]  # for rows
 Out[206]: array([[ 0,  1,  2],
 [18, 19, 20]])
 In [207]:
 p3[0::2 , 0 , ::2] # for columns
 Out[207]: array([[ 0,  2],
 [18, 20]])
 EXPLANATION : Here we take (0::2) first adn last column , so we did jump using this, and we
 took (0) for first row , and we (::2) ignored middle column
 23/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb 24/29
 Iterating
 In [208]:
 In [211]:
 In [209]:
 In [212]:
 In [210]:
 Out[208]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
 0
 1
 2
 3
 4
 5
 6
 7
 8
 9
 Out[209]: array([[ 0,  1,  2,  3],
      [ 4,  5,  6,  7],
      [ 8,  9, 10, 11]])
 [0 1 2 3]
 [4 5 6 7]
 [ 8  9 10 11]
 Out[210]: array([[[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8]],
      [[ 9, 10, 11],
       [12, 13, 14],
       [15, 16, 17]],
      [[18, 19, 20],
       [21, 22, 23],
       [24, 25, 26]]])
 p1
 # Looping on 1D array
 
for i in p1:
    print(i)
 p2
 ## Looping on 2D array
 
for i in p2:
    print(i) # prints rows
 p3
5/26/23, 4:29 PM NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb 25/29
 In [213]:
 print all items in 3D using nditer ----> first convert in to 1D and applying Loop
 In [215]:
 Reshaping
 Transpose ---> Converts rows in to clumns ad columns into rows
 [[0 1 2]
 [3 4 5]
 [6 7 8]]
 [[ 9 10 11]
 [12 13 14]
 [15 16 17]]
 [[18 19 20]
 [21 22 23]
 [24 25 26]]
 0
 1
 2
 3
 4
 5
 6
 7
 8
 9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 for i in p3:
    print(i)
 for i in np.nditer(p3):
    print(i)
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [217]:
 p2
 Out[217]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 In [219]:
 np.transpose(p2)
 Out[219]: array([[ 0,  4,  8],
 [ 1,  5,  9],
 [ 2,  6, 10],
 [ 3,  7, 11]])
 In [222]:
 # Another method
 p2.T
 Out[222]: array([[ 0,  4,  8],
 [ 1,  5,  9],
 [ 2,  6, 10],
 [ 3,  7, 11]])
 In [221]:
 p3
 Out[221]: array([[[ 0,  1,  2],
 [ 3,  4,  5],
 [ 6,  7,  8]],
 [[ 9, 10, 11],
 [12, 13, 14],
 [15, 16, 17]],
 [[18, 19, 20],
 [21, 22, 23],
 [24, 25, 26]]])
 In [223]:
 p3.T
 Out[223]: array([[[ 0,  9, 18],
 [ 3, 12, 21],
 [ 6, 15, 24]],
 [[ 1, 10, 19],
 [ 4, 13, 22],
 [ 7, 16, 25]],
 [[ 2, 11, 20],
 [ 5, 14, 23],
 [ 8, 17, 26]]])
 Ravel
 Converting any dimensions to 1D
 26/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [225]:
 p2
 Out[225]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 In [224]:
 p2.ravel()
 Out[224]: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
 In [226]:
 p3
 Out[226]: array([[[ 0,  1,  2],
 [ 3,  4,  5],
 [ 6,  7,  8]],
 [[ 9, 10, 11],
 [12, 13, 14],
 [15, 16, 17]],
 [[18, 19, 20],
 [21, 22, 23],
 [24, 25, 26]]])
 In [227]:
 p3.ravel()
 Out[227]: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
 Stacking
 Stacking is the concept of joining arrays in NumPy. Arrays having the same dimensions can be
 stacked
 In [230]:
 In [231]:
 # Horizontal stacking
 w1 = np.arange(12).reshape(3,4)
 w2 = np.arange(12,24).reshape(3,4)
 w1
 Out[231]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 In [232]:
 w2
 Out[232]: array([[12, 13, 14, 15],
 [16, 17, 18, 19],
 [20, 21, 22, 23]])
 27/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 using hstack for Horizontal stacking
 In [236]:
 np.hstack((w1,w2))
 Out[236]: array([[ 0,  1,  2,  3, 12, 13, 14, 15],
 [ 4,  5,  6,  7, 16, 17, 18, 19],
 [ 8,  9, 10, 11, 20, 21, 22, 23]])
 In [237]:
 # Vertical stacking
 w1
 Out[237]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 In [238]:
 w2
 Out[238]: array([[12, 13, 14, 15],
 [16, 17, 18, 19],
 [20, 21, 22, 23]])
 using vstack for vertical stacking
 In [239]:
 np.vstack((w1,w2))
 Out[239]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11],
 [12, 13, 14, 15],
 [16, 17, 18, 19],
 [20, 21, 22, 23]])
 Splitting
 its opposite of Stacking .
 In [240]:
 # Horizontal splitting
 w1
 Out[240]: array([[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]])
 28/29
 localhost:8888/notebooks/ NumPy Fundamentals ( Prudhvi Vardhan Notes).ipynb
5/26/23, 4:29 PM
 NumPy Fundamentals ( Prudhvi Vardhan Notes) - Jupyter Notebook
 In [241]:
 np.hsplit(w1,2) # splitting by 2
 Out[241]: [array([[0, 1],
 [4, 5],
 [8, 9]]),
 array([[ 2,  3],
 [ 6,  7],
 [10, 11]])]
 In [242]:
 np.hsplit(w1,4) # splitting by 4
 Out[242]: [array([[0],
 [4],
 [8]]),
 array([[1],
 [5],
 [9]]),
 array([[ 2],
 [ 6],
 [10]]),
 array([[ 3],
 [ 7],
 [11]])]
 In [244]:
 # Vertical splitting
 w2
 Out[244]: array([[12, 13, 14, 15],
 [16, 17, 18, 19],
 [20, 21, 22, 23]])
 In [246]:
 np.vsplit(w2,3) # splittig into 3 rows
 Out[246]: [array([[12, 13, 14, 15]]),
 array([[16, 17, 18, 19]]),
 array([[20, 21, 22, 23]])]
