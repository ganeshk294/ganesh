 Importing Pandas
 In [1]:
 You should be able to import Pandas after installing it
 We'll import 
pandas as its alias name 
import pandas as pd 
import numpy as np 
pd
 Introduction: Why to use Pandas?
 How is it different from numpy ?
 The major limitation of numpy is that it can only work with 1 datatype at a time
 Most real-world datasets contain a mixture of different datatypes
 Like names of places would be string but their population would be int
 ==> It is difficult to work with data having heterogeneous values using Numpy
 Pandas can work with numbers and strings together
 So lets see how we can use pandas
 Imagine that you are a Data Scientist with McKinsey
 McKinsey wants to understand the relation between GDP per capita and life expectancy and various
 trends for their clients.
 The company has acquired data from multiple surveys in different countries in the past
 This contains info of several years about:
 country
 population size
 life expectancy
 GDP per Capita
We have to analyse the data and draw inferences meaningful to the company
 Reading dataset in Pandas
 Link:https://drive.google.com/file/d/1E3bwvYGf1ig32RmcYiWc0IXPN-mD_bI_/view?usp=sharing
 In [3]:
 In [4]:
 Out[4]:
 In [5]:
 df = pd.read_csv(r"C:\Users\kumar\Downloads\mckinsey.csv") 
Now how should we read this dataset?
 Pandas makes it very easy to work with these kinds of files
 df 
country year population continent life_exp
 gdp_cap
 0 Afghanistan 1952
 1 Afghanistan 1957
 8425333
 Asia
 28.801 779.445314
 9240934
 2 Afghanistan 1962
 3 Afghanistan 1967
 10267083
 Asia
 Asia
 30.332 820.853030
 31.997 853.100710
 11537966
 4 Afghanistan 1972
 ...
 ...
 ...
 13079460
 Asia
 Asia
 34.020 836.197138
 36.088 739.981106
 ...
 1699
 Zimbabwe 1987
 9216418
 ...
 Africa
 ...
 ...
 62.351 706.157306
 1700
 Zimbabwe 1992 10704340
 Africa
 1701
 1702
 Zimbabwe 1997 11404948
 Zimbabwe 2002 11926563
 Africa
 60.377 693.420786
 46.809 792.449960
 Africa
 1703
 Zimbabwe 2007 12311143
 1704 rows × 6 columns
 Dataframe and Series
 Africa
 39.989 672.038623
 43.487 469.709298
 What can we observe from the above dataset ?
 We can see that it has:
 6 columns
 1704 rows
 What do you think is the datatype of 
type(df) 
df ?
 pandas.core.frame.DataFrame
 Out[5]:
 Its a pandas DataFrame
What is a pandas DataFrame ?
 It is a table-like representation of data in Pandas => Structured Data
 Structured Data here can be thought of as tabular data in a proper order
 Considered as counterpart of 2D-Matrix in Numpy
 Now how can we access a column, say 
In [6]:
 df["country"] 
country of the dataframe?
 Out[6]:
 In [7]:
 Out[7]:
 In [8]:
 0       Afghanistan 
1       Afghanistan 
2       Afghanistan 
3       Afghanistan 
4       Afghanistan 
           ...      
1699       Zimbabwe 
1700       Zimbabwe 
1701       Zimbabwe 
1702       Zimbabwe 
1703       Zimbabwe 
Name: country, Length: 1704, dtype: object
 As you can see we get all the values in the column country
 Now what is the data-type of a column?
 type(df["country"]) 
pandas.core.series.Series
 Its a pandas Series
 What is a pandas Series ?
 Series in Pandas is what a Vector is in Numpy
 What exactly does that mean?
 It means a Series is a single column of data
 Multiple Series stack together to form a DataFrame
 Now we have understood what Series and DataFrames are
 What if a dataset has 100 rows ... Or 100 columns ?
 How can we find the datatype, name, total entries in each column ?
 df.info() 
<class 'pandas.core.frame.DataFrame'> 
RangeIndex: 1704 entries, 0 to 1703 
Data columns (total 6 columns): 
 #   Column      Non-Null Count  Dtype   ---  ------      --------------  -----   
 0   country     1704 non-null   object  
 1   year        1704 non-null   int64   
 2   population  1704 non-null   int64   
 3   continent   1704 non-null   object  
 4   life_exp    1704 non-null   float64 
 5   gdp_cap     1704 non-null   float64 
dtypes: float64(2), int64(2), object(2) 
memory usage: 80.0+ KB 
df.info() gives a list of columns with:
 Name/Title of Columns
 How many non-null values (blank cells) each column has
 Type of values in each column - int, float, etc.
 By default, it shows data-type as object for anything other than int or float - Will come back later
 Now what if we want to see the first few rows in the dataset ?
 country year population continent life_exp gdp_cap
 0 Afghanistan 1952 8425333 Asia 28.801 779.445314
 1 Afghanistan 1957 9240934 Asia 30.332 820.853030
 2 Afghanistan 1962 10267083 Asia 31.997 853.100710
 3 Afghanistan 1967 11537966 Asia 34.020 836.197138
 4 Afghanistan 1972 13079460 Asia 36.088 739.981106
 It Prints top 5 rows by default
 We can also pass in number of rows we want to see in head()
 country year population continent life_exp gdp_cap
 0 Afghanistan 1952 8425333 Asia 28.801 779.445314
 1 Afghanistan 1957 9240934 Asia 30.332 820.853030
 2 Afghanistan 1962 10267083 Asia 31.997 853.100710
 3 Afghanistan 1967 11537966 Asia 34.020 836.197138
 4 Afghanistan 1972 13079460 Asia 36.088 739.981106
 5 Afghanistan 1977 14880372 Asia 38.438 786.113360
 6 Afghanistan 1982 12881816 Asia 39.854 978.011439
 7 Afghanistan 1987 13867957 Asia 40.822 852.395945
 8 Afghanistan 1992 16317921 Asia 41.674 649.341395
 9 Afghanistan 1997 22227415 Asia 41.763 635.341351
 10 Afghanistan 2002 25268405 Asia 42.129 726.734055
 11 Afghanistan 2007 31889923 Asia 43.828 974.580338
 12 Albania 1952 1282697 Europe 55.230 1601.056136
 13 Albania 1957 1476505 Europe 59.280 1942.284244
 In [9]: df.head() 
Out[9]:
 In [10]: df.head(20) 
Out[10]:
14 Albania 1962 1728137 Europe 64.820 2312.888958
 15 Albania 1967 1984060 Europe 66.220 2760.196931
 16 Albania 1972 2263554 Europe 67.690 3313.422188
 17 Albania 1977 2509048 Europe 68.930 3533.003910
 18 Albania 1982 2780097 Europe 70.420 3630.880722
 19 Albania 1987 3075321 Europe 72.000 3738.932735
 Similarly what if we want to see the last 20 rows ?
 country year population continent life_exp gdp_cap
 1684 Zambia 1972 4506497 Africa 50.107 1773.498265
 1685 Zambia 1977 5216550 Africa 51.386 1588.688299
 1686 Zambia 1982 6100407 Africa 51.821 1408.678565
 1687 Zambia 1987 7272406 Africa 50.821 1213.315116
 1688 Zambia 1992 8381163 Africa 46.100 1210.884633
 1689 Zambia 1997 9417789 Africa 40.238 1071.353818
 1690 Zambia 2002 10595811 Africa 39.193 1071.613938
 1691 Zambia 2007 11746035 Africa 42.384 1271.211593
 1692 Zimbabwe 1952 3080907 Africa 48.451 406.884115
 1693 Zimbabwe 1957 3646340 Africa 50.469 518.764268
 1694 Zimbabwe 1962 4277736 Africa 52.358 527.272182
 1695 Zimbabwe 1967 4995432 Africa 53.995 569.795071
 1696 Zimbabwe 1972 5861135 Africa 55.635 799.362176
 1697 Zimbabwe 1977 6642107 Africa 57.674 685.587682
 1698 Zimbabwe 1982 7636524 Africa 60.363 788.855041
 1699 Zimbabwe 1987 9216418 Africa 62.351 706.157306
 1700 Zimbabwe 1992 10704340 Africa 60.377 693.420786
 1701 Zimbabwe 1997 11404948 Africa 46.809 792.449960
 1702 Zimbabwe 2002 11926563 Africa 39.989 672.038623
 1703 Zimbabwe 2007 12311143 Africa 43.487 469.709298
 How can we find the shape of the dataframe?
 (1704, 6)
 Similar to Numpy, it gives No. of Rows and Columns -- Dimensions
 Now we know how to do some basic operations on dataframes
 In [11]: df.tail(20) #Similar to head 
Out[11]:
 In [12]: df.shape 
Out[12]:
But what if we aren't loading a dataset, but want to create our own.
 Let's take a subset of the original dataset
 country year population continent life_exp gdp_cap
 0 Afghanistan 1952 8425333 Asia 28.801 779.445314
 1 Afghanistan 1957 9240934 Asia 30.332 820.853030
 2 Afghanistan 1962 10267083 Asia 31.997 853.100710
 How can we create a DataFrame from scratch?
 Approach 1: Row-oriented
 It takes 2 arguments - Because DataFrame is 2-dimensional
 A list of rows
 Each row is packed in a list []
 All rows are packed in an outside list [[]] - To pass a list of rows
 A list of column names/labels
 country year population continent life_exp gdp_cap
 0 Afghanistan 1952 8425333 Asia 28.801 779.445314
 1 Afghanistan 1957 9240934 Asia 30.332 820.853030
 2 Afghanistan 1962 102267083 Asia 31.997 853.100710
 Can you create a single row dataframe?--------------------------------------------------------------------------- 
ValueError                                Traceback (most recent call last) 
Input In [15], in <cell line: 1>() ----> 1 pd.DataFrame(['Afghanistan',1952, 8425333, 'Asia', 28.801, 779.445314 ],  
      2              columns=['country','year','population','continent','life_exp','gdp_
 cap']) 
 
File ~\anaconda3\lib\site-packages\pandas\core\frame.py:737, in DataFrame.__init__(self,
 data, index, columns, dtype, copy) 
    729         mgr = arrays_to_mgr( 
    730             arrays, 
    731             columns, 
   (...) 
    734             typ=manager, 
    735         ) 
In [13]: df.head(3) # We take the first 3 rows to create our dataframe 
Out[13]:
 In [14]: pd.DataFrame([['Afghanistan',1952, 8425333, 'Asia', 28.801, 779.445314 ], 
              ['Afghanistan',1957, 9240934, 'Asia', 30.332, 820.853030 ], 
              ['Afghanistan',1962, 102267083, 'Asia', 31.997, 853.100710 ]],  
             columns=['country','year','population','continent','life_exp','gdp_cap']) 
Out[14]:
 In [15]: pd.DataFrame(['Afghanistan',1952, 8425333, 'Asia', 28.801, 779.445314 ],  
             columns=['country','year','population','continent','life_exp','gdp_cap']) 
    736     else: --> 737         mgr = ndarray_to_mgr( 
    738             data, 
    739             index, 
    740             columns, 
    741             dtype=dtype, 
    742             copy=copy, 
    743             typ=manager, 
    744         ) 
    745 else: 
    746     mgr = dict_to_mgr( 
    747         {}, 
    748         index, 
   (...) 
    751         typ=manager, 
    752     ) 
 
File ~\anaconda3\lib\site-packages\pandas\core\internals\construction.py:351, in ndarray
 _to_mgr(values, index, columns, dtype, copy, typ) 
    346 # _prep_ndarray ensures that values.ndim == 2 at this point 
    347 index, columns = _get_axes( 
    348     values.shape[0], values.shape[1], index=index, columns=columns 
    349 ) --> 351 _check_values_indices_shape_match(values, index, columns) 
    353 if typ == "array": 
    355     if issubclass(values.dtype.type, str): 
 
File ~\anaconda3\lib\site-packages\pandas\core\internals\construction.py:422, in _check_
 values_indices_shape_match(values, index, columns) 
    420 passed = values.shape 
    421 implied = (len(index), len(columns)) --> 422 raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}") 
 
ValueError: Shape of passed values is (6, 1), indices imply (6, 6)
 Why did this give an error?
 Because we passed in a list of values
 DataFrame() expects a list of rows
 country year population continent life_exp gdp_cap
 0 Afghanistan 1952 8425333 Asia 28.801 779.445314
 Approach 2: Column-oriented
 country year population continent life_exp gdp_cap
 0 Afghanistan 1952 842533 Asia 28.801 779.445314
 1 Afghanistan 1957 9240934 Asia 30.332 820.853030
 We pass the data as a dictionary
 In [16]: pd.DataFrame([['Afghanistan',1952, 8425333, 'Asia', 28.801, 779.445314 ]],  
             columns=['country','year','population','continent','life_exp','gdp_cap']) 
Out[16]:
 In [17]: pd.DataFrame({'country':['Afghanistan', 'Afghanistan'], 'year':[1952,1957], 
              'population':[842533, 9240934], 'continent':['Asia', 'Asia'], 
              'life_exp':[28.801, 30.332], 'gdp_cap':[779.445314, 820.853030]}) 
Out[17]:
Key is the Column Name/Label
 Value is the list of values column-wise
 We now have a basic idea about the dataset and creating rows and columns
 What kind of other operations can we perform on the dataframe?
 Thinking from database perspective:
 Adding data
 Removing data
 Updating/Modifying data
 and so on
 Basic operations on columns
 Now what operations can we do using columns?
 Maybe add a column
 or delete a column
 or we can rename the column too
 and so on.
 We can see that our dataset has 6 cols
 But what if our dataset has 20 cols ? ... or 100 cols ? We can't see ther names in one go.
 How can we get the names of all these cols ?
 We can do it in two ways:
 1. df.columns
 2. df.keys
 In [18]:
 Out[18]:
 In [19]:
 Out[19]:
 df.columns  # using attribute `columns` of dataframe 
Index(['country', 'year', 'population', 'continent', 'life_exp', 'gdp_cap'], dtype='obje
 ct')
 df.keys()  # using method keys() of dataframe 
Index(['country', 'year', 'population', 'continent', 'life_exp', 'gdp_cap'], dtype='obje
 ct')
 Note:
 Here, 
Index is a type of pandas class used to store the 
address of the series/dataframe
 It is an Immutable sequence used for indexing and alignment.
 # df['country'].head()  # Gives values in Top 5 rows pertaining to the key 
In [20]:
 Pandas DataFrame and Series are specialised dictionary
In [21]:
 But what is so "special" about this dictionary?
 It can take multiple keys
 df[['country', 'life_exp']].head()  
Out[21]:
 In [22]:
 Out[22]:
 country life_exp
 0 Afghanistan
 1 Afghanistan
 28.801
 30.332
 2 Afghanistan
 3 Afghanistan
 31.997
 34.020
 4 Afghanistan
 36.088
 And what if we pass a single column name?
 df[['country']].head()  
country
 0 Afghanistan
 1 Afghanistan
 2 Afghanistan
 3 Afghanistan
 4 Afghanistan
 Note:
 Notice how this output type is different from our earlier output using 
==> 
['country'] gives series while 
df['country']
 [['country']] gives dataframe
 Now that we know how to access columns, lets answer some questions
 How can we find the countries that have been surveyed ?
 We can find the unique vals in the 
country col
 How can we find unique values in a column?
 In [23]:
 Out[23]:
 df['country'].unique() 
array(['Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina', 
       'Australia', 'Austria', 'Bahrain', 'Bangladesh', 'Belgium', 
       'Benin', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 
       'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 
       'Canada', 'Central African Republic', 'Chad', 'Chile', 'China', 
       'Colombia', 'Comoros', 'Congo, Dem. Rep.', 'Congo, Rep.', 
       'Costa Rica', "Cote d'Ivoire", 'Croatia', 'Cuba', 'Czech Republic', 
       'Denmark', 'Djibouti', 'Dominican Republic', 'Ecuador', 'Egypt', 
       'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Ethiopia', 
       'Finland', 'France', 'Gabon', 'Gambia', 'Germany', 'Ghana', 
       'Greece', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Haiti', 
       'Honduras', 'Hong Kong, China', 'Hungary', 'Iceland', 'India', 
       'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 
       'Jamaica', 'Japan', 'Jordan', 'Kenya', 'Korea, Dem. Rep.', 
       'Korea, Rep.', 'Kuwait', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 
       'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Mauritania', 
       'Mauritius', 'Mexico', 'Mongolia', 'Montenegro', 'Morocco', 
       'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands', 
       'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Norway', 'Oman', 
       'Pakistan', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland', 
       'Portugal', 'Puerto Rico', 'Reunion', 'Romania', 'Rwanda', 
       'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 
       'Sierra Leone', 'Singapore', 'Slovak Republic', 'Slovenia', 
       'Somalia', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan', 
       'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 
       'Tanzania', 'Thailand', 'Togo', 'Trinidad and Tobago', 'Tunisia', 
       'Turkey', 'Uganda', 'United Kingdom', 'United States', 'Uruguay', 
       'Venezuela', 'Vietnam', 'West Bank and Gaza', 'Yemen, Rep.', 
       'Zambia', 'Zimbabwe'], dtype=object)
 Now what if you also want to check the count of each country in the dataframe?
 In [24]:
 Out[24]:
 In [25]:
 Out[25]:
 df['country'].value_counts() 
Afghanistan          12 
Pakistan             12 
New Zealand          12 
Nicaragua            12 
Niger                12 
                     .. 
Eritrea              12 
Equatorial Guinea    12 
El Salvador          12 
Egypt                12 
Zimbabwe             12 
Name: country, Length: 142, dtype: int64
 Note:
 value_counts() shows the output in decreasing order of frequency
 What if we want to change the name of a column ?
 We can rename the column by:
 passing the dictionary with 
specifying 
axis=1
 old_name:new_name pair
 df.rename({"population": "Population", "country":"Country" }, axis = 1) 
Country year Population continent life_exp
 gdp_cap
 0 Afghanistan 1952
 1 Afghanistan 1957
 8425333
 Asia
 28.801 779.445314
 9240934
 2 Afghanistan 1962 10267083
 3 Afghanistan 1967 11537966
 Asia
 Asia
 30.332 820.853030
 31.997 853.100710
 Asia
 4 Afghanistan 1972 13079460
 ...
 ...
 ...
 Asia
 34.020 836.197138
 36.088 739.981106
 ...
 1699
 Zimbabwe 1987
 9216418
 ...
 Africa
 ...
 ...
 62.351 706.157306
1700 Zimbabwe 1992 10704340 Africa 60.377 693.420786
 1701 Zimbabwe 1997 11404948 Africa 46.809 792.449960
 1702 Zimbabwe 2002 11926563 Africa 39.989 672.038623
 1703 Zimbabwe 2007 12311143 Africa 43.487 469.709298
 1704 rows × 6 columns
 Alternatively, we can also rename the column without using axis
 by using the column parameter
 Country year population continent life_exp gdp_cap
 0 Afghanistan 1952 8425333 Asia 28.801 779.445314
 1 Afghanistan 1957 9240934 Asia 30.332 820.853030
 2 Afghanistan 1962 10267083 Asia 31.997 853.100710
 3 Afghanistan 1967 11537966 Asia 34.020 836.197138
 4 Afghanistan 1972 13079460 Asia 36.088 739.981106
 ... ... ... ... ... ... ...
 1699 Zimbabwe 1987 9216418 Africa 62.351 706.157306
 1700 Zimbabwe 1992 10704340 Africa 60.377 693.420786
 1701 Zimbabwe 1997 11404948 Africa 46.809 792.449960
 1702 Zimbabwe 2002 11926563 Africa 39.989 672.038623
 1703 Zimbabwe 2007 12311143 Africa 43.487 469.709298
 1704 rows × 6 columns
 We can set it inplace by setting the inplace argument = True
 Country year population continent life_exp gdp_cap
 0 Afghanistan 1952 8425333 Asia 28.801 779.445314
 1 Afghanistan 1957 9240934 Asia 30.332 820.853030
 2 Afghanistan 1962 10267083 Asia 31.997 853.100710
 3 Afghanistan 1967 11537966 Asia 34.020 836.197138
 4 Afghanistan 1972 13079460 Asia 36.088 739.981106
 ... ... ... ... ... ... ...
 1699 Zimbabwe 1987 9216418 Africa 62.351 706.157306
 1700 Zimbabwe 1992 10704340 Africa 60.377 693.420786
 1701 Zimbabwe 1997 11404948 Africa 46.809 792.449960
 In [26]: df.rename(columns={"country":"Country"}) 
Out[26]:
 In [27]: df.rename({"country": "Country"}, axis = 1, inplace = True) 
df 
Out[27]:
1702
 1703
 Zimbabwe 2002 11926563
 Zimbabwe 2007 12311143
 1704 rows × 6 columns
 Note
 Africa
 Africa
 39.989 672.038623
 43.487 469.709298
 .rename has default value of axis=0
 If two columns have the same name, then 
df['column'] will display both columns
 Now lets try another way of accessing column vals
 In [28]:
 Out[28]:
 In [29]:
 df.Country 
0       Afghanistan 
1       Afghanistan 
2       Afghanistan 
3       Afghanistan 
4       Afghanistan 
           ...      
1699       Zimbabwe 
1700       Zimbabwe 
1701       Zimbabwe 
1702       Zimbabwe 
1703       Zimbabwe 
Name: Country, Length: 1704, dtype: object
 This however doesn't work everytime
 What do you think could be the problems with using attribute style for accessing the
 columns?
 Problems such as
 if the column names are not strings
 Starting with number: E.g., 
2nd
 Contains a space: E.g., 
Roll Number
 or if the column names conflict with methods of the DataFrame
 E.g. 
shape
 It is generally better to avoid this type of accessing columns
 Are all the columns in our data necessary?
 We already know the continents in which each country lies
 So we don't need this column
