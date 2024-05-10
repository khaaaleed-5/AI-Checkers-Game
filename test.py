import numpy as np
array = [
    [8,10,12,14],
    [1,2,3,4],
    [5,6,7,8],
    [6,8,9,1]
    ] 



val = np.argmax(array)
print(val)
index = val%4
print(index)
print(val // len(array))