import numpy as np
import matplotlib.pyplot as plt

fn="./asset/data.txt"
# with open(fn,"r"):
data=np.loadtxt(fn)

# plt.scatter(data[:,0],data[:,1])


with open("test.geo","w+") as f:
    for i in range(len(data)):
        x=data[i][0]
        y=data[i][1]
        if abs(x)>1 and abs(y)>1:
            f.write("Point({0}) = {{{1}, {2}, 0.0 }};\n".format(i+1, x, y))


