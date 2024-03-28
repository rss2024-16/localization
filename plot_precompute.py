import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

vals = np.load('precomputed_table.npy')
phits = np.load('phits.npy')
pothers = np.load('pothers.npy')
# axis = [i for i in range(vals.shape[0])]
# print(vals.shape[0])

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# xs = [i for i in range(vals.shape[0])]
# z = []
# print('plotting')
# for x in range(vals.shape[0]):
#     for y in range(vals.shape[0]):
#         z.append(vals[x][y])

# ax.scatter3D(xs,xs,z)
# print('done plotting')
# plt.imshow(vals)
x = np.arange(vals.shape[0])
y = np.arange(vals.shape[1])

X,Y = np.meshgrid(x,y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X,Y,vals)
# ax.plot_surface(X,Y,pothers)


# print(vals[1][1])
plt.show()