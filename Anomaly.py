# exercise 11.4.1
import numpy as np
from matplotlib.pyplot import (figure, imshow, bar, title, xticks, yticks, cm,
                               subplot, show)
from scipy.io import loadmat
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors

from initData import *

N, M = np.shape(stdX)

### Gausian Kernel density estimator
# cross-validate kernel width by leave-one-out-cross-validation
# (efficient implementation in gausKernelDensity function)
# evaluate for range of kernel widths
widths = stdX.var(axis=0).max() * (2.0**np.arange(-10,3))
logP = np.zeros(np.size(widths))
for i,w in enumerate(widths):
   print('Fold {:2d}, w={:f}'.format(i,w))
   density, log_density = gausKernelDensity(stdX,w)
   logP[i] = log_density.sum()
   9
val = logP.max()
ind = logP.argmax()

width=widths[ind]
print('Optimal estimated width is: {0}'.format(width))

# evaluate density for estimated width
density, log_density = gausKernelDensity(stdX,width)

# Sort the densities
i = (density.argsort(axis=0)).ravel()
density = density[i].reshape(-1,)

# Plot density estimate of outlier score
fig = figure()
bar(range(20),density[:20])
title('Density estimate')
show()
fig.savefig('fig/densityEstimate.eps', format='eps', dpi=1200)   
fig.clf


### K-neighbors density estimator
# Neighbor to use:
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(stdX)
D, i = knn.kneighbors(stdX)

density = 1./(D.sum(axis=1)/K)

# Sort the scores
i = density.argsort()
density = density[i]

# Plot k-neighbor estimate of outlier score (distances)
fig = figure()
bar(range(20),density[:20])
title('KNN density: Outlier score')
show()
fig.savefig('fig/knnDensity.eps', format='eps', dpi=1200)   
fig.clf

### K-nearest neigbor average relative density
# Compute the average relative density

knn = NearestNeighbors(n_neighbors=K).fit(stdX)
D, i = knn.kneighbors(stdX)
density = 1./(D.sum(axis=1)/K)
avg_rel_density = density/(density[i[:,1:]].sum(axis=1)/K)

# Sort the avg.rel.densities
i_avg_rel = avg_rel_density.argsort()
avg_rel_density = avg_rel_density[i_avg_rel]

# Plot k-neighbor estimate of outlier score (distances)
fig = figure()
bar(range(20),avg_rel_density[:20])
title('KNN average relative density: Outlier score')
show()
fig.savefig('fig/knnAvgDensity.eps', format='eps', dpi=1200)   
fig.clf

### Distance to 5'th nearest neighbor outlier score
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(stdX)
D, i = knn.kneighbors(stdX)

# Outlier score
score = D[:,K-1]
# Sort the scores
i = score.argsort()
score = score[i[::-1]]

# Plot k-neighbor estimate of outlier score (distances)
fig = figure()
bar(range(20),score[:20])
title('27th neighbor distance: Outlier score')
show()
fig.savefig('fig/knnDistanceScore.eps', format='eps', dpi=1200)   
fig.clf
