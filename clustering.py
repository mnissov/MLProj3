# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:08:37 2018

@author: emma0
"""
#%% Import section
from mpl_toolkits import mplot3d

from toolbox_024502 import clusterplot
from toolbox_024502 import clusterval
from sklearn.mixture import GaussianMixture

from initData import *
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import model_selection
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

#%% 2 classes
classYbis = np.asarray(np.mat(np.empty((N))).T).squeeze()
for i in range(0,N):
    if y[i] <= np.percentile(y,50):
        classYbis[i] = 0
    else: 
        classYbis[i] = 1
classNamesbis = ['Poor-Lower', 'Middle-Upper']    
Cbis = len(classNamesbis)
#%% 3 classes
classY3 = np.asarray(np.mat(np.empty((N))).T).squeeze()
for i in range(0,N):
    if y[i] <= np.percentile(y,25):
        classY3[i] = 0
    elif y[i] <= np.percentile(y,80):
        classY3[i] = 1
    else: 
        classY3[i] = 2
classNames3 = ['Poor', 'Lower-Middle', 'Upper']    
C3 = len(classNames3)
#%% 5 classes
classY5 = np.asarray(np.mat(np.empty((N))).T).squeeze()
for i in range(0,N):
    if y[i] <= np.percentile(y,20):
        classY5[i] = 0
    elif y[i] <= np.percentile(y,40):
        classY5[i] = 1
    elif y[i] <= np.percentile(y,60):
        classY5[i] = 2
    elif y[i] <= np.percentile(y,80):
        classY5[i] = 3
    else: 
        classY5[i] = 4
classNames5 = ['Poor', 'Lower','Middle','Upper','Rich']    
C5 = len(classNames5)
#%% PCA stdX 2 components
pca = decomposition.PCA(n_components=2)
pca.fit(stdX)
stdX_new = pca.transform(stdX)
N, M = stdX_new.shape
variance_ration = sum(pca.explained_variance_ratio_) #0.52
#%% Plot on the 2 PC with 4 labels
fig1 = figure(1)
#ax = axes(projection='3d')
styles = ['ob', 'or', 'og', 'oy']
#styles = ['ob', 'or', 'og', 'oy']
for c in range(C):
    class_mask = (classY==c)
    plot(stdX_new[class_mask,0], stdX_new[class_mask,1], styles[c])
title('Classification (4 labels) projected onto the first 2 principal directions')
legend(classNames)
fig1.savefig('4labels.png', format='png', dpi=1200)
fig1.clf
#%% Plot on the 2 PC with 2 labels
fig2 = figure(1)
#ax = axes(projection='3d')
styles = ['ob', 'or', 'og', 'oy']
#styles = ['ob', 'or', 'og', 'oy']
for c in range(Cbis):
    class_mask = (classYbis==c)
    plot(stdX_new[class_mask,0], stdX_new[class_mask,1], styles[c])
title('Classification (2 labels) projected onto the first 2 principal directions')
legend(classNamesbis)
fig2.savefig('2labels.png', format='png', dpi=1200)
fig2.clf
#%% Plot on the 2 PC with 3 labels
fig3 = figure(1)
#ax = axes(projection='3d')
styles = ['ob', 'or', 'og', 'oy']
#styles = ['ob', 'or', 'og', 'oy']
for c in range(C3):
    class_mask = (classY3==c)
    plot(stdX_new[class_mask,0], stdX_new[class_mask,1], styles[c])
title('Classification (3 labels) projected onto the first 2 principal directions')
legend(classNames3)
fig3.savefig('3labels.png', format='png', dpi=1200)
fig3.clf
#%% Plot on the 2 PC with 5 labels
fig4 = figure(1)
#ax = axes(projection='3d')
styles = ['ob', 'or', 'og', 'oy','om']
#styles = ['ob', 'or', 'og', 'oy']
for c in range(C5):
    class_mask = (classY5==c)
    plot(stdX_new[class_mask,0], stdX_new[class_mask,1], styles[c])
title('Classification (5 labels) projected onto the first 2 principal directions')
legend(classNames5)
fig4.savefig('5labels.png', format='png', dpi=1200)
fig4.clf
#%% K = 3
# Number of clusters
K = 3
cov_type = 'full'
# type of covariance
reps = 10 
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps).fit(stdX_new)
cls = gmm.predict(stdX_new) 
# extract cluster labels 
cds = gmm.means_ 
# extract cluster centroids (means of gaussians)       
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)

# Plot results:
fig5 = figure(figsize=(14,9))
clusterplot(stdX_new, clusterid=cls, centroids=cds, y=classY3, covars=covs, classNames=classNames3)
title('Results GMM with K=3')
fig5.savefig('3GMM.png', format='png', dpi=1200)
fig5.clf
#%% K = 4
# Number of clusters
K = 4
cov_type = 'full'       
# type of covariance
reps = 10           
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps).fit(stdX_new)
cls = gmm.predict(stdX_new)    
# extract cluster labels
cds = gmm.means_        
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)

# Plot results:
fig6 = figure(figsize=(14,9))
clusterplot(stdX_new, clusterid=cls, centroids=cds, y=classY3, covars=covs, classNames=classNames3)
title('Results GMM with K=4')
fig6.savefig('4GMM.png', format='png', dpi=1200)
fig6.clf

RandGMM, JaccardGMM, NMIGMM = clusterval(classY3,cls) 
#%% Cross-validation, AIC, BIC on GMM
# Range of K's to try
KRange = range(1,11)
T = len(KRange)

covar_type = 'full'     # you can try out 'diag' as well
reps = 3                # number of fits with different initalizations, best result will be kept

# Allocate variables
BIC = np.zeros((T,))
AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10,shuffle=True)

for t,K in enumerate(KRange):
        print('Fitting model for K={0}'.format(K))

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(stdX_new)

        # Get BIC and AIC
        BIC[t,] = gmm.bic(stdX_new)
        AIC[t,] = gmm.aic(stdX_new)

        # For each crossvalidation fold
        for train_index, test_index in CV.split(stdX_new):

            # extract training and test set for current CV fold
            X_train = stdX_new[train_index]
            X_test = stdX_new[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

            # compute negative log likelihood of X_test
            CVE[t] += -gmm.score_samples(X_test).sum()
            

# Plot results
fig7 = figure(1); 
plot(KRange, BIC,'-*b')
plot(KRange, AIC,'-xr')
plot(KRange, 2*CVE,'-ok')
legend(['BIC', 'AIC', 'Crossvalidation'])
xlabel('K')
title('Results cross-validation GMM')
fig7.savefig('cross-validation GMM.png', format='png', dpi=1200)
fig7.clf

#%% Evaluate the quality of GMM in terms of our label information
Rand = np.zeros((10,))
Jaccard = np.zeros((10,))
NMI = np.zeros((10,))

for K in range(1,11):
    cov_type = 'full'
    reps = 10 
    gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps).fit(stdX_new)
    cls = gmm.predict(stdX_new)  
    # compute cluster validities:
    Rand[K-1], Jaccard[K-1], NMI[K-1] = clusterval(classYbis,cls)    
        
# Plot results:
# Plot results:
fig8 = figure(1)
title('Cluster validity')
plot(np.arange(10)+1, Rand)
plot(np.arange(10)+1, Jaccard)
plot(np.arange(10)+1, NMI)
legend(['Rand', 'Jaccard', 'NMI'], loc=2)
fig8.savefig('3_classes_GMM_validity.png', format='png', dpi=1200)
fig8.clf
#%% Hierarchical clustering average linkage K=4
Method = 'average'
Metric = 'euclidean'

Z = linkage(stdX_new, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 4
cls = fcluster(Z, criterion='maxclust', t=Maxclust)

fig9 = figure(1)
clusterplot(stdX_new, cls.reshape(cls.shape[0],1), y=classY3,classNames = classNames3)
title('Results hierarchical clustering (average)')
fig9.savefig('HC average1.png', format='png', dpi=1200)
fig9.clf
# Display dendrogram
max_display_levels=20

fig10 = figure(2,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)
title('Results hierarchical clustering (average)')
fig10.savefig('HC average2.png', format='png', dpi=1200)
fig10.clf

Randavg, Jaccardavg, NMIavg = clusterval(classY3,cls)
#%% Hierarchical clustering maximum linkage K=4
Method = 'complete'
Metric = 'euclidean'

Z = linkage(stdX_new, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 4
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
fig11 = figure(1)
clusterplot(stdX_new, cls.reshape(cls.shape[0],1), y=classY3,classNames = classNames3)
title('Results hierarchical clustering (maximum)')
fig11.savefig('HC maximum1.png', format='png', dpi=1200)
fig11.clf

# Display dendrogram
max_display_levels=20

fig12 = figure(2,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)
title('Results hierarchical clustering (maximum)')
fig12.savefig('HC maximum2.png', format='png', dpi=1200)
fig12.clf

Randmax, Jaccardmax, NMImax = clusterval(classY3,cls)
#%% Evaluate the quality of the clustering in terms of our label information
Rand = [RandGMM, Randavg, Randmax]
Jaccard = [JaccardGMM, Jaccardavg, Jaccardmax]
NMI = [NMIGMM, NMIavg, NMImax]

# Plot results:
fig13 = figure(1)
title('Cluster validity')
plot(['GMM K=5','HC K=5 average','HC K=5 maximum'], Rand,'ob')
plot(['GMM K=5','HC K=5 average','HC K=5 maximum'], Jaccard,'og')
plot(['GMM K=5','HC K=5 average','HC K=5 maximum'], NMI,'or')
legend(['Rand', 'Jaccard', 'NMI'], loc=5)
fig13.savefig('methods_validity.png', format='png', dpi=1200)
fig13.clf