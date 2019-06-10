import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#Author: Jonathan Tynan. Creates a PCA plot of the data.

X_imp_train = np.loadtxt('data/X_train_classification.txt')
y_imp_train = np.loadtxt('data/Y_train_classification.txt')
X_imp_dev = np.loadtxt('data/X_dev_classification.txt')
y_imp_dev = np.loadtxt('data/Y_dev_classification.txt').tolist()
X_imp_test = np.loadtxt('data/X_test_classification.txt')
y_imp_test = np.loadtxt('data/Y_test_classification.txt')

posX = X_imp_test[y_imp_test==1]
negX = X_imp_test[y_imp_test==0]

pca = PCA(n_components=2)
pcPos = pca.fit_transform(posX)
pcNeg = pca.fit_transform(negX)

pca.fit(posX)
print(pca.explained_variance_ratio_)

pca.fit(X_imp_test)

#plot the best two principal components
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
plt.scatter(pcPos[:, 0], pcPos[:, 1], marker = 'o', color = 'green', alpha = 0.4, label = 'Positive') 
plt.scatter(pcNeg[:, 0], pcNeg[:, 1], marker = 'o', color = 'red', alpha = 0.4, label = 'Negative')


ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

ax.legend(loc='upper right')


plt.savefig('pca.png', dpi=300)

