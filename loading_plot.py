import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#Author Jonathan Tynan. Creates a loading plot from the data.

features = ['Finishing', 'ShotPower', 'Jumping', 'Balance', 'LongPassing', 'Reactions', 'Overall', 'Positioning', 'GKHandling', 'Marking', 'SprintSpeed', 'Volleys', 'Interceptions', 'ShortPassing', 'Dribbling', 'LongShots', 'Composure', 'HeadingAccuracy', 'StandingTackle', 'Strength', 'Curve', 'Agility', 'Crossing', 'Age', 'Acceleration', 'SlidingTackle', 'Stamina', 'GKDiving', 'GKReflexes', 'Vision', 'Penalties', 'GKPositioning', 'GKKicking', 'Aggression', 'BallControl', 'LB', 'RDM', 'RAM', 'RM', 'RWB', 'RW', 'RS', 'LCM', 'RF', 'CB', 'GK', 'RCB', 'ST', 'RCM', 'LWB', 'LAM', 'LW', 'CF', 'CDM', 'LM', 'LCB', 'LDM', 'CAM', 'RB', 'LS', 'LF', 'CM']
X_imp_train = np.loadtxt('data/X_train_classification.txt')
y_imp_train = np.loadtxt('data/Y_train_classification.txt')
X_imp_test = np.loadtxt('data/X_test_classification.txt')
y_imp_test = np.loadtxt('data/Y_test_classification.txt')

scaler = StandardScaler()
scaler.fit(X_imp_test)
X=scaler.transform(X_imp_test)

pca = PCA()
pca.fit(X,y_imp_test)
x_new = pca.transform(X)   

#Citation: StackOverflow user makaros' answer a biplot question helped me write the following code:
#https://stackoverflow.com/questions/47370795/pca-on-sklearn-how-to-interpret-pca-components
def biplot(score,coeff,labels=features):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    print(n)

    for i in range(n):
        if coeff[i,0] > 0.06 or coeff[i,1] > 0.15 or coeff[i,0] < -0.2 or coeff[i,1] < -0.2:
            plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_ylim(-.75, .75)
ax.set_xlim(-.75, .75)

plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()

biplot(x_new[:,0:2], pca. components_) 
plt.show()
