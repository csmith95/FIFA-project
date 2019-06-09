import matplotlib.pyplot as plt
import numpy as np

scores_naive = [0.64, 0.57, 0.56]
scores_lr = [0.66, 0.62, 0.62]
scores_svm = [1.0, 0.57, 0.56]
scores_rf = [0.98, 0.59, 0.60]
scores_nn = [0.65, 0.59, 0.60]

data = np.asarray([scores_naive, scores_lr, scores_svm, scores_rf, scores_nn])

# legend = ['Naive', 'Logistic Regression', 'SVM', '5-Layer NN with BN/L2 Reg']
# plt.hist([scores_naive, scores_lr, scores_svm, score], color=['red', 'green', 'blue'])
# plt.xlabel("Classifier Model")
# plt.ylabel("Accuracy")
# plt.legend(legend)
# plt.title('Player Improvement Classification Accuracies')
# plt.show()

# Setting the positions and width for the bars
train = data[:, 0].flatten()
dev = data[:, 1].flatten()
test = data[:, 2].flatten()

pos = list(range(len(train))) 
width = 0.25 
    
# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

# Create a bar with pre_score data,
# in position pos,
plt.bar(pos, 
        #using df['pre_score'] data,
        train, 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#EE3224', 
        # with label the first value in first_name
        label='train') 

# Create a bar with mid_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos], 
        #using df['mid_score'] data,
        dev,
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#F78F1E', 
        # with label the second value in first_name
        label='dev') 

# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + width*2 for p in pos], 
        #using df['post_score'] data,
        test, 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#FFC222', 
        # with label the third value in first_name
        label='test') 

# Set the y axis label
ax.set_ylabel('Accuracy')

# Set the chart's title
ax.set_title('Player Improvement Classification Accuracy')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(['Naive', 'Log Reg', 'SVM', 'RF', '5-Layer NN with BN/Reg'])

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
plt.ylim([0, 1.0] )

# Adding the legend and showing the plot
plt.legend(['Train', 'Dev', 'Test'], loc='upper right')
plt.grid()
plt.show()