import numpy as np
import matplotlib.pyplot as plt

def hist_pair_heat(X, X_header, y, hist=True, pairplot=True, heatmap=True):
    no_of_cols = len(X_header)
    classes = list(set(y))
    class1_indices = np.where(y==classes[0])
    class2_indices = np.where(y==classes[1])
    cols = {}
    plt.figure(figsize = (40, 3))
    for i in range(no_of_cols):
        cols[X_header[i]] = X[:, i].astype(float)
        
    if hist: 
        for i in range(no_of_cols):
            plt.subplot(1, no_of_cols, i+1)
            plt.hist(X[:, i], bins=20)
            plt.xlabel(X_header[i])

    if pairplot:
        # Estimate number of rows and columns for subplots based on the number of features
        if no_of_cols%2 == 0:
            plt_cols = (no_of_cols-1)//2
            plt_rows = no_of_cols
        else:
            plt_cols = no_of_cols
            plt_rows = (no_of_cols-1)//2
        plt.figure(figsize = (plt_cols*8, plt_rows*8))
        plt_count = 0
        for i in range(1, no_of_cols):
            for j in range(i):
                plt_count+=1
                plt.subplot(plt_rows, plt_cols, plt_count)
                plt.scatter(X[class1_indices, i], X[class1_indices, j], label = classes[0])
                plt.scatter(X[class2_indices, i], X[class2_indices, j], label = classes[1])
                plt.xlabel(X_header[i])
                plt.ylabel(X_header[j])
                plt.legend()
    
    if heatmap:
        # Calculate the correalation matrix using numpy inbuilt function.
        cm = np.corrcoef(np.array(list(cols.values())))
        
        plt.figure(figsize = (10, 10))
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xticks(np.arange(no_of_cols), labels = X_header)
        ax.set_yticks(np.arange(no_of_cols), labels = X_header)
        # Rotate the xticklabels by 90 degree
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode = "anchor")
        for i in range(no_of_cols):
            for j in range(no_of_cols):
                ax.text(j, i, round(cm[i, j], 2), ha="center", va="center", color="black", fontsize = 8)
        ax.set_title("Correlation between features")