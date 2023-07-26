import numpy as np

def describe_matrix(X):
    """
    Displays basic information about the dataset X and target varible y.
    """
    # Define the header for the statistics table
    heading = "Column\t\tMean\t\tstd\t\tMin\t\t25%\t\tMedian\t\t75%\t\tMax\t\tdtype\t\t\tNull_counts\tOutlier_counts"
    print(heading)
    print("-"*(len(heading)*2+28))
    for i in range(X.shape[1]):
        col = np.array(X[:, i])
        minimum = np.min(col)
        maximum = np.max(col)
        mean = np.mean(col)
        first_quartile = np.percentile(col, 25)
        third_quartile = np.percentile(col, 75)
        median = np.median(col)
        std = np.std(col)
        dtype = type(col[0])
        null_counts = np.isnan(col.astype(float)).sum()   # Counts all the NaN and None values in the column
        outlier_count = len(col[(col < (first_quartile - 1.5 * (third_quartile - first_quartile)))|(col > (third_quartile + 1.5 * (third_quartile - first_quartile)))])
        # Print the statistics for the i-th column
        print(f"{i:<10d}\t{mean:<10.2f}\t{std:<10.2f}\t{minimum:<10.2f}\t{first_quartile:<10.2f}\t{median:<10.2f}\t{third_quartile:<10.2f}\t{maximum:<10.2f}\t{dtype}\t\t{null_counts:<10d}\t{outlier_count:<10d}")