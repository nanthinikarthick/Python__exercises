import csv
import matplotlib.pyplot as plt

def plot_data(csv_file_path):
   
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        precision = []
        recall = []
        for row in reader:
            precision.append(float(row['precision']))
            recall.append(float(row['recall']))
    
    
    plt.plot(precision, recall, marker='o', linestyle='-')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()


csv_file_path = "data_file.csv"
with open(csv_file_path, "w") as f:
    w = csv.writer(f)
    _ = w.writerow(["precision", "recall"])
    w.writerows([[0.013, 0.951],
                 [0.376, 0.851],
                 [0.441, 0.839],
                 [0.570, 0.758],
                 [0.635, 0.674],
                 [0.721, 0.604],
                 [0.837, 0.531],
                 [0.860, 0.453],
                 [0.962, 0.348],
                 [0.982, 0.273],
                 [1.0, 0.0]])


plot_data(csv_file_path)



 #1. For some reason the plot is not showing correctly, can you find out what is going wrong?
#   The obvious issue here seems to be that the plot_data function is referenced but not defined.

# 2.How could this be fixed?
# After defining the plot_data function, the problem might be related to how the data is being plotted. 
# It's possible that the x and y coordinates are being interpreted incorrectly or that the plot settings need adjustment.
