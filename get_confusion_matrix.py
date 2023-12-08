import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
from argparse import ArgumentParser


def save_confusion_matrix(results_path):
    df = pd.read_csv(results_path)
    # Convert class labels to integers using label encoding
    label_encoder = LabelEncoder()
    df['ground_truth'] = label_encoder.fit_transform(df['ground_truth'])
    df['prediction'] = label_encoder.transform(df['prediction'])

    # Extract ground truth and predictions
    y_true = df['ground_truth']
    y_pred = df['prediction']

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Get class labels
    class_labels = label_encoder.classes_
    sns.set(font_scale=4.5)
    # Plot confusion matrix with class labels
    plt.figure(figsize=(40, 40))
    g = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    # g.set_xticklabels(g.get_xticklabels(), rotation = 20)
    # g.set_yticklabels(g.get_yticklabels(), rotation = 0)
    plt.tight_layout()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save the plot with a specific name (replace 'confusion_matrix_plot.png' with your desired file name)
    outdir = "/".join(results_path.split("/")[:-1])
    outname = results_path.split("/")[-1].replace(".csv","_confusion_matrix.png")
    outfile = os.path.join(outdir,outname)
    # exec(os.environ.get("DEBUG"))
    plt.savefig(outfile)
    # Show the plot
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='')
    parser.add_argument('--results_path', type=str, default='/home/k.subash/land-type-classification/logs/NWPU_RESISC45_using_SoundingEarth_trained_ckpt.csv')  
    args = parser.parse_args()
    results_path = args.results_path
    save_confusion_matrix(results_path)