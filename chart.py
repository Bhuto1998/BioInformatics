import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_file = "Loss-Accuracy.csv"
df = pd.read_csv(input_file)

def create_plots(df):
    plt.plot(df['Train Accuracy'])
    plt.plot(df['Validation Accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy_updated.png')
    plt.clf()

    plt.plot(df['Train Loss'])
    plt.plot(df['Validation Loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_updated.png')
    plt.clf()

create_plots(df)