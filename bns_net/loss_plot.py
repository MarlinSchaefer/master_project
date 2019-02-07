import json
import matplotlib.pyplot as plt

def make_loss_plot(data_path, image_path):
    with open(data_path, 'r') as FILE:
        data = json.load(FILE)
    
    x_markers = [pt[0] for pt in data]
    tr_loss = [pt[1][0] for pt in data]
    te_loss = [pt[2][0] for pt in data]
    
    plt.plot(x_markers, tr_loss, color='blue', label='Training set loss')
    plt.plot(x_markers, te_loss, color='red', label='Testing set loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(image_path)
    plt.close()
    
