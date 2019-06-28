import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_loss_plot(data_path, image_path):
    with open(data_path, 'r') as FILE:
        data = json.load(FILE)
    
    x_markers = [pt[0] for pt in data]
    tr_loss = [pt[1][0] for pt in data]
    te_loss = [pt[2][0] for pt in data]
    
    dpi = 96
    plt.figure(figsize=(1920.0/dpi, 1440.0/dpi), dpi=dpi)
    plt.rcParams.update({'font.size': 22, 'text.usetex': 'true'})
    
    plt.plot(x_markers, tr_loss, color='blue', label='Training set loss')
    plt.plot(x_markers, te_loss, color='red', label='Testing set loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(image_path)
    plt.close()
    
