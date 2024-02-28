import math
import numpy as np
import matplotlib.pyplot as plt
import torch

from torchvision.utils import save_image

plt.style.use('ggplot')

def psnr(label, output, max_val=1.):
    '''
    Computes the Peak Signal to Noise Ratio. (higher the better.)
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    Output must be normalised as max_val is 1, not 255. Must explain this. 

    This metric is used to monitor the SRCNN model while training. The way it works is that the PSNR
    keeps increasing between output image from the network and the ground truth high resolution 
    keeps on decreasing. 
    ''' 
    label = label.cpu().detach().numpy()    # What the fuck does this do?
    output = output.cpu().detach().numpy()
    difference = output - label
    rmse = math.sqrt(np.mean((difference) ** 2 ))
    if rmse == 0:
        return 100
    else:
        psnr = 20 * math.log10(max_val/rmse)
        return psnr
    

def savePlot(train_loss, val_loss, train_psnr, val_psnr):
    '''
    A simple function to save the loss and PSNR graphs for training and validation. 
    '''
    # Loss Plots
    fig, ax = plt.subplots(2, figsize=(12, 8), sharex=True)
    ax[0].plot(train_loss, color='orange', label='train_loss')
    ax[0].plot(val_loss, color='red', label='validation loss')
    ax[0].xlabel('Epochs')
    ax[0].ylabel('Loss')
    ax[0].legend()
    
    # PSNR plot
    ax[1].plot(train_psnr, color='green', label='train PSNR dB')
    ax[1].plot(val_psnr, color='blue', label='validation PSNR dB')
    ax[1].xlabel('Epochs')
    ax[1].ylabel('PSNR (dB)')
    ax[1].legend()

    fig.savefig(f'../outputs/psnr.png')
    plt.close()


def saveModelState(model):
    '''
    Saves model state(trained weights) to disk. 
    '''
    print('Saving model...')
    torch.save(model.state_dict(), '../ouputs/model.pth')

def saveModel(epochs, model, optimiser, criterion):
    '''
    Saves entire model checkpoint to disk.
    '''
    torch.save(
        {
            'epoch': epochs + 1,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'loss': criterion,
        },
        f'../outputs/model_ckpt.pth'
    )

def saveValidationResults(output, epoch, batch_iter):
    '''
    Saves the validation reconstructed images. 
    '''
    save_image(
        output, 
        f'../outputs/valid_results/val_sr_{epoch}_{batch_iter}.png'
    )