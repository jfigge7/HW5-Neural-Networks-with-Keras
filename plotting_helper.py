import matplotlib.pyplot as plt

# Plotting helpers

def plot_loss(history):
    # plot loss vs val_loss
    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def plot_accuracy(history):
    # plot the training and validation accuracy
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))     
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

def plot_loss_and_accuracy(history):
    return plot_loss(history), plot_accuracy(history)