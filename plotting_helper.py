import matplotlib.pyplot as plt
import pandas as pd

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

def plot_loss_and_accuracy_lines():
    model_comparison = pd.DataFrame({
        "Type": ["Baseline Model", "GridSearchCV", "Optuna"],
        "Train Loss": [0.2030, 0.3032, 0.2970],
        "Test Loss": [0.5545, 0.3114, 0.3107],
        "Train Accuracy": [0.9017, 0.8593, 0.8617],
        "Test Accuracy": [0.8433, 0.8588, 0.8584],
    })

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = dict(zip(model_comparison["Type"], colors))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for _, row in model_comparison.iterrows():
        axes[0].plot(
            ["Train", "Test"],
            [row["Train Loss"], row["Test Loss"]],
            marker="o",
            linewidth=2,
            color=color_map[row["Type"]],
            label=row["Type"],
        )
        axes[1].plot(
            ["Train", "Test"],
            [row["Train Accuracy"], row["Test Accuracy"]],
            marker="o",
            linewidth=2,
            color=color_map[row["Type"]],
            label=row["Type"],
        )

    axes[0].set_title("Train vs. Test Loss")
    axes[0].set_xlabel("Metric Split")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Train vs. Test Accuracy")
    axes[1].set_xlabel("Metric Split")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Type", loc="center right")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
