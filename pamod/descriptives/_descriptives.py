import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


class DescriptivesPlotter:
    def __init__(self, df):
        """
        Initializes the DescriptivesClass with a DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame containing the data to be used for plotting.
        """
        self.df = df

    def plt_matrix(self, vertical, horizontal, cmap="Oranges", n=None, normalize="rows"):
        """
        Plots heatmap/confusion matrix.

        Args:
        vertical (np.series): series of values to be represented on the vertical axis
        horizontal (np.series): series of values to be represented on the horizontal axis
        title (str): title of the plot
        n (int): number of total observations
        xlabel (str): label for the x-axis
        ylabel (str): label for the y-axis
        normalize (str): normalization method, 'rows' for row-wise normalization, 'columns' for column-wise normalization
        """
        # Calculate confusion matrix
        vertical_data = self.df[vertical]
        horizontal_data = self.df[horizontal]
        cm = confusion_matrix(vertical_data, horizontal_data)

        # Normalize the matrix
        if normalize == "rows":
            row_sums = cm.sum(axis=1)
            normalized_cm = (cm / row_sums[:, np.newaxis]) * 100
        elif normalize == "columns":
            col_sums = cm.sum(axis=0)
            normalized_cm = (cm / col_sums) * 100
        else:
            raise ValueError("Invalid value for 'normalize'. Use 'rows' or 'columns'.")

        # Plot the heatmap
        plt.figure(figsize=(8, 6), dpi=300)
        sns.heatmap(normalized_cm, cmap=cmap, fmt="g", linewidths=0.5, square=True, cbar_kws={"label": "Percent"})

        for i in range(len(cm)):
            for j in range(len(cm)):
                if normalized_cm[i, j] > 50:  # Value greater than 50%
                    plt.text(j + 0.5, i + 0.5, cm[i, j], ha="center", va="center", color="white")
                else:
                    plt.text(j + 0.5, i + 0.5, cm[i, j], ha="center", va="center")

        if n is not None:
            title = f"Data Overview (n={n})"
        plt.title(title, fontsize=18)
        plt.xlabel("Pocket depth before therapy", fontsize=18)
        plt.ylabel("Pocket depth after therapy", fontsize=18)
        plt.show()

    def pocket_comparison(self, column1, column2):
        """
        Creates two bar plots with vertical red lines and labels for before and after therapy.

        Parameters:
        ----------
        column1 : str
            Column name for the first plot.
        column2 : str
            Column name for the second plot.
        title_prefix_1 : str
            Title for the first plot.
        title_prefix_2 : str
            Title for the second plot.
        xlabel : str
            Label for the x-axis (used for both plots).
        ylabel : str
            Label for the y-axis (used for both plots).
        """
        # First plot
        value_counts_1 = self.df[column1].value_counts()
        x_values_1 = value_counts_1.index
        heights_1 = value_counts_1.values
        total_values_1 = sum(heights_1)

        # Second plot
        value_counts_2 = self.df[column2].value_counts()
        x_values_2 = value_counts_2.index
        heights_2 = value_counts_2.values
        total_values_2 = sum(heights_2)

        # Create side-by-side bar plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, dpi=300)

        # Plotting the first bar plot
        ax1.bar(x_values_1, heights_1)
        ax1.set_ylabel("Count", fontsize=18)
        ax1.set_title(f"Pocket depth before therapy (n={total_values_1})", fontsize=18)
        ax1.set_yticks(np.arange(0, 90001, 10000))
        ax1.set_xticks(np.arange(0, 12.1, 1))
        ax1.tick_params(axis="both", labelsize=16)

        # Adding vertical dotted red lines at 3 and 6 for the first plot
        ax1.axvline(x=3.5, color="red", linestyle="--")
        ax1.axvline(x=5.5, color="red", linestyle="--")
        ax1.axhline(y=5000, linestyle="--")
        ax1.axhline(y=10000, linestyle="--")
        ax1.axhline(y=30000, linestyle="--")

        # Plotting the second bar plot
        ax2.bar(x_values_2, heights_2)
        ax2.set_title(f"Pocket depth after therapy (n={total_values_2})", fontsize=18)
        ax2.tick_params(axis="both", labelsize=16)

        # Adding vertical dotted red lines at 3 and 6 for the second plot
        ax2.axvline(x=3.5, color="red", linestyle="--")
        ax2.axvline(x=5.5, color="red", linestyle="--")
        ax2.axhline(y=5000, linestyle="--")
        ax2.axhline(y=10000, linestyle="--")
        ax2.axhline(y=30000, linestyle="--")

        # Set the xlabel at the figure level and center it
        fig.text(0.55, 0, "Pocket Depth", ha="center", fontsize=18)
        plt.tight_layout()
        plt.show()

    def pocket_group_comparison(self, column_before, column_after):
        """
        Creates a side-by-side bar plot comparing two variables (before and after therapy).

        Parameters:
        ----------
        column_before : str
            Column name for the first bar plot (before therapy).
        column_after : str
            Column name for the second bar plot (after therapy).
        xlabel : str
            Label for the x-axis.
        ylabel : str
            Label for the y-axis.
        """
        # Data for first plot
        value_counts = self.df[column_before].value_counts()
        x_values = value_counts.index
        heights = value_counts.values
        total_values = sum(heights)

        # Data for second plot
        value_counts2 = self.df[column_after].value_counts()
        x_values2 = value_counts2.index
        heights2 = value_counts2.values
        total_values2 = sum(heights2)

        # Plotting the bar plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, dpi=300)

        # First plot
        bars1 = ax1.bar(x_values, heights)
        ax1.set_ylabel("Count", fontsize=18)
        ax1.set_title(f"Pocket depth before therapy (n={total_values})", fontsize=18)
        ax1.set_yticks(np.arange(0, 90001, 10000))
        ax1.set_xticks(np.arange(0, 2.1, 1))
        ax1.tick_params(axis="both", labelsize=16)

        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(
                "{}".format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
            )

        # Second plot
        bars2 = ax2.bar(x_values2, heights2)
        ax2.set_title(f"Pocket depth after therapy (n={total_values2})", fontsize=18)
        ax2.tick_params(axis="both", labelsize=16)

        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(
                "{}".format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
            )

        # Set xlabel at the figure level
        fig.text(0.55, 0, "Pocket Depth", ha="center", fontsize=18)
        plt.tight_layout()
        plt.show()

    def histogram_2d(self, column_before, column_after):
        """
        Creates a 2D histogram plot based on two columns.

        Parameters:
        ----------
        column_before : str
            Column name for pocket depth before therapy.
        column_after : str
            Column name for pocket depth after therapy.
        """
        # Create a 2D histogram
        heatmap, xedges, yedges = np.histogram2d(self.df[column_before], self.df[column_after], bins=(12, 12))

        # Plot heatmap
        plt.figure(figsize=(10, 8), dpi=300)
        plt.imshow(heatmap.T, origin="lower", cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Frequency")

        plt.xlabel("Pocket depth before therapy", fontsize=20)
        plt.ylabel("Pocket depth after therapy", fontsize=20)
        plt.xticks(np.arange(12), np.arange(1, 13), fontsize=18)
        plt.yticks(np.arange(12), np.arange(1, 13), fontsize=18)

        # Add red lines
        plt.plot([-0.5, 2.5], [2.5, 2.5], "r--", lw=2)  # Horizontal line
        plt.plot([2.5, 2.5], [-0.5, 2.5], "r--", lw=2)  # Vertical line
        plt.show()
