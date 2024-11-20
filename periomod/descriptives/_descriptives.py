from typing import Optional

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


class DescriptivesPlotter:
    """Class for creating various descriptive plots based on periodontal data.

    This class provides methods for visualizing data through heatmaps, bar plots,
    2D histograms, and other descriptive plots to analyze pocket depth and therapy
    outcomes.

    Args:
        df (pd.DataFrame): DataFrame containing the data for plotting.

    Attributes:
        df (pd.DataFrame): Stores the input DataFrame for use in plotting.

    Methods:
        plt_matrix: Plots a heatmap/confusion matrix based on two columns.
        pocket_comparison: Creates bar plots to compare values before
            and after therapy.
        pocket_group_comparison: Generates side-by-side bar plots for pocket
            depth categories before and after therapy.
        histogram_2d: Creates a 2D histogram plot based on two columns, visualizing
            values before and after therapy.
        outcome_descriptive: Creates a bar plot for an outcome variable, useful
            for examining therapy outcomes.

    Example:
        ```
        from periomod.data import ProcessedDataLoader
        from periomod.descriptives import DescriptivesPlotter

        df = dataloader.load_data(path="data/processed/processed_data.csv")

        # instantiate plotter with dataframe
        plotter = DescriptivesPlotter(df)
        plotter.plt_matrix(vertical="pdgrouprevaluation", horizontal="pdgroupbase")
        plotter.pocket_comparison(col1="pdbaseline", col2="pdrevaluation")
        plotter.histogram_2d(col_before="pdbaseline", col_after="pdrevaluation")
        ```
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initializes DescriptivesPlotter with pd.DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing data for plotting.
        """
        self.df = df

    def plt_matrix(
        self,
        vertical: str,
        horizontal: str,
        x_label: str = "Pocket depth before therapy",
        y_label: str = "Pocket depth after therapy",
        name: Optional[str] = None,
        normalize: str = "rows",
        save: bool = False,
    ) -> None:
        """Plots a heatmap/confusion matrix.

        Args:
            vertical (str): Column name for the vertical axis.
            horizontal (str): Column name for the horizontal axis.
            x_label (str): Label for x-axis. Defaults to "Pocket depth before therapy".
            y_label (str): Label for y-axis. Defaults to "Pocket depth after therapy".
            name (str): Title of the plot and name for saving the plot.
            normalize (str, optional): Normalization method ('rows' or 'columns').
                Defaults to 'rows'.
            save (bool, optional): Save the plot as an SVG. Defaults to False.
        """
        vertical_data = self.df[vertical]
        horizontal_data = self.df[horizontal]
        cm = confusion_matrix(vertical_data, horizontal_data)
        custom_cmap = LinearSegmentedColormap.from_list(
            "teal_cmap", ["#FFFFFF", "#078294"]
        )

        if normalize == "rows":
            row_sums = cm.sum(axis=1)
            normalized_cm = (cm / row_sums[:, np.newaxis]) * 100
        elif normalize == "columns":
            col_sums = cm.sum(axis=0)
            normalized_cm = (cm / col_sums) * 100
        else:
            raise ValueError("Invalid value for 'normalize'. Use 'rows' or 'columns'.")

        plt.figure(figsize=(6, 4), dpi=300)
        sns.heatmap(
            normalized_cm,
            cmap=custom_cmap,
            fmt="g",
            linewidths=0.5,
            square=True,
            cbar_kws={"label": "Percent"},
        )

        for i in range(len(cm)):
            for j in range(len(cm)):
                if normalized_cm[i, j] > 50:
                    plt.text(
                        j + 0.5,
                        i + 0.5,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="white",
                    )
                else:
                    plt.text(j + 0.5, i + 0.5, cm[i, j], ha="center", va="center")

        title = "Data Overview"

        plt.title(title, fontsize=12)

        ax = plt.gca()
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        cbar = ax.collections[0].colorbar
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(1)

        ax.add_patch(
            Rectangle(
                (0, 0), cm.shape[1], cm.shape[0], fill=False, edgecolor="black", lw=2
            )
        )

        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)

        if save:
            if name is None:
                raise ValueError("'name' argument required when 'save' is True.")
            plt.savefig(name + ".svg", format="svg", dpi=300)

        plt.show()

    def pocket_comparison(
        self,
        col1: str,
        col2: str,
        title_1: str = "Pocket depth before therapy",
        title_2: str = "Pocket depth after therapy",
        name: Optional[str] = None,
        save: bool = False,
    ) -> None:
        """Creates two bar plots for comparing pocket depth before and after therapy.

        Args:
            col1 (str): Column name for the first plot (before therapy).
            col2 (str): Column name for the second plot (after therapy).
            title_1 (str): Label for x-axis. Defaults to "Pocket depth before therapy".
            title_2 (str): Label for y-axis. Defaults to "Pocket depth after therapy".
            name (str): Name for saving the plot.
            save (bool, optional): Save the plot as an SVG. Defaults to False.
        """
        value_counts_1 = self.df[col1].value_counts()
        x_values_1 = value_counts_1.index
        heights_1 = value_counts_1.values

        value_counts_2 = self.df[col2].value_counts()
        x_values_2 = value_counts_2.index
        heights_2 = value_counts_2.values

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(8, 5), sharex=True, sharey=True, dpi=300
        )

        ax1.bar(x_values_1, heights_1, edgecolor="black", color="#078294", linewidth=1)
        ax1.set_ylabel("Number of sites", fontsize=12)
        ax1.set_title(title_1, fontsize=12, pad=10)
        ax1.set_yticks(np.arange(0, 90001, 10000))
        ax1.set_xticks(np.arange(1, 12.5, 1))
        ax1.tick_params(axis="both", labelsize=12)

        ax1.axvline(x=3.5, color="red", linestyle="--", linewidth=1, alpha=0.3)
        ax1.axvline(x=5.5, color="red", linestyle="--", linewidth=1, alpha=0.3)

        ax1.grid(True, axis="y", color="black", linestyle="--", linewidth=1, alpha=0.3)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        ax1.tick_params(width=1)
        for spine in ax1.spines.values():
            spine.set_linewidth(1)

        ax2.bar(x_values_2, heights_2, edgecolor="black", color="#078294", linewidth=1)
        ax2.set_title(title_2, fontsize=12, pad=10)
        ax2.tick_params(axis="both", labelsize=12)

        ax2.axvline(x=3.5, color="red", linestyle="--", linewidth=1, alpha=0.3)
        ax2.axvline(x=5.5, color="red", linestyle="--", linewidth=1, alpha=0.3)

        ax2.grid(True, axis="y", color="black", linestyle="--", linewidth=1, alpha=0.3)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for spine in ax2.spines.values():
            spine.set_linewidth(1)

        ax2.tick_params(width=1)
        for spine in ax2.spines.values():
            spine.set_linewidth(1)

        fig.supxlabel("Pocket Depth [mm]", fontsize=12)
        plt.tight_layout()

        if save:
            if name is None:
                raise ValueError("'name' argument must required when 'save' is True.")
            plt.savefig(name + ".svg", format="svg", dpi=300)

        plt.show()

    def pocket_group_comparison(
        self,
        col_before: str,
        col_after: str,
        title_1: str = "Pocket depth before therapy",
        title_2: str = "Pocket depth after therapy",
        name: Optional[str] = None,
        save: bool = False,
    ) -> None:
        """Creates side-by-side bar plots for pocket depth before and after therapy.

        Args:
            col_before (str): Column name for the first plot (before therapy).
            col_after (str): Column name for the second plot (after therapy).
            title_1 (str): Label for x-axis. Defaults to "Pocket depth before therapy".
            title_2 (str): Label for y-axis. Defaults to "Pocket depth after therapy".
            name (str): Name for saving the plot.
            save (bool, optional): Save the plot as an SVG. Defaults to False.
        """
        value_counts = self.df[col_before].value_counts()
        x_values = value_counts.index
        heights = value_counts.values
        total_values = sum(heights)

        value_counts2 = self.df[col_after].value_counts()
        x_values2 = value_counts2.index
        heights2 = value_counts2.values
        total_values2 = sum(heights2)

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(8, 4), sharex=True, sharey=True, dpi=300
        )

        bars1 = ax1.bar(
            x_values, heights, edgecolor="black", color="#078294", linewidth=1
        )
        ax1.set_ylabel("Number of sites", fontsize=12)
        ax1.set_title(f"{title_1} (n={total_values})", fontsize=12, pad=10)
        ax1.set_yticks(np.arange(0, 100001, 10000))
        ax1.set_xticks(np.arange(0, 2.1, 1))

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        for spine in ax1.spines.values():
            spine.set_linewidth(1)
        ax1.tick_params(axis="both", labelsize=12)

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

        bars2 = ax2.bar(
            x_values2, heights2, edgecolor="black", color="#078294", linewidth=1
        )
        ax2.set_title(f"{title_2} (n={total_values2})", fontsize=12, pad=10)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for spine in ax2.spines.values():
            spine.set_linewidth(1)

        ax2.tick_params(axis="both", labelsize=12, width=1)

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
        fig.supxlabel("Pocket depth categories", fontsize=12)
        plt.tight_layout()

        if save:
            if name is None:
                raise ValueError("'name' argument must required when 'save' is True.")
            plt.savefig(name + ".svg", format="svg", dpi=300)

        plt.show()

    def histogram_2d(
        self,
        col_before: str,
        col_after: str,
        x_label: str = "Pocket depth before therapy [mm]",
        y_label: str = "Pocket depth after therapy [mm]",
        name: Optional[str] = None,
        save: bool = False,
    ) -> None:
        """Creates a 2D histogram plot based on two columns.

        Args:
            col_before (str): Column name for pocket depth before therapy.
            col_after (str): Column name for pocket depth after therapy.
            x_label (str): Label for x-axis. Defaults to
                "Pocket depth before therapy [mm]".
            y_label (str): Label for y-axis. Defaults to
                "Pocket depth after therapy [mm]".
            name (str): Name for saving the plot.
            save (bool, optional): Save the plot as an SVG. Defaults to False.
        """
        heatmap, _, _ = np.histogram2d(
            self.df[col_before], self.df[col_after], bins=(12, 12)
        )

        plt.figure(figsize=(8, 6), dpi=300)

        plt.imshow(heatmap.T, origin="lower", cmap="viridis", interpolation="nearest")
        cbar = plt.colorbar()
        cbar.set_label("Frequency", fontsize=12)
        cbar.outline.set_linewidth(1)

        plt.xlabel(xlabel=x_label, fontsize=12)
        plt.ylabel(ylabel=y_label, fontsize=12)
        plt.xticks(np.arange(12), np.arange(1, 13), fontsize=12)
        plt.yticks(np.arange(12), np.arange(1, 13), fontsize=12)

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(1)
        ax.tick_params(width=1)

        cbar.ax.tick_params(labelsize=12)

        plt.plot([-0.5, 2.5], [2.5, 2.5], "r--", lw=2)
        plt.plot([2.5, 2.5], [-0.5, 2.5], "r--", lw=2)

        if save:
            if name is None:
                raise ValueError("'name' argument must required when 'save' is True.")
            plt.savefig(name + ".svg", format="svg", dpi=300)

        plt.tight_layout()
        plt.show()

    def outcome_descriptive(
        self, outcome: str, title: str, name: Optional[str] = None, save: bool = False
    ) -> None:
        """Creates a bar plot for the outcome variable.

        Args:
            outcome (str): Column name for the outcome variable.
            title (str): Title of the plot.
            name (str): Filename for saving the plot.
            save (bool, optional): Save the plot as an SVG. Defaults to False.
        """
        df_temp = self.df
        if outcome == "improvement" and "pdgroupbase" in self.df.columns:
            df_temp = df_temp.query("pdgroupbase in [1, 2]")

        value_counts = df_temp[outcome].value_counts()
        x_values = value_counts.index.astype(str)
        heights = value_counts.values

        plt.figure(figsize=(6, 4), dpi=300)
        bars = plt.bar(
            x_values, heights, edgecolor="black", color="#078294", linewidth=1
        )
        plt.ylabel("Number of sites", fontsize=12)
        plt.title(title, fontsize=12, pad=10)

        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                "{}".format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
            )

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(1)
        ax.tick_params(width=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if save:
            if name is None:
                raise ValueError("'name' argument is required when 'save' is True.")
            plt.savefig(name + ".svg", format="svg", dpi=300)

        plt.tight_layout()
        plt.show()
