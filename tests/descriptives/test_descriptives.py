"""Tests for descriptives module."""

from unittest.mock import patch

import pandas as pd
import pytest

from periomod.descriptives._descriptives import DescriptivesPlotter


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing.

    Returns:
        pd.DataFrame: A sample dataset with depth measurements, pocket closure,
        and outcome variables.
    """
    return pd.DataFrame({
        "depth_before": [3, 4, 5, 6, 4, 3],
        "depth_after": [2, 3, 4, 5, 3, 2],
        "pocketclosure": [1, 0, 1, 0, 1, 1],
        "outcome_variable": [0, 1, 0, 1, 0, 0],
    })


def test_descriptives_plotter_initialization(sample_dataframe):
    """Test initialization of DescriptivesPlotter."""
    plotter = DescriptivesPlotter(sample_dataframe)
    assert isinstance(plotter.df, pd.DataFrame)


@patch("matplotlib.pyplot.show")
def test_plt_matrix(mock_show, sample_dataframe):
    """Test the plt_matrix method."""
    plotter = DescriptivesPlotter(sample_dataframe)
    with patch("matplotlib.pyplot.savefig") as mock_savefig:
        plotter.plt_matrix(
            vertical="depth_before",
            horizontal="depth_after",
            name="test_plot",
            save=True,
        )
        mock_savefig.assert_called_once_with("test_plot.svg", format="svg", dpi=300)
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_pocket_comparison(mock_show, sample_dataframe):
    """Test the pocket_comparison method."""
    plotter = DescriptivesPlotter(sample_dataframe)
    with patch("matplotlib.pyplot.savefig") as mock_savefig:
        plotter.pocket_comparison(
            col1="depth_before",
            col2="depth_after",
            name="test_pocket_comparison",
            save=True,
        )
        mock_savefig.assert_called_once_with(
            "test_pocket_comparison.svg", format="svg", dpi=300
        )
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_pocket_group_comparison(mock_show, sample_dataframe):
    """Test the pocket_group_comparison method."""
    plotter = DescriptivesPlotter(sample_dataframe)
    with patch("matplotlib.pyplot.savefig") as mock_savefig:
        plotter.pocket_group_comparison(
            col_before="depth_before",
            col_after="depth_after",
            name="test_pocket_group_comparison",
            save=True,
        )
        mock_savefig.assert_called_once_with(
            "test_pocket_group_comparison.svg", format="svg", dpi=300
        )
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_histogram_2d(mock_show, sample_dataframe):
    """Test the histogram_2d method."""
    plotter = DescriptivesPlotter(sample_dataframe)
    with patch("matplotlib.pyplot.savefig") as mock_savefig:
        plotter.histogram_2d(
            col_before="depth_before",
            col_after="depth_after",
            name="test_histogram_2d",
            save=True,
        )
        mock_savefig.assert_called_once_with(
            "test_histogram_2d.svg", format="svg", dpi=300
        )
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_outcome_descriptive(mock_show, sample_dataframe):
    """Test the outcome_descriptive method."""
    plotter = DescriptivesPlotter(sample_dataframe)
    with patch("matplotlib.pyplot.savefig") as mock_savefig:
        plotter.outcome_descriptive(
            outcome="outcome_variable",
            title="Outcome Plot",
            name="test_outcome_descriptive",
            save=True,
        )
        mock_savefig.assert_called_once_with(
            "test_outcome_descriptive.svg", format="svg", dpi=300
        )
    mock_show.assert_called_once()


def test_missing_name_when_save():
    """Test that ValueError is raised when name is None and save is True."""
    sample_df = pd.DataFrame({
        "depth_before": [3, 4],
        "depth_after": [2, 3],
    })
    plotter = DescriptivesPlotter(sample_df)
    with pytest.raises(
        ValueError, match="'name' argument required when 'save' is True."
    ):
        plotter.plt_matrix(vertical="depth_before", horizontal="depth_after", save=True)


def test_invalid_normalize_value():
    """Test that ValueError is raised when normalize parameter is invalid."""
    sample_df = pd.DataFrame({
        "depth_before": [3, 4],
        "depth_after": [2, 3],
    })
    plotter = DescriptivesPlotter(sample_df)
    with pytest.raises(
        ValueError, match="Invalid value for 'normalize'. Use 'rows' or 'columns'."
    ):
        plotter.plt_matrix(
            vertical="depth_before", horizontal="depth_after", normalize="invalid"
        )
