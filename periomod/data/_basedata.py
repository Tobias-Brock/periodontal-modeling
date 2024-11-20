from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Optional, Union
import warnings

import pandas as pd

from ..base import BaseConfig


class BaseLoader(BaseConfig, ABC):
    """Abstract base class for loading and saving processed data.

    This class provides a structure for loading and saving datasets and defines
    abstract methods that need to be implemented by subclasses. It includes methods
    for specifying data paths and filenames for loading and saving operations.

    Inherits:
        - `BaseConfig`: Provides configuration settings for data processing.
        - `ABC`: Specifies abstract methods for subclasses to implement.

    Abstract Methods:
        - `load_data`: Load processed data from the specified path and file.
        - `save_data`: Save processed data to the specified path and file.

    Example:
        ```
        loader = SomeConcreteLoader()
        data = loader.load_data(path=Path("/data"), name="processed_data.csv")
        loader.save_data(df=data, path=Path("/data"), name="saved_data.csv")
        ```
    """

    def __init__(self) -> None:
        """Initializes the BaseLoader, defining the `load_data` abstract method."""
        super().__init__()

    @abstractmethod
    def load_data(self, path: Path):
        """Loads the processed data from the specified path.

        Args:
            path (Path): Directory path for the processed data.
        """

    @abstractmethod
    def save_data(self, df: pd.DataFrame, path: Union[str, Path]) -> None:
        """Saves the processed data to the specified path as a CSV file.

        Args:
            df (pd.DataFrame): The processed DataFrame.
            path (Union[str, Path]): Path (including filename) where data will be saved.

        Raises:
            ValueError: If the DataFrame is empty.
        """
        path = Path(path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if df.empty:
            raise ValueError("Data must be processed before saving.")
        if path.suffix != ".csv":
            path = path.with_suffix(".csv")
        os.makedirs(path.parent, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Data saved to {path}")


class BaseProcessor(BaseLoader, ABC):
    """Abstract base class defining essential data processing methods.

    This class provides core processing capabilities such as loading and saving
    data, along with abstract methods that must be implemented by any subclass.
    These methods include data imputation, feature creation, and outcome variable
    generation for specialized data processing.

    Inherits:
        - `BaseLoader`: Provides loading and saving capabilities for processed data.
        - `ABC`: Specifies abstract methods for subclasses to implement.

    Args:
        behavior (bool): If True, includes behavior columns in the data processing.

    Attributes:
        behavior (bool): Flag indicating whether to include behavior columns
            during data processing.

    Methods:
        load_data: Load processed data from the specified path and file.
        save_data: Save processed data to the specified path and file.

    Abstract Methods:
        - `impute_missing_values`: Impute missing values in the DataFrame.
        - `create_tooth_features`: Generate features related to tooth data.
        - `create_outcome_variables`: Create outcome variables for analysis.
        - `process_data`: Clean, impute, and scale the data.
    """

    def __init__(self, behavior: bool) -> None:
        """Initializes the BaseProcessor with behavior flag."""
        super().__init__()
        self.behavior = behavior

    def load_data(
        self,
        path: Union[str, Path] = Path("data/raw/raw_data.xlsx"),
    ) -> pd.DataFrame:
        """Loads the dataset and validates required columns.

        Args:
            path (str, optional): Directory where dataset is located.
                Defaults to Path("data/raw/raw_data.xlsx").

        Returns:
            pd.DataFrame: The loaded DataFrame.

        Raises:
            ValueError: If any required columns are missing.
        """
        path = Path(path)
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_excel(path, header=[1])

        actual_columns_lower = {col.lower(): col for col in df.columns}
        required_columns_lower = [col.lower() for col in self.required_columns]

        missing_columns = [
            col for col in required_columns_lower if col not in actual_columns_lower
        ]
        if missing_columns:
            missing_columns_names = [
                self.required_columns[required_columns_lower.index(col)]
                for col in missing_columns
            ]
            warnings.warn(
                f"Warning: Missing columns: {', '.join(missing_columns_names)}",
                stacklevel=2,
            )

        available_required_columns = [
            col for col in required_columns_lower if col in actual_columns_lower
        ]
        actual_required_columns = [
            actual_columns_lower[col] for col in available_required_columns
        ]

        if self.behavior:
            behavior_columns_lower = [
                col.lower() for col in self.behavior_columns["binary"]
            ] + [col.lower() for col in self.behavior_columns["categorical"]]
            missing_behavior_columns = [
                col for col in behavior_columns_lower if col not in actual_columns_lower
            ]
            if missing_behavior_columns:
                missing_behavior_names = [
                    col.capitalize() for col in missing_behavior_columns
                ]
                warnings.warn(
                    f"Warning: Missing cols: {', '.join(missing_behavior_names)}",
                    stacklevel=2,
                )

            available_behavior_columns = [
                col for col in behavior_columns_lower if col in actual_columns_lower
            ]
            actual_required_columns += [
                actual_columns_lower[col] for col in available_behavior_columns
            ]

        return df[actual_required_columns]

    def save_data(
        self,
        df: pd.DataFrame,
        path: Union[str, Path] = Path("data/processed/processed_data.csv"),
    ) -> None:
        """Saves the processed DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The processed DataFrame.
            path (str, optional): Directory where dataset is saved.
                Defaults to Path("data/processed/processed_data.csv".
        """
        super().save_data(df=df, path=path)

    @abstractmethod
    def impute_missing_values(self, df: pd.DataFrame):
        """Imputes missing values in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame with potential missing values.
        """

    @abstractmethod
    def create_tooth_features(self, df: pd.DataFrame):
        """Creates additional features related to tooth data.

        Args:
            df (pd.DataFrame): The DataFrame containing tooth data.
        """

    @abstractmethod
    def create_outcome_variables(self, df: pd.DataFrame):
        """Generates outcome variables for analysis.

        Args:
            df (pd.DataFrame): The DataFrame with original outcome variables.
        """

    @abstractmethod
    def process_data(self, df: pd.DataFrame):
        """Processes dataset with data cleaning, imputations and scaling.

        Args:
            df (pd.DataFrame): The input DataFrame.
        """


class BaseDataLoader(BaseLoader, ABC):
    """Abstract base class for loading, encoding, and scaling processed data.

    This class provides methods for loading and saving processed data, verifying
    encoded and scaled columns, and defines abstract methods for encoding and scaling
    that must be implemented by subclasses.

    Inherits:
        - `BaseLoader`: Provides loading and saving capabilities for processed data.
        - `ABC`: Specifies abstract methods for subclasses to implement.

    Args:
        task (str): Specifies the task column name.
        encoding (Optional[str]): Defines the encoding method for categorical columns.
            Options include 'one_hot', 'target', or None.
        encode (bool): If True, applies encoding to categorical columns.
        scale (bool): If True, applies scaling to numeric columns.

    Attributes:
        task (str): Task column name used in transformations.
        encoding (str): Encoding method specified for categorical columns.
        encode (bool): Flag to apply encoding to categorical columns.
        scale (bool): Flag to apply scaling to numeric columns.

    Methods:
        load_data: Load processed data from the specified path and file.
        save_data: Save processed data to the specified path and file.

    Abstract Methods:
        - `encode_categorical_columns`: Encodes categorical columns in the DataFrame.
        - `scale_numeric_columns`: Scales numeric columns in the DataFrame.
        - `transform_data`: Processes and transforms the data.
    """

    def __init__(
        self, task: str, encoding: Optional[str], encode: bool, scale: bool
    ) -> None:
        """Initializes the ProcessedDataLoader with the specified task column."""
        super().__init__()
        self.task = task
        self.encoding = encoding
        self.encode = encode
        self.scale = scale

    @staticmethod
    def load_data(
        path: Union[str, Path] = Path("data/processed/processed_data.csv"),
    ) -> pd.DataFrame:
        """Loads the processed data from the specified path, with lowercasing.

        Args:
            path (str): Directory path for the processed data.

        Returns:
            pd.DataFrame: Loaded DataFrame with lowercase column names.
        """
        path = Path(path)

        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return pd.read_csv(path).rename(columns=str.lower)

    def save_data(
        self,
        df: pd.DataFrame,
        path: Union[str, Path] = Path("data/training/training_data.csv"),
    ) -> None:
        """Saves the processed DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The processed DataFrame.
            path (str, optional): Directory where dataset is saved.
                Path("data/training/training_data.csv".
        """
        super().save_data(df=df, path=path)

    def _check_encoded_columns(self, df: pd.DataFrame) -> None:
        """Verifies that categorical columns were correctly one-hot or target encoded.

        Args:
            df (pd.DataFrame): The DataFrame to check.

        Raises:
            ValueError: If columns are not correctly encoded.
        """
        if self.encoding == "one_hot":
            cat_vars = [col for col in self.all_cat_vars if col in df.columns]

            for col in cat_vars:
                if col in df.columns:
                    raise ValueError(
                        f"Column '{col}' was not correctly one-hot encoded."
                    )
                matching_columns = [c for c in df.columns if c.startswith(f"{col}_")]
                if not matching_columns:
                    raise ValueError(f"No one-hot encoded columns for '{col}'.")
        elif self.encoding == "target":
            if "toothside" not in df.columns:
                raise ValueError("Target encoding for 'toothside' failed.")
        elif self.encoding is None:
            print("No encoding was applied.")
        else:
            raise ValueError(f"Invalid encoding '{self.encoding}'.")

    def _check_scaled_columns(self, df: pd.DataFrame) -> None:
        """Verifies that scaled columns are within expected ranges.

        Args:
            df (pd.DataFrame): The DataFrame to check.

        Raises:
            ValueError: If any columns are not correctly scaled.
        """
        if self.scale:
            for col in self.scale_vars:
                scaled_min = df[col].min()
                scaled_max = df[col].max()
                if scaled_min < -10 or scaled_max > 20:
                    raise ValueError(f"Column {col} is not correctly scaled.")

    @abstractmethod
    def encode_categorical_columns(self, df: pd.DataFrame):
        """Encodes categorical columns in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing categorical columns.
        """

    @abstractmethod
    def scale_numeric_columns(self, df: pd.DataFrame):
        """Scales numeric columns in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing numeric columns.
        """

    @abstractmethod
    def transform_data(self, df: pd.DataFrame):
        """Processes and transforms the data.

        Args:
            df (pd.DataFrame): The DataFrame to transform.
        """
