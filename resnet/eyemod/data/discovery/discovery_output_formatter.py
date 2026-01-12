import pandas as pd
import numpy as np
from typing import NamedTuple

from .etdrs_grid import ETDRSGrid


class DataColumn(NamedTuple):
    name_export: str
    name: str
    type: str
    include: bool = True
    unit: str = None


DISCOVERY_COLUMNS = [
    DataColumn("FILENAME", "filename", "string"),
    DataColumn("FILEPATH", "filepath", "string"),
    DataColumn("STUDYID", "patient", "string"),
    DataColumn("DOMAIN", "domain", "string"),
    DataColumn("USUBJID", "usubjid", "string", include=False),
    DataColumn("FOCID", "laterality", "string"),
    DataColumn("OESEQ", "oeseq", "string", include=False),
    DataColumn("OEGRPID", "oegrpid", "string", include=False),
    DataColumn("OELNKID", "oelnkid", "string", include=False),
    DataColumn("OELNKGRP", "oelnkgrp", "string", include=False),
    DataColumn("OETESTCD", "biomarker", "string"),
    DataColumn("OETEST", "biomarker long", "string", include=False),
    DataColumn("OECAT", "oecat", "string", include=False),
    DataColumn("OEORRES", "oeorres", "Float64", include=False),
    DataColumn("OEORRESU", "oeorresu", "string", include=False),
    DataColumn("OESTRESC", "oestresc", "string", include=False),
    DataColumn("OESTRESN", "oestresn", "string", include=False),
    DataColumn("OESTRESU", "oestresu", "string", include=False),
    DataColumn("OEXFN", "oexfn", "string", include=False),
    DataColumn("OELOC", "oeloc", "string", include=False),
    DataColumn("OELAT", "oelat", "string", include=False),
    DataColumn("OEMETHOD", "oemethod", "string", include=False),
    DataColumn("OEDTC", "date", "datetime64[ns, UTC]"),
    DataColumn("SERIESDTC", "seriesdtc", "string", include=False),
    DataColumn("STUDYDTC", "studydtc", "string", include=False),
    DataColumn("DEVICE", "device", "string"),
    DataColumn("PIXL1", "resolution_x", "Int64"),
    DataColumn("PIXL2", "slices", "Int64"),
    DataColumn("PIXL3", "resolution_y", "Int64"),
    DataColumn("RNG1", "ring_range_x", "Float64"),
    DataColumn("RNG2", "ring_range_y", "Float64"),
    DataColumn("RNG3", "rng3", "Float64", include=False),
    DataColumn("THICKNESS_BG", "thickness_bg", "Float64", unit="µm"),
    DataColumn("THICKNESS_S6", "thickness_s6", "Float64", unit="µm"),
    DataColumn("THICKNESS_N6", "thickness_n6", "Float64", unit="µm"),
    DataColumn("THICKNESS_I6", "thickness_i6", "Float64", unit="µm"),
    DataColumn("THICKNESS_T6", "thickness_t6", "Float64", unit="µm"),
    DataColumn("THICKNESS_S3", "thickness_s3", "Float64", unit="µm"),
    DataColumn("THICKNESS_N3", "thickness_n3", "Float64", unit="µm"),
    DataColumn("THICKNESS_I3", "thickness_i3", "Float64", unit="µm"),
    DataColumn("THICKNESS_T3", "thickness_t3", "Float64", unit="µm"),
    DataColumn("THICKNESS_C1", "thickness_c1", "Float64", unit="µm"),
    DataColumn("VOLUME_BG", "volume_bg", "Float64", unit="nL"),
    DataColumn("VOLUME_S6", "volume_s6", "Float64", unit="nL"),
    DataColumn("VOLUME_N6", "volume_n6", "Float64", unit="nL"),
    DataColumn("VOLUME_I6", "volume_i6", "Float64", unit="nL"),
    DataColumn("VOLUME_T6", "volume_t6", "Float64", unit="nL"),
    DataColumn("VOLUME_S3", "volume_s3", "Float64", unit="nL"),
    DataColumn("VOLUME_N3", "volume_n3", "Float64", unit="nL"),
    DataColumn("VOLUME_I3", "volume_i3", "Float64", unit="nL"),
    DataColumn("VOLUME_T3", "volume_t3", "Float64", unit="nL"),
    DataColumn("VOLUME_C1", "volume_c1", "Float64", unit="nL"),
    DataColumn("PROBABILITY", "probability", "string"),
    DataColumn("PROCESSOR_VERSION", "processor_version", "string", include=False),
    DataColumn("URL", "url", "string"),
]

# Holds the mapping between the biomarker the corresponding the value type
BIOMARKER_DICT = {
    "IRF_bin": "present",
    "SRF_bin": "present",
    "HF": "present",
    "DRUSEN": "present",
    "RPD": "present",
    "ERM": "present",
    "GA": "present",
    "ORA": "present",
    "FPED": "present",
    "BACKGROUND": "thickness",
    "RNFL": "thickness",
    "GCL+IPL": "thickness",
    "INL+OPL": "thickness",
    "ONL": "thickness",
    "PR+RPE": "thickness",
    "CC+CS": "thickness",
    "IRF": "volume",
    "SRF": "volume",
    "PED": "volume",
    "RT": "thickness",
}


class DiscoveryOutputFormatter:
    """Class to format raw discovery output.

    Args:
        workbook_id (str, optional): The ID of the workbook. Required to correct the URL to the images on discovery. Defaults to None.

    Returns:
        pd.DataFrame: The formatted discovery data.
    """

    def __init__(self, workbook_id: str = None):
        self.workbook_id = workbook_id

    def __call__(self, raw_data: pd.DataFrame):
        return self.format(raw_data)

    def format(self, raw_data: pd.DataFrame):
        formatted = raw_data.copy()

        formatted = self.rename_columns(formatted)
        formatted = self.drop_excluded_columns(formatted)
        formatted = self.format_datatypes(formatted)

        formatted = self.map_laterality(formatted)

        formatted = self.average_etdrs_grid(formatted, "thickness")
        formatted = self.average_etdrs_grid(formatted, "volume")
        formatted = self.aggregate_quadrants(formatted, "thickness")
        formatted = self.aggregate_quadrants(formatted, "volume")
        formatted = self.compute_biomarker_presence(formatted)

        if self.workbook_id:
            formatted = self.correct_url(formatted, workbook_id=self.workbook_id)

        formatted = self.make_biomarker_wide(formatted)
        return formatted

    @staticmethod
    def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=get_renaming_dict())
        return df

    @staticmethod
    def drop_excluded_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=get_excluded_columns())
        return df

    @staticmethod
    def format_datatypes(df: pd.DataFrame) -> pd.DataFrame:
        """Format datatypes of discovery data.

        Args:
            df (pd.DataFrame): Discovery data frame.

        Returns:
            pd.DataFrame: Discovery data frame with formatted datatypes.
        """
        datatype_dict = get_datatypes_dict()

        for col, datatype in datatype_dict.items():
            if col in df.columns:
                df[col] = df[col].astype(datatype)
        return df

    @staticmethod
    def average_etdrs_grid(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """Average ETDRS grid values. Compute average of ETDRS rings values
        and area weighted average of full ETDRS grid.

        Args:
            df (pd.DataFrame): Discovery data frame. Requires columns named with <feature_name>_<sector> e.g. thickness_c1.
            feature_name (str): Name of the feature e.g. thickness.

        Returns:
            pd.DataFrame: Data with averaged ETDRS grid values as new columns.
        """

        # select columns with feature values
        df_feature = df.filter(regex=feature_name)
        df_feature = df_feature.astype(float)

        # compute average of ETDRS rings values
        # regex pattern to match underscore, a letter and the number 1, 3 or 6
        df[f"{feature_name}_1"] = df_feature.filter(regex=r"_\w1").mean(
            axis=1, skipna=True
        )
        df[f"{feature_name}_3"] = df_feature.filter(regex=r"_\w3").mean(
            axis=1, skipna=True
        )
        df[f"{feature_name}_6"] = df_feature.filter(regex=r"_\w6").mean(
            axis=1, skipna=True
        )

        # # compute weighted average of full ETDRS grid
        df[f"{feature_name}_avg"] = (
            df[f"{feature_name}_1"] * ETDRSGrid().get_relative_area("center")
            + df[f"{feature_name}_3"] * ETDRSGrid().get_relative_area("inner")
            + df[f"{feature_name}_6"] * ETDRSGrid().get_relative_area("outer")
        )
        return df

    @staticmethod
    def aggregate_quadrants(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """Aggregate quadrant values. Compute average of quadrant values.

        Args:
            df (pd.DataFrame): Discovery data frame. Requires columns named with <feature_name>_<quadrant> e.g. thickness_c1.
            feature_name (str): Name of the feature e.g. thickness.

        Returns:
            pd.DataFrame: Data with averaged quadrant values as new columns.
        """
        area_inner = ETDRSGrid().get_area("inner") / 4
        area_outer = ETDRSGrid().get_area("outer") / 4
        area_full = area_inner + area_outer

        w_inner = area_inner / area_full
        w_outer = area_outer / area_full

        # select columns with feature values
        df_feature = df.filter(regex=feature_name)
        df_feature = df_feature.astype(float)

        # compute average of quadrant values
        for quadrant in ["s", "n", "i", "t"]:
            df[f"{feature_name}_{quadrant}"] = (
                df_feature.loc[:, f"{feature_name}_{quadrant}3"] * w_inner
                + df_feature.loc[:, f"{feature_name}_{quadrant}6"] * w_outer
            )
        return df

    @staticmethod
    def compute_biomarker_presence(
        df: pd.DataFrame, threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Compute the presence of biomarkers based on the probability of biomarkers being present in a slice.

        Args:
            df (pd.DataFrame): The input DataFrame with a column named "probability".
            threshold (float, optional): The threshold value for determining the presence of biomarkers. Defaults to 0.5.

        Returns:
            pd.DataFrame: The input DataFrame with an additional column named "present" with the proportion of slices with probability above threshold.
        """

        def presence_proportion(x):
            x = x.strip("[]")
            x = np.fromstring(x, sep=",")

            if len(x) > 0:
                above_threshold = x > threshold
                # represents the proportion of slices with probability above threshold
                return above_threshold.mean()
            else:
                return np.nan

        df["present"] = df["probability"].apply(presence_proportion)
        return df

    @staticmethod
    def correct_url(df: pd.DataFrame, workbook_id: str) -> pd.DataFrame:
        """
        Correct the URL column to point to the correct workbook.

        Args:
            df (pd.DataFrame): The input DataFrame with a column named "url".
            workbook_id (str): The ID of the workbook.

        Returns:
            pd.DataFrame: The input DataFrame with an updated column named "url".
        """
        incorrect_sequence = "foo/viewport?"
        correct_sequence = (
            f"http://hulk.artorg.unibe.ch/viewport?workbook={workbook_id}&"
        )
        df["url"] = df["url"].str.replace(
            incorrect_sequence, correct_sequence, regex=False
        )
        return df

    @staticmethod
    def make_multi_indexed(df: pd.DataFrame) -> pd.DataFrame:
        """
        Make a multi-indexed DataFrame with the columns "patient", "laterality" and "date".

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The input DataFrame with a multi-index.
        """
        df = df.set_index(["patient", "laterality", "date"])
        df = df.sort_index()
        return df

    @staticmethod
    def make_tidy(df: pd.DataFrame) -> pd.DataFrame:
        """
        Make a tidy DataFrame with the columns "patient", "domain", "biomarker", "laterality", "date", "device" and "value".

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The input DataFrame in tidy format.
        """

        df = df.melt(
            id_vars=["patient", "date", "domain", "biomarker", "laterality", "device"],
            var_name="feature",
            value_name="value",
        )
        return df

    @staticmethod
    def make_biomarker_wide(df: pd.DataFrame) -> pd.DataFrame:
        """
        Make a wide DataFrame with the columns "patient", "domain", "biomarker", "laterality", "date", "device" and "value".

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The input DataFrame in wide format.
        """

        multi_index_cols = ["patient", "laterality", "date"]

        # use multi-index to unambiguously identify the source of the biomarker data
        if df.index.names != multi_index_cols:
            df = df.set_index(multi_index_cols)

        # select only columns which hold biomarker related information
        df_biomarker = df.filter(regex=r"(biomarker|thickness|volume|present)")

        # drop columns with old format
        df = df.drop(columns=df_biomarker.columns)

        # drop probability column as it holds biomarker related information and is not needed at this point
        if "probability" in df.columns:
            df = df.drop(columns=["probability"])

        # remove duplicates such that only one row per image remains
        df = df.drop_duplicates()

        # pivot table such that each measurement-biomarker pairing has its own column (e.g. (thickness_c1, RNFL))
        df_biomarker = df_biomarker.pivot_table(
            index=multi_index_cols, columns="biomarker"
        )

        # drop unnecessary measurement-biomarker pairings according to BIOMARKER_DICT (e.g. (thickness_c1, IRF))
        measurements = df_biomarker.columns.get_level_values(0)
        biomarkers = df_biomarker.columns.get_level_values(1)
        columns_to_drop = []
        for measurement_name, biomarker_name in zip(measurements, biomarkers):
            if BIOMARKER_DICT.get(biomarker_name) not in measurement_name:
                columns_to_drop.append((measurement_name, biomarker_name))

        df_biomarker = df_biomarker.drop(columns=columns_to_drop)

        df_biomarker.columns = df_biomarker.columns.to_flat_index()

        df = df.merge(df_biomarker, left_index=True, right_index=True)

        return df

    @staticmethod
    def map_laterality(df: pd.DataFrame) -> pd.DataFrame:
        """Map laterality to 'L' and 'R'.

        Args:
            df (pd.DataFrame): The input DataFrame with a column named "laterality".

        Returns:
            pd.DataFrame: The input DataFrame with "laterality" mapped to 'L' and 'R'.
        """
        mapping = {"OS": "L", "OD": "R"}
        df["laterality"] = df["laterality"].map(mapping)
        return df


def get_renaming_dict():
    return {col.name_export: col.name for col in DISCOVERY_COLUMNS}


def get_excluded_columns():
    return [col.name for col in DISCOVERY_COLUMNS if not col.include]


def get_datatypes_dict():
    return {col.name: col.type for col in DISCOVERY_COLUMNS}
