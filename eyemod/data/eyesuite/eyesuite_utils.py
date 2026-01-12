import numpy as np
import pandas as pd


def remove_extended_g_pattern_locations(row: pd.Series) -> pd.Series:
    """
    Remove the locations that are part of the extended, peripheral G pattern.
    Extended G pattern locations are defined as locations with x and y coordinates greater than 30.
    """
    RADIUS = 30
    coordinates = list(zip(row["x"], row["y"]))
    is_regular_g_pattern = [abs(x) < RADIUS and abs(y) < RADIUS for x, y in coordinates]

    for key in ["x", "y", "ph1", "ph2", "normative value"]:
        row[key] = np.asarray(row[key])[is_regular_g_pattern].tolist()

    row["testlocenumber"] = sum(is_regular_g_pattern)

    return row

def sort_g_pattern(row: pd.Series, target_order: list[tuple[int, int]]) -> pd.Series:
    """
    Sort the G pattern locations according to the target order.
    Target order is expected to be for an right eye
    """
    assert len(target_order) == len(row["x"]), "Target order and G pattern locations do not have the same length."

    x = np.asarray(row["x"]).astype(int)
    y = np.asarray(row["y"]).astype(int)
  
    laterality = row['eye']
    if laterality == 'OS': # Left eye
        x = -x

    # check that all target order and G pattern coordinates are the same
    assert set(target_order) == set(zip(x, y)), f"Coordinates in G pattern for {row['patient id']} are not in target order."
  
    coordinates = list(zip(x, y))
    indices = [coordinates.index(t) for t in target_order]

    for key in ["x", "y", "ph1", "ph2", "normative value"]:
        row[key] = np.asarray(row[key])[indices].tolist()

    return row

def drop_same_day_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop visual fields duplicates that were taken on the same day.
    Always keep the last examination taken on that day.
    """
    df = df.copy()
    df["date"] = df["examination"].dt.date
    df = df.sort_values("examination", ascending=False)
    df = df.drop_duplicates(subset=["patient id", "eye", "date"], keep="first")
    df = df.drop(columns=["date"])      
    return df
