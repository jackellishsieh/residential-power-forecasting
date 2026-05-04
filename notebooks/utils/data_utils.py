import pickle

import pandas as pd


def load_split(path: str) -> pd.DataFrame:
    """Load a train/test split pickle and return a preprocessed DataFrame.

    The pickle maps ``home_id -> (has_ev, city, df)``, where ``df`` has columns
    ``localminute``, ``load``, ``car1``, ``charge_state``.

    Returns a flat DataFrame with columns:
        home_id, has_ev, city, total_load, ev_load, charge_state,
        day, time, time_index

    Only full 96-interval days (complete 15-min coverage) are retained.
    """
    raw = pickle.load(open(path, "rb"))
    df = pd.concat(
        [frame.assign(home_id=home_id, has_ev=has_ev, city=city)
         for home_id, (has_ev, city, frame) in raw.items()]
    ).reset_index()
    df.rename(columns={"car1": "ev_load", "load": "total_load"}, inplace=True)
    df["day"] = df["localminute"].dt.normalize()
    df["time"] = df["localminute"].dt.strftime("%H:%M")
    df = df.drop(columns="localminute")
    df["time"] = pd.to_timedelta(df["time"] + ":00")
    df["time_index"] = (df["time"].dt.total_seconds() // (15 * 60)).astype(int)
    df["charge_state"] = df["charge_state"].astype("int")

    full_day_mask = (
        df.groupby(["home_id", "day"])["time_index"]
        .transform(lambda x: x.nunique() == 96)
    )
    df = df[full_day_mask]
    return df
