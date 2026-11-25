import pandas as pd
from catboost import Pool
from pathlib import Path
from .features import featurize

def read_table(
    path: Path,
    columns: list | None = None,
) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def get_data(
    path: Path | list[Path] | None = None,
    columns: list | None = None,
) -> pd.DataFrame:
    if path is None:
        path = Path(
            "/Users/wehadgoodtimes/.cache/kagglehub/datasets"
            "/nickpudovkin/avito-cup/versions/1/competition/competition/train/"
            "impressions/part-00000-e3428b2d-e8b5-4109-8f1d-0bb6189ed749-c000.zstd.parquet"
        )

    if isinstance(path, list):
        return pd.concat(
            [read_table(p, columns) for p in path],
            ignore_index=True,
        )

    return read_table(path, columns)


def get_pool(config: dict) -> Pool:
    features = config["features"]
    target = config["target"]
    group_id = config.get("group_id")

    columns = features + [target]

    if group_id is not None:
        columns.append(group_id)

    data = get_data(
        path=config.get("path"),
        columns=config.get("columns", columns),
    )

    data = featurize(data)

    return Pool(
        data=data[features],
        label=data[target],
        group_id=data[group_id] if group_id is not None else None,
        cat_features=config.get("cat_features"),
        embedding_features=config.get("embedding_features"),
    )
