import pandas as pd
import numpy as np

def featurize(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # логарифмы численных колонок
    data["f1_log"] = np.log1p(data["f1"])
    data["f2_log"] = np.log1p(data["f2"])
    data["f3_log"] = np.log1p(data["f3"])
    data["f4_log"] = np.log1p(data["f4"])
    data["f5_log"] = np.log1p(data["f5"])
    

    return data