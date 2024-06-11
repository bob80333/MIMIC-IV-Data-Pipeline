import os
from glob import glob
from uuid import uuid1
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
from typing import Union, List, Tuple
import pandas as pd
import numpy as np

if not os.path.exists("./data/temp"):
    os.makedirs("./data/temp")

def hadm_imputer(
    charttime: pd._libs.tslibs.timestamps.Timestamp,
    hadm_old: Union[str, float],
    hadm_ids_w_timestamps: List[
        Tuple[
            str,
            pd._libs.tslibs.timestamps.Timestamp,
            pd._libs.tslibs.timestamps.Timestamp,
        ]
    ],
) -> Tuple[str, pd._libs.tslibs.timestamps.Timestamp]:
    if not np.isnan(hadm_old):
        hadm_old = int(hadm_old)
        admtime, dischtime = next(
            (adm_time, disch_time)
            for h_id, adm_time, disch_time in hadm_ids_w_timestamps
            if h_id == hadm_old
        )        
        return (
            hadm_old,
            admtime.strftime("%Y-%m-%d %H:%M:%S"),
            dischtime.strftime("%Y-%m-%d %H:%M:%S"),
        )

    hadm_ids_w_timestamps = [
        [
            hadm_id,
            admittime.strftime("%Y-%m-%d %H:%M:%S"),
            dischtime.strftime("%Y-%m-%d %H:%M:%S"),
            charttime.normalize() - admittime.normalize(),
            charttime.normalize() - dischtime.normalize(),
        ]
        for hadm_id, admittime, dischtime in hadm_ids_w_timestamps
    ]
    
    hadm_ids_w_timestamps = [
        x for x in hadm_ids_w_timestamps if x[3].days >= 0 and x[4].days <= 0
    ]
    
    hadm_ids_w_timestamps = sorted(hadm_ids_w_timestamps, key=lambda x: x[3])
    
    if not hadm_ids_w_timestamps:
        return None, None, None
    
    return hadm_ids_w_timestamps[0][:3]

def impute_missing_hadm_ids(
    lab_table: pd.DataFrame, subject_hadm_admittime_tracker: defaultdict
) -> str:
    list_rows_lab = []
    all_lab_cols = lab_table.columns
    for row in lab_table.itertuples():
        existing_data = {col_name: getattr(row, col_name) for col_name in all_lab_cols}
        new_hadm_id, new_admittime, new_dischtime = hadm_imputer(
            row.charttime,
            row.hadm_id,
            subject_hadm_admittime_tracker.get(row.subject_id, []),
        )
        existing_data["hadm_id_new"] = new_hadm_id
        existing_data["admittime"] = new_admittime
        existing_data["dischtime"] = new_dischtime
        list_rows_lab.append(existing_data)
    
    tab_name = f"./data/temp/{str(uuid1())}.csv"
    pd.DataFrame(list_rows_lab).to_csv(tab_name, index=False)
    return tab_name

def impute_hadm_ids(
    lab_table: Union[str, pd.DataFrame], admission_table: Union[str, pd.DataFrame]
) -> pd.DataFrame:
    if isinstance(lab_table, str):
        lab_table = pd.read_csv(lab_table)
    if isinstance(admission_table, str):
        admission_table = pd.read_csv(admission_table)
    lab_table["charttime"] = pd.to_datetime(lab_table.charttime)
    admission_table["admittime"] = pd.to_datetime(admission_table.admittime)
    admission_table["dischtime"] = pd.to_datetime(admission_table.dischtime)

    subject_hadm_admittime_tracker = defaultdict(list)
    for row in admission_table.itertuples():
        subject_hadm_admittime_tracker[row.subject_id].append([row.hadm_id, row.admittime, row.dischtime])

    lab_table_chunks = np.array_split(lab_table, 8)
    impute_missing_hadm_ids_w_lookup = partial(impute_missing_hadm_ids, subject_hadm_admittime_tracker=subject_hadm_admittime_tracker)
    with Pool(8) as p:
        result_files = p.map(impute_missing_hadm_ids_w_lookup, lab_table_chunks)

    # Load all intermediate CSV files and concatenate
    lab_tab = pd.concat([pd.read_csv(file) for file in result_files], ignore_index=True)

    # Clean up intermediate CSV files
    for file in result_files:
        os.remove(file)
    
    return lab_tab

