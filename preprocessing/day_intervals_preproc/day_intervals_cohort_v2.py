import datetime
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import importlib
import disease_cohort
importlib.reload(disease_cohort)
import disease_cohort
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
os.makedirs("./data/cohort", exist_ok=True)

def get_visit_pts(mimic4_path, group_col, visit_col, admit_col, disch_col, adm_visit_col, use_admn, disease_label, use_ICU):
    """
    Combines the MIMIC-IV core/patients table information with either the icu/icustays or core/admissions data.
    """
    file_path = "icu/icustays.csv.gz" if use_ICU else "hosp/admissions.csv.gz"
    visit = pd.read_csv(
        mimic4_path + file_path, 
        compression='gzip', 
        header=0, 
        parse_dates=[admit_col, disch_col]
    )
    
    if not use_ICU:
        visit['los'] = (visit[disch_col] - visit[admit_col]).dt.days
        if use_admn:
            # remove hospitalizations with a death; impossible for readmission for such visits
            visit = visit.loc[visit.hospital_expire_flag == 0]

    if use_admn:
        pts = pd.read_csv(
            mimic4_path + "hosp/patients.csv.gz", 
            compression='gzip', 
            header=0, 
            usecols=['subject_id', 'dod'], 
            parse_dates=['dod']
        )
        visit = visit.merge(pts, how='inner', left_on='subject_id', right_on='subject_id')
        visit = visit.loc[(visit.dod.isna()) | (visit.dod >= visit[disch_col])]

    if disease_label:
        hids = disease_cohort.extract_diag_cohort(visit['hadm_id'], disease_label, mimic4_path)
        visit = visit[visit['hadm_id'].isin(hids['hadm_id'])]
        print(f"[ READMISSION DUE TO {disease_label} ]")
    
    pts = pd.read_csv(
        mimic4_path + "hosp/patients.csv.gz", 
        compression='gzip', 
        header=0, 
        usecols=[group_col, 'anchor_year', 'anchor_age', 'anchor_year_group', 'dod', 'gender']
    )
    pts['yob'] = pts['anchor_year'] - pts['anchor_age']
    pts['min_valid_year'] = pts['anchor_year'] + (2019 - pts['anchor_year_group'].str.slice(start=-4).astype(int))
    
    if use_ICU:
        visit_pts = visit[[group_col, visit_col, adm_visit_col, admit_col, disch_col, 'los']].merge(
            pts, how='inner', left_on=group_col, right_on=group_col
        )
    else:
        visit_pts = visit[[group_col, visit_col, admit_col, disch_col, 'los']].merge(
            pts, how='inner', left_on=group_col, right_on=group_col
        )

    visit_pts['Age'] = visit_pts['anchor_age']
    visit_pts = visit_pts.loc[visit_pts['Age'] >= 18]

    eth = pd.read_csv(
        mimic4_path + "hosp/admissions.csv.gz", 
        compression='gzip', 
        header=0, 
        usecols=['hadm_id', 'insurance', 'race'], 
        index_col=None
    )
    visit_pts= visit_pts.merge(eth, how='inner', left_on='hadm_id', right_on='hadm_id')
    return visit_pts if use_ICU else visit_pts.dropna(subset=['min_valid_year'])

def partition_by_los(df, los, group_col, admit_col,disch_col):
    cohort = df.dropna(subset=[admit_col, disch_col, 'los'])
    pos_cohort = cohort[cohort['los'] > los].fillna(0)
    neg_cohort = cohort[cohort['los'] <= los].fillna(0)

    pos_cohort['label'] = 1
    neg_cohort['label'] = 0
    print("[ LOS LABELS FINISHED ]")
    return pd.concat([pos_cohort, neg_cohort]).sort_values(by=[group_col, admit_col])

def partition_by_readmit(df, gap, group_col, admit_col, disch_col, valid_col):
    case_list, ctrl_list = [], []
    grouped = df.sort_values(by=[group_col, admit_col]).groupby(group_col)
    
    for _, group in tqdm(grouped):
        if group.shape[0] <= 1:
            ctrl_list.append(group.iloc[0])
        else:
            for idx in range(group.shape[0] - 1):
                visit_time = group.iloc[idx][disch_col]
                readmissions = group.loc[
                    (group[admit_col] > visit_time) & 
                    (group[admit_col] - visit_time <= gap)
                ]
                if readmissions.shape[0] >= 1:
                    case_list.append(group.iloc[idx])
                else:
                    ctrl_list.append(group.iloc[idx])
            ctrl_list.append(group.iloc[-1])

    case = pd.DataFrame(case_list)
    ctrl = pd.DataFrame(ctrl_list)

    print("[ READMISSION LABELS FINISHED ]")
    return case, ctrl

def partition_by_mort(df,group_col,admit_col, disch_col, death_col):
    cohort = df.loc[(~df[admit_col].isna()) & (~df[disch_col].isna())]
    cohort['label']=0
    pos_cohort=cohort[~cohort[death_col].isna()]
    neg_cohort=cohort[cohort[death_col].isna()]
    neg_cohort=neg_cohort.fillna(0)
    pos_cohort=pos_cohort.fillna(0)
    pos_cohort[death_col] = pd.to_datetime(pos_cohort[death_col])
    pos_cohort['label'] = np.where((pos_cohort[death_col] >= pos_cohort[admit_col]) & (pos_cohort[death_col] <= pos_cohort[disch_col]),1,0)
    pos_cohort['label'] = pos_cohort['label'].astype("Int32")
    cohort=pd.concat([pos_cohort,neg_cohort], axis=0)
    cohort=cohort.sort_values(by=[group_col,admit_col])
    print("[ MORTALITY LABELS FINISHED ]")
    return cohort

def get_case_ctrls(df, gap, group_col, visit_col, admit_col, disch_col, valid_col, death_col, use_mort=False, use_admn=False, use_los=False):
    if use_mort:
        return partition_by_mort(df,group_col,admit_col, disch_col, death_col), pd.DataFrame()
    elif use_admn:
        case, ctrl = partition_by_readmit(df, datetime.timedelta(days=gap), group_col, admit_col, disch_col, valid_col)
        case['label'] = 1
        ctrl['label'] = 0
        return pd.concat([case, ctrl]), pd.DataFrame()
    elif use_los:
        return partition_by_los(df, gap, group_col, admit_col, disch_col), pd.DataFrame()

def extract_data(use_ICU, label, time, icd_code, root_dir, disease_label, cohort_output=None, summary_output=None):
    """
    Extracts cohort data and summary from MIMIC-IV data based on provided parameters.
    """
    cohort_output = cohort_output or f"cohort_{use_ICU.lower()}_{label.lower().replace(' ', '_')}_{time}_{disease_label}"
    summary_output = summary_output or f"summary_{use_ICU.lower()}_{label.lower().replace(' ', '_')}_{time}_{disease_label}"
    
    print(f"EXTRACTING FOR: | {use_ICU.upper()} | {label.upper()} {'DUE TO ' + disease_label.upper() if disease_label else ''} | {'ADMITTED DUE TO ' + icd_code.upper() if icd_code != 'No Disease Filter' else ''} | {time} |")
    ICU = use_ICU
    use_mort = label == "Mortality"
    use_admn = label == "Readmission"
    use_los = label == "Length of Stay"
    los = time if use_los else 0
    use_ICU = use_ICU == "ICU"

    group_col = 'subject_id'
    visit_col = 'stay_id' if use_ICU else 'hadm_id'
    admit_col = 'intime' if use_ICU else 'admittime'
    disch_col = 'outtime' if use_ICU else 'dischtime'
    death_col = 'dod'
    adm_visit_col = 'hadm_id' if use_ICU else None

    pts = get_visit_pts(
        mimic4_path=root_dir + "/mimiciv/2.2/",
        group_col=group_col,
        visit_col=visit_col,
        admit_col=admit_col,
        disch_col=disch_col,
        adm_visit_col=adm_visit_col,
        use_admn=use_admn,
        disease_label=disease_label,
        use_ICU=use_ICU
    )
    cols = [group_col, visit_col, admit_col, disch_col, 'Age', 'gender', 'ethnicity', 'insurance', 'label']
    if use_ICU:
        cols.append(adm_visit_col)
    if use_mort:
        cols.append(death_col)

    cohort, invalid = get_case_ctrls(pts, time, group_col, visit_col, admit_col, disch_col, 'min_valid_year', death_col, use_mort, use_admn, use_los)

    if icd_code != "No Disease Filter":
        hids = disease_cohort.extract_diag_cohort(cohort['hadm_id'], icd_code, f"{root_dir}/mimiciv/2.2/")
        cohort = cohort[cohort['hadm_id'].isin(hids['hadm_id'])]
        cohort_output += f"_{icd_code}"
        summary_output += f"_{icd_code}"

    cohort = cohort.rename(columns={"race": "ethnicity"})
    cohort[cols].to_csv(f"./data/cohort/{cohort_output}.csv.gz", index=False, compression='gzip')
    print("[ COHORT SUCCESSFULLY SAVED ]")

    summary = "\n".join([
        f"{label} FOR {ICU} DATA",
        f"# Admission Records: {cohort.shape[0]}",
        f"# Patients: {cohort[group_col].nunique()}",
        f"# Positive cases: {cohort[cohort['label'] == 1].shape[0]}",
        f"# Negative cases: {cohort[cohort['label'] == 0].shape[0]}"
    ])

    print(summary)
    with open(f"./data/cohort/{summary_output}.txt", "w") as text_file:
        text_file.write(summary)
    print("[ SUMMARY SUCCESSFULLY SAVED ]")
    return cohort_output

def extract_imputation_data(use_ICU, root_dir, cohort_output=None, summary_output=None):
    cohort_output = cohort_output or f"cohort_{use_ICU.lower()}_imputation"
    print(f"EXTRACTING FOR: | {use_ICU.upper()} | IMPUTATION |")

    use_ICU = use_ICU == "ICU"
    group_col = 'subject_id'
    #visit_col = 'stay_id' if use_ICU else 'hadm_id'
    admit_col = 'intime' if use_ICU else 'admittime'
    disch_col = 'outtime' if use_ICU else 'dischtime'
    #death_col = 'dod'
    #adm_visit_col = 'hadm_id' if use_ICU else None
    mimic4_path = root_dir + "/mimiciv/2.2/"
    file_path = "icu/icustays.csv.gz" if use_ICU else "hosp/admissions.csv.gz"
    visit = pd.read_csv(
        mimic4_path + file_path, 
        compression='gzip', 
        header=0, 
        parse_dates=[admit_col, disch_col]
    )
    
    if not use_ICU:
        visit['los'] = (visit[disch_col] - visit[admit_col])
    
    pts = pd.read_csv(
        mimic4_path + "hosp/patients.csv.gz", 
        compression='gzip', 
        header=0
    )

    pts['yob'] = pts['anchor_year'] - pts['anchor_age']
    pts['min_valid_year'] = pts['anchor_year'] + (2019 - pts['anchor_year_group'].str.slice(start=-4).astype(int))
    visit_pts = visit.merge(pts, how='inner', left_on=group_col, right_on=group_col)
    
    if use_ICU:
        eth = pd.read_csv(
        mimic4_path + "hosp/admissions.csv.gz", 
        compression='gzip', 
        header=0, 
        index_col=None
        )
        visit_pts= visit_pts.merge(eth, how='inner', left_on='hadm_id', right_on='hadm_id')
    visit_pts.to_csv(f"./data/cohort/{cohort_output}.csv.gz", index=False, compression='gzip')
    return cohort_output
    #return visit_pts if use_ICU else visit_pts.dropna(subset=['min_valid_year'])