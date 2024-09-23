def preproc_proc_imputation(dataset_path: str, cohort_path:str, time_col:str, anchor_col:str, dtypes: dict, usecols: list) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""

    def merge_module_cohort() -> pd.DataFrame:
        """Gets the initial module data with patients anchor year data and only the year of the charttime"""
        
        # read module w/ custom params
        module = pd.read_csv(dataset_path, compression='gzip', usecols=usecols, dtype=dtypes, parse_dates=[time_col]).drop_duplicates()

        # Only consider values in our cohort
        cohort = pd.read_csv(cohort_path, compression='gzip', parse_dates = ['admittime'])
        
        #print(module.head())
        #print(cohort.head())

        # merge module and cohort
        return module.merge(cohort[['hadm_id', 'admittime','dischtime']], how='inner', left_on='hadm_id', right_on='hadm_id')

    df_cohort = merge_module_cohort()
    df_cohort['proc_time_from_admit'] = df_cohort['chartdate'] - df_cohort['admittime']
    #df_cohort=df_cohort.dropna()
    # Print unique counts and value_counts
    print("# Unique ICD9 Procedures:  ", df_cohort.loc[df_cohort.icd_version == 9].icd_code.dropna().nunique())
    print("# Unique ICD10 Procedures: ",df_cohort.loc[df_cohort.icd_version == 10].icd_code.dropna().nunique())

    print("\nValue counts of each ICD version:\n", df_cohort.icd_version.value_counts())
    print("# Admissions:  ", df_cohort.hadm_id.nunique())
    print("Total number of rows: ", df_cohort.shape[0])

    # Only return module measurements within the observation range, sorted by subject_id
    return df_cohort