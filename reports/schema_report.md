# Schema Report

- raw_dir: `C:\Users\gayat\Desktop\CSE\Sem7\6 PP2\PD_Project\data\raw`

- id_raw: `PATNO`

## demographics
- group: `demographics`
- file: `Demographics_07Sep2025.csv`
- exists: `True`
- shape (sample/full): `(7450, 29)`
- id `PATNO` present: `True`, unique: `True`, dupes: `0`

## diagnosis_events
- group: `diagnosis_events`
- file: `PD_Diagnosis_History_07Sep2025.csv`
- exists: `True`
- shape (sample/full): `(1939, 15)`
- id `PATNO` present: `True`, unique: `True`, dupes: `0`

## plasma_biomarkers[0]
- group: `plasma_biomarkers`
- file: `Current_Biospecimen_Analysis_Results_06Sep2025.csv`
- exists: `True`
- shape (sample/full): `(10000, 13)`
- id `PATNO` present: `True`, unique: `False`, dupes: `9284`

## plasma_biomarkers[1]
- group: `plasma_biomarkers`
- file: `SAA_Biospecimen_Analysis_Results_06Sep2025.csv`
- exists: `True`
- shape (sample/full): `(2167, 48)`
- id `PATNO` present: `True`, unique: `False`, dupes: `806`

## instruments_optional[0]
- group: `instruments_optional`
- file: `Montreal_Cognitive_Assessment__MoCA__07Sep2025.csv`
- exists: `True`
- shape (sample/full): `(10000, 35)`
- id `PATNO` present: `True`, unique: `False`, dupes: `8375`

## instruments_optional[1]
- group: `instruments_optional`
- file: `MDS_UPDRS_Part_II__Patient_Questionnaire_07Sep2025.csv`
- exists: `True`
- shape (sample/full): `(10000, 22)`
- id `PATNO` present: `True`, unique: `False`, dupes: `9255`

## instruments_optional[2]
- group: `instruments_optional`
- file: `MDS-UPDRS_Part_I_Patient_Questionnaire_07Sep2025.csv`
- exists: `True`
- shape (sample/full): `(10000, 16)`
- id `PATNO` present: `True`, unique: `False`, dupes: `9255`

## instruments_optional[3]
- group: `instruments_optional`
- file: `MDS-UPDRS_Part_I_Non-Motor_Aspects__Online__07Sep2025.csv`
- exists: `True`
- shape (sample/full): `(10000, 15)`
- id `PATNO` present: `True`, unique: `False`, dupes: `8041`

## instruments_optional[4]
- group: `instruments_optional`
- file: `MDS-UPDRS_Part_IV__Motor_Complications_07Sep2025.csv`
- exists: `True`
- shape (sample/full): `(10000, 23)`
- id `PATNO` present: `True`, unique: `False`, dupes: `8590`

## visit_types
- group: `visit_types`
- file: `Visit_Type_07Sep2025.csv`
- exists: `True`
- shape (sample/full): `(9779, 9)`
- id `PATNO` present: `True`, unique: `False`, dupes: `6542`

## code_list
- group: `code_list`
- file: `Code_List_-__Annotated__07Sep2025.csv`
- exists: `True`
- shape (sample/full): `(7483, 5)`
- id `PATNO` present: `False`, unique: `None`, dupes: `None`

## data_dictionary
- group: `data_dictionary`
- file: `Data_Dictionary_-__Annotated__07Sep2025.csv`
- exists: `True`
- shape (sample/full): `(6022, 13)`
- id `PATNO` present: `False`, unique: `None`, dupes: `None`
