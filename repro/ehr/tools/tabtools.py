"""Path-fixed copy of EHR/ehragent/tools/tabtools.py (absolute MIMIC-III paths).
Logic is identical to the repo; only ehr_dict paths are made absolute so the
generated code executes against the real (truncated) MIMIC-III CSVs."""
import pandas as pd
import json
import re
import sqlite3
import Levenshtein
import os

BASE = "/home/user/MINJA/EHR/ehragent/ehrsql-ehragent/mimic_iii"

def db_loader(target_ehr):
    ehr_dict = {
        "admissions": f"{BASE}/ADMISSIONS.csv",
        "chartevents": f"{BASE}/CHARTEVENTS.csv",
        "cost": f"{BASE}/COST.csv",
        "d_icd_diagnoses": f"{BASE}/D_ICD_DIAGNOSES.csv",
        "d_icd_procedures": f"{BASE}/D_ICD_PROCEDURES.csv",
        "d_items": f"{BASE}/D_ITEMS.csv",
        "d_labitems": f"{BASE}/D_LABITEMS.csv",
        "diagnoses_icd": f"{BASE}/DIAGNOSES_ICD.csv",
        "icustays": f"{BASE}/ICUSTAYS.csv",
        "inputevents_cv": f"{BASE}/INPUTEVENTS_CV.csv",
        "labevents": f"{BASE}/LABEVENTS.csv",
        "microbiologyevents": f"{BASE}/MICROBIOLOGYEVENTS.csv",
        "outputevents": f"{BASE}/OUTPUTEVENTS.csv",
        "patients": f"{BASE}/PATIENTS.csv",
        "prescriptions": f"{BASE}/PRESCRIPTIONS.csv",
        "procedures_icd": f"{BASE}/PROCEDURES_ICD.csv",
        "transfers": f"{BASE}/TRANSFERS.csv",
    }
    data = pd.read_csv(ehr_dict[target_ehr])
    return data

def data_filter(data, argument):
    backup_data = data
    commands = argument.split('||')
    for i in range(len(commands)):
        try:
            if '>=' in commands[i]:
                command = commands[i].split('>='); column_name = command[0]; value = command[1]
                try: value = type(data[column_name][0])(value)
                except: pass
                data = data[data[column_name] >= value]
            elif '<=' in commands[i]:
                command = commands[i].split('<='); column_name = command[0]; value = command[1]
                try: value = type(data[column_name][0])(value)
                except: pass
                data = data[data[column_name] <= value]
            elif '>' in commands[i]:
                command = commands[i].split('>'); column_name = command[0]; value = command[1]
                try: value = type(data[column_name][0])(value)
                except: pass
                data = data[data[column_name] > value]
            elif '<' in commands[i]:
                command = commands[i].split('<'); column_name = command[0]; value = command[1]
                if value[0] == "'" or value[0] == '"': value = value[1:-1]
                try: value = type(data[column_name][0])(value)
                except: pass
                data = data[data[column_name] < value]
            elif '=' in commands[i]:
                command = commands[i].split('='); column_name = command[0]; value = command[1]
                if value[0] == "'" or value[0] == '"': value = value[1:-1]
                try:
                    examplar = backup_data[column_name].tolist()[0]
                    value = type(examplar)(value)
                except: pass
                data = data[data[column_name] == value]
            elif ' in ' in commands[i]:
                command = commands[i].split(' in '); column_name = command[0]; value = command[1]
                value_list = [s.strip() for s in value.strip("[]").split(',')]
                value_list = [s.strip("'").strip('"') for s in value_list]
                value_list = list(map(type(data[column_name][0]), value_list))
                data = data[data[column_name].isin(value_list)]
            elif 'max' in commands[i]:
                command = commands[i].split('max('); column_name = command[1].split(')')[0]
                data = data[data[column_name] == data[column_name].max()]
            elif 'min' in commands[i]:
                command = commands[i].split('min('); column_name = command[1].split(')')[0]
                data = data[data[column_name] == data[column_name].min()]
        except:
            if column_name not in data.columns.tolist():
                columns = ', '.join(data.columns.tolist())
                raise Exception("The filtering query {} is incorrect. Please modify the column name or use LoadDB to read another table. The column names in the current DB are {}.".format(commands[i], columns))
            if column_name == '' or value == '':
                raise Exception("The filtering query {} is incorrect. There is syntax error in the command.".format(commands[i]))
        if len(data) == 0:
            column_values = list(set(backup_data[column_name].tolist()))
            if ('=' in commands[i]) and (not value in column_values) and (not '>=' in commands[i]) and (not '<=' in commands[i]):
                ld = {cv: Levenshtein.distance(str(cv), str(value)) for cv in column_values}
                ld = sorted(ld.items(), key=lambda x: x[1])
                column_values = ', '.join([str(i[0]) for i in ld[:5]])
                raise Exception("The filtering query {} is incorrect. There is no {} value in the column. Five example values in the column are {}.".format(commands[i], value, column_values))
            else:
                return data
    return data

def get_value(data, argument):
    try:
        commands = argument.split(', ')
        if len(commands) == 1:
            column = argument
            while column[0] == '[' or column[0] == "'": column = column[1:]
            while column[-1] == ']' or column[-1] == "'": column = column[:-1]
            if len(data) == 1:
                return str(data.iloc[0][column])
            else:
                answer_list = [str(i) for i in list(set(data[column].tolist()))]
                return ', '.join(answer_list)
        else:
            column = commands[0]
            if 'mean' in commands[-1]:
                res = [float(i) for i in data[column].tolist()]; return sum(res)/len(res)
            elif 'max' in commands[-1]:
                res = data[column].tolist()
                try: res = [float(i) for i in res]
                except: res = [str(i) for i in res]
                return max(res)
            elif 'min' in commands[-1]:
                res = data[column].tolist()
                try: res = [float(i) for i in res]
                except: res = [str(i) for i in res]
                return min(res)
            elif 'sum' in commands[-1]:
                return sum(float(i) for i in data[column].tolist())
            elif 'list' in commands[-1]:
                return [str(i) for i in data[column].tolist()]
            else:
                raise Exception("The operation {} contains syntax errors.".format(commands[-1]))
    except:
        column_values = ', '.join(data.columns.tolist())
        raise Exception("The column name {} is incorrect. The columns in this table include {}.".format(commands[0] if commands else '', column_values))

def sql_interpreter(command):
    con = sqlite3.connect(f"{BASE}/mimic_iii.db")
    cur = con.cursor()
    return cur.execute(command).fetchall()

def date_calculator(argument):
    try:
        con = sqlite3.connect(f"{BASE}/mimic_iii.db")
        cur = con.cursor()
        return cur.execute("select datetime(current_time, '{}')".format(argument)).fetchall()[0][0]
    except:
        raise Exception("The date calculator {} is incorrect.".format(argument))
