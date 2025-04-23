import os
import numpy as np
import pandas as pd
import json

for dataset in ['santa', 'rnd']:
    for epochs in ['16', '128']:

        # read files (test + santa)
        def filter_file(f):
            split = f.split("_")
            if len(split) < 5:
                return False
            else:
                return (split[0]=='test') & (split[1].split("-")[2]==dataset) & (f.split("_")[3]==epochs)
        files = os.listdir("logs/")
        files = [f for f in files if filter_file(f)]

        # create joined df
        DF = []
        for file in files:
            with open(f"logs/{file}") as f:
                df = pd.DataFrame(json.load(f))
                split = file.split("_")[1].split("-")
                df['group_target_id'] = split[0][1:] + '-' + split[1][1:]
                DF.append(df)
        DF = pd.concat(DF)
        DF = DF[['test_num', 'solution_length', 'group_target_id']]

        # accumulate min dist
        df = DF.groupby(["group_target_id", "test_num"]).min()['solution_length']

        # print avg results
        mean_nan = df.isna().groupby('group_target_id').mean()
        mean_length = df.groupby('group_target_id').mean()
        unique_tests = DF.groupby('group_target_id')['test_num'].nunique()
        mean_res = pd.DataFrame({
            '# tests': unique_tests,
            'solved, %': (100*(1-mean_nan)).round().astype(int),
            'avg sol length': mean_length.round(1)
        })

        # add santa
        df_santa = pd.read_csv("notebooks/avg-santa-scores.csv", index_col='group_target_id')
        df_santa.rename(columns={'Santa Best': 'avg santa sol length'}, inplace=True)
        mean_res_joined = mean_res.join(df_santa['avg santa sol length'].round(1), how='left')
        mean_res_joined = mean_res_joined.join(df_santa['puzzle_type'], how='left')

        print(f"\n\n------- epochs={epochs:3}, dataset={dataset:5} -------\n") 
        print(mean_res_joined)