from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os

data_name = "5fold_stratified_folds"

def main():
    df = pd.read_json(open("../preprocessed_data.json"))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y = df.label

    for counter, idx in enumerate(skf.split(np.zeros(len(df)), y)):
        X_train, X_test = df.iloc[idx[0]], df.iloc[idx[1]]

        os.makedirs(data_name + "/"+str(counter+1)+"/", exist_ok=True)
        X_train.to_csv(data_name + "/"+str(counter+1)+"/train.csv", sep='\t', index=False, header=False)
        X_test.to_csv(data_name + "/"+str(counter+1)+"/test.csv", sep='\t', index=False, header=False)
        dev = open(data_name + "/"+str(counter+1)+"/dev.csv", "a").close()


if __name__ == '__main__':
    main()
