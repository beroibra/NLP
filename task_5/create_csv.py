import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_json("../preprocessed_data.json")
sms = df["sms"].to_list()
labels = df["label"].to_list()

X_train, X_test, y_train, y_test = train_test_split(sms, labels, test_size=0.2, random_state=42,
                                                    shuffle=True, stratify=labels)

train_df_sms = pd.Series(data=X_train, name='sms', dtype='str')
train_df_labels = pd.Series(data=y_train, name='label', dtype='str')
train_df = pd.concat([train_df_labels, train_df_sms], axis=1)
train_df['label'] = '__label__' + train_df['label'].astype(str)
train_df.to_csv("./data/train.csv", sep='\t', index=False, header=False)


test_df_sms = pd.Series(data=X_test, name='sms', dtype='str')
test_df_labels = pd.Series(data=y_test, name='label', dtype='str')
test_df = pd.concat([test_df_labels, test_df_sms], axis=1)
test_df['label'] = '__label__' + test_df['label'].astype(str)
test_df.to_csv("./data/test.csv", sep='\t', index=False, header=False)


dev = open("./data/dev.csv", "a").close()

print()