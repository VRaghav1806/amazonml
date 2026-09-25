import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

train_s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", nrows=5)
print("--- TRAIN S1 sample ---")
print(train_s1)

train_s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", nrows=5)
print("\n--- TRAIN S2 sample ---")
print(train_s2)

train_gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=5)
print("\n--- TRAIN GT sample ---")
print(train_gt)

