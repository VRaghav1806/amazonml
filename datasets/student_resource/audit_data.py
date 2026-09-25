import os
import pandas as pd
import numpy as np

DATA_DIR = "dataset"

def audit_dataset():
    print("================ DATASET AUDIT ================", flush=True)
    train_s1 = pd.read_csv(os.path.join(DATA_DIR, "train", "train_source1.tsv"), sep="\t")
    print("Loaded train_s1", flush=True)
    train_s2 = pd.read_csv(os.path.join(DATA_DIR, "train", "train_source2.tsv"), sep="\t")
    print("Loaded train_s2", flush=True)
    train_s3 = pd.read_csv(os.path.join(DATA_DIR, "train", "train_source3.tsv"), sep="\t")
    print("Loaded train_s3", flush=True)
    train_gt = pd.read_csv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), sep="\t")
    print("Loaded train_gt", flush=True)
    
    test_s1 = pd.read_csv(os.path.join(DATA_DIR, "test", "test_source1.tsv"), sep="\t")
    print("Loaded test_s1", flush=True)
    test_s2 = pd.read_csv(os.path.join(DATA_DIR, "test", "test_source2.tsv"), sep="\t")
    print("Loaded test_s2", flush=True)
    test_s3 = pd.read_csv(os.path.join(DATA_DIR, "test", "test_source3.tsv"), sep="\t")
    print("Loaded test_s3", flush=True)
    
    print("\n--- TRAIN RECORD COUNTS ---")
    print(f"Train Source 1 records: {len(train_s1)}")
    print(f"Train Source 2 records: {len(train_s2)}")
    print(f"Train Source 3 records: {len(train_s3)}")
    print(f"Train Ground Truth rows: {len(train_gt)}")
    
    print("\n--- TEST RECORD COUNTS ---")
    print(f"Test Source 1 records: {len(test_s1)}")
    print(f"Test Source 2 records: {len(test_s2)}")
    print(f"Test Source 3 records: {len(test_s3)}")
    
    print("\n--- MISSING VALUES (TRAIN) ---")
    for name, df in [("Train S1", train_s1), ("Train S2", train_s2), ("Train S3", train_s3)]:
        print(f"\n{name}:")
        for col in ["business_name", "business_address", "country"]:
            missing = df[col].isna().sum() if col in df.columns else 0
            pct = (missing / len(df)) * 100
            print(f"  {col}: {missing} missing ({pct:.2f}%)")

    print("\n--- MISSING VALUES (TEST) ---")
    for name, df in [("Test S1", test_s1), ("Test S2", test_s2), ("Test S3", test_s3)]:
        print(f"\n{name}:")
        for col in ["business_name", "business_address", "country"]:
            missing = df[col].isna().sum() if col in df.columns else 0
            pct = (missing / len(df)) * 100
            print(f"  {col}: {missing} missing ({pct:.2f}%)")
            
    print("\n--- GROUND TRUTH MATCH DISTRIBUTION ---")
    train_gt["matched_list"] = train_gt["matched_entity_ids"].fillna("").apply(
        lambda x: [i.strip() for i in str(x).split(",") if i.strip()]
    )
    train_gt["num_matches"] = train_gt["matched_list"].apply(len)
    
    match_counts = train_gt["num_matches"].value_counts().sort_index()
    print("Match count breakdown:")
    for k, v in match_counts.items():
        pct = (v / len(train_gt)) * 100
        print(f"  {k} matches: {v} entities ({pct:.2f}%)")
        
    singletons = (train_gt["num_matches"] == 0).sum()
    multi_matches = (train_gt["num_matches"] > 1).sum()
    single_match = (train_gt["num_matches"] == 1).sum()
    print(f"\nTotal singletons (0 matches): {singletons} ({(singletons/len(train_gt))*100:.2f}%)")
    print(f"Total single match (1 match): {single_match} ({(single_match/len(train_gt))*100:.2f}%)")
    print(f"Total multi match (>1 match): {multi_matches} ({(multi_matches/len(train_gt))*100:.2f}%)")
    
    # S1 -> S2, S1 -> S3 counts
    def get_source(id_str):
        if id_str.startswith("S2-"):
            return "S2"
        elif id_str.startswith("S3-"):
            return "S3"
        return "UNKNOWN"
        
    train_gt["s2_matches"] = train_gt["matched_list"].apply(lambda lst: sum(1 for x in lst if get_source(x)=="S2"))
    train_gt["s3_matches"] = train_gt["matched_list"].apply(lambda lst: sum(1 for x in lst if get_source(x)=="S3"))
    
    print("\nSource breakdown in GT:")
    print(f"  Total S2 matches in GT: {train_gt['s2_matches'].sum()}")
    print(f"  Total S3 matches in GT: {train_gt['s3_matches'].sum()}")
    
    print("\n--- COUNTRY DISTRIBUTION (TRAIN) ---")
    for name, df in [("Train S1", train_s1), ("Train S2", train_s2), ("Train S3", train_s3)]:
        print(f"{name}: {dict(df['country'].value_counts(dropna=False))}")
        
    print("\n--- COUNTRY DISTRIBUTION (TEST) ---")
    for name, df in [("Test S1", test_s1), ("Test S2", test_s2), ("Test S3", test_s3)]:
        print(f"{name}: {dict(df['country'].value_counts(dropna=False))}")

if __name__ == "__main__":
    audit_dataset()
