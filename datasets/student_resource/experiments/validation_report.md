# Data Profiling & Audit Report

## 1. Record Counts

### Training Set
- **Train Source 1 (Reference Entities)**: 2,206,821
- **Train Source 2 Records**: 5,034,616
- **Train Source 3 Records**: 5,285,603
- **Train Ground Truth Records**: 2,206,821

### Test Set
- **Test Source 1 Records**: 1,732,544
- **Test Source 2 Records**: 4,887,273
- **Test Source 3 Records**: 5,082,316

---

## 2. Missing Values

### Training Data
- **Train S1**: `business_name` 0.00% missing, `business_address` 0.00% missing, `country` 0.00% missing.
- **Train S2**: `business_name` 0.00% (2 missing), `business_address` 3.36% (168,967 missing), `country` 0.00% missing.
- **Train S3**: `business_name` 0.00% (13 missing), `business_address` 3.33% (175,916 missing), `country` 0.00% missing.

### Test Data
- **Test S1**: `business_name` 0.00% missing, `business_address` 0.00% missing, `country` 0.00% missing.
- **Test S2**: `business_name` 0.00% (46 missing), `business_address` 2.65% (129,408 missing), `country` 0.00% missing.
- **Test S3**: `business_name` 0.00% (59 missing), `business_address` 2.68% (136,098 missing), `country` 0.00% missing.

---

## 3. Ground Truth Match Distribution

- **0 matches (Singletons)**: 123,247 (5.58%)
- **1 match**: 119,157 (5.40%)
- **2 matches**: 375,212 (17.00%)
- **3 matches**: 530,841 (24.05%)
- **4 matches**: 484,115 (21.94%)
- **5 matches**: 321,957 (14.59%)
- **6 matches**: 164,868 (7.47%)
- **7 matches**: 63,968 (2.90%)
- **8 matches**: 18,680 (0.85%)
- **9 matches**: 4,205 (0.19%)
- **10+ matches**: 571 (0.02%)

**Total Ground Truth Matches**: 7,638,365 pairs (S2: 3,693,619, S3: 3,944,746).

---

## 4. Country Distributions

### Training Set
- **US**: 1,323,633 (S1), 3,016,817 (S2), 3,170,056 (S3)
- **India**: 883,188 (S1), 2,017,799 (S2), 2,115,547 (S3)

### Test Set
- **India**: 809,986 (S1), 2,312,565 (S2), 2,405,000 (S3)
- **US**: 663,106 (S1), 1,871,330 (S2), 1,945,701 (S3)
- **France**: 259,452 (S1), 703,378 (S2), 731,615 (S3) [Unseen in Training]

---

## Key Modeling Insights
1. **Unseen Country (France)**: Country must be treated as open-set feature without hardcoded static rules for US/India.
2. **Multi-match & Singleton ratio**: ~89% of S1 entities match multiple records across S2 and S3, ~5.6% are singletons.
3. **Missing Addresses**: ~3% of S2/S3 records have missing addresses. Address similarity features must handle null/empty values cleanly without throwing errors or introducing NaN features.
