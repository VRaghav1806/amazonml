# Detailed Error Analysis Report

## Summary Metrics
- **Best Model**: XGBoost + Hard Negative Mining
- **Optimal Decision Threshold**: 0.750
- **Validation Macro F0.5**: 0.906912
- **Validation Precision**: 0.985578
- **Validation Recall**: 0.829836
- **Singleton Accuracy**: 0.950172

---

## Top False Positives (False Merges)
False merges represent different real-world businesses assigned to the same S1 entity.

### False Positive Example 1
- **S1 Entity (S1-105568631)**: Name: `Economic Development Coalition IV` | Address: `189 Laurel Road, Arden, NC`
- **Predicted Candidate (S2-329810939)**: Name: `ECONOMIC DEVELOPMENT COALITION` | Address: `47 HILLCREST ROAD, AHEVILLE, NC`
- **Country**: `us`
### False Positive Example 2
- **S1 Entity (S1-601212223)**: Name: `Chennai Technologies Private Limited` | Address: `No.1/1525B, 4Th Street B.M.Nagar, Chennai, Tamil Nadu`
- **Predicted Candidate (S3-327244878)**: Name: `Chennai Services` | Address: `தமிழ்நாடு, Fg-312, Chennai`
- **Country**: `india`
### False Positive Example 3
- **S1 Entity (S1-601212223)**: Name: `Chennai Technologies Private Limited` | Address: `No.1/1525B, 4Th Street B.M.Nagar, Chennai, Tamil Nadu`
- **Predicted Candidate (S3-319005428)**: Name: `Chennai-International  Private Limited` | Address: `New No:25, TN, Chennai`
- **Country**: `india`
### False Positive Example 4
- **S1 Entity (S1-386663637)**: Name: `Williams Starry` | Address: `11710 750, Middlebury, IN`
- **Predicted Candidate (S2-778060502)**: Name: `WILLIAMS STARRY,` | Address: ``
- **Country**: `us`
### False Positive Example 5
- **S1 Entity (S1-86778868)**: Name: `Internal Medicine Clinic` | Address: `3197 Marie Avenue, Millcreek, UT`
- **Predicted Candidate (S2-247247924)**: Name: `Internal Medicine [Clinic]` | Address: ``
- **Country**: `us`

---

## Top False Negatives (Missed Matches)
### False Negative Example 1
- **S1 Entity (S1-105568631)**: Name: `Economic Development Coalition IV` | Address: `189 Laurel Road, Arden, NC`
- **Missed Ground Truth Match (S3-686929711)**: Name: `Economic Development Coalition  lV Co` | Address: `189 Laurel Road, Arden, North Carolina`
- **Country**: `us`
### False Negative Example 2
- **S1 Entity (S1-708400226)**: Name: `Trinity Lutheran Church` | Address: `1023 Dakota Street, Spring Valley, IL`
- **Missed Ground Truth Match (S3-614607391)**: Name: `Deltavio` | Address: `1023 Dakota Saint, Spring Valley, Illinois`
- **Country**: `us`
### False Negative Example 3
- **S1 Entity (S1-708400226)**: Name: `Trinity Lutheran Church` | Address: `1023 Dakota Street, Spring Valley, IL`
- **Missed Ground Truth Match (S2-737021373)**: Name: `TRINITY LUTSHRN CHURCH` | Address: `DAKOTA ST, SPRING VALLEY, IL`
- **Country**: `us`
### False Negative Example 4
- **S1 Entity (S1-151299429)**: Name: `Hari Consultancy Limited` | Address: `Flate No 406 Shakar Splun, Vamali Sama Savli Road, Vadodara, Gujarat`
- **Missed Ground Truth Match (S3-939400153)**: Name: `Hari Consultancy Ltd` | Address: ``
- **Country**: `india`
### False Negative Example 5
- **S1 Entity (S1-597619383)**: Name: `Esquire Marketing Private Limited` | Address: `D-34, Gali No.9 Brahmpuri, Shahdara, Delhi, East Delhi, Delhi`
- **Missed Ground Truth Match (S3-567956556)**: Name: `Esquire Marketing Prmvmase Limited` | Address: ``
- **Country**: `india`
