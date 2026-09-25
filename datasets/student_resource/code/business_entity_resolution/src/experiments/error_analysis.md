# Detailed Error Analysis Report

## Summary Metrics
- **Best Model**: XGBoost + Hard Negative Mining
- **Optimal Decision Threshold**: 0.730
- **Validation Macro F0.5**: 0.907790
- **Validation Precision**: 0.986410
- **Validation Recall**: 0.830874
- **Singleton Accuracy**: 0.956186

---

## Top False Positives (False Merges)
False merges represent different real-world businesses assigned to the same S1 entity.

### False Positive Example 1
- **S1 Entity (S1-950028092)**: Name: `Khazana (India) India Ltd` | Address: `227/2 Songaon, S Nimb, Satara, Maharashtra`
- **Predicted Candidate (S3-351714802)**: Name: `LLP Khazana (India) Inhia` | Address: `Door No 227/ Songaon, Satara, MH`
- **Country**: `india`
### False Positive Example 2
- **S1 Entity (S1-386663637)**: Name: `Williams Starry` | Address: `11710 750, Middlebury, IN`
- **Predicted Candidate (S2-778060502)**: Name: `WILLIAMS STARRY,` | Address: ``
- **Country**: `us`
### False Positive Example 3
- **S1 Entity (S1-972151648)**: Name: `Internal Medicine Care Associates LLC` | Address: `6424 Burro Creek Place, Tucson, AZ`
- **Predicted Candidate (S2-118450955)**: Name: `Internal Medicine Care Associates Enterprises` | Address: ``
- **Country**: `us`
### False Positive Example 4
- **S1 Entity (S1-972151648)**: Name: `Internal Medicine Care Associates LLC` | Address: `6424 Burro Creek Place, Tucson, AZ`
- **Predicted Candidate (S2-629877451)**: Name: `Internal Medicine Care Associates Corporation` | Address: ``
- **Country**: `us`
### False Positive Example 5
- **S1 Entity (S1-616924638)**: Name: `Seven Consultants` | Address: `Plot No Ci/2, Midc Hingna Midc, Nagpur, Maharashtra`
- **Predicted Candidate (S3-488082325)**: Name: `Care Seven Services` | Address: `2, Nagpur, महाराष्ट्र`
- **Country**: `india`

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
