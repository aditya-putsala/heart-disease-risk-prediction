# Heart Disease Risk Prediction

Predicting whether a patient has heart disease from 13 clinical measurements, using Logistic Regression.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Results at a Glance](#results-at-a-glance)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modelling](#modelling)
- [Model Results](#model-results)
- [Known Issues and Limitations](#known-issues-and-limitations)
- [Key Takeaways](#key-takeaways)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)

---

## Problem Statement

Heart disease is often found late. Given a patient's age, sex, chest pain type, blood pressure, cholesterol, exercise test results and other clinical measurements, the task is to build a machine learning model that predicts whether the patient has heart disease.

This is a **supervised, binary classification** problem. The target column is `target` (`1` = heart disease, `0` = healthy).

---

## Results at a Glance

| Item | Value |
|------|-------|
| Patients / columns | 303 / 14 |
| Heart disease (`1`) | 139 (45.9%) |
| Healthy (`0`) | 164 (54.1%) |
| Missing values | None |
| Duplicate rows | None |
| Models trained in the notebook | 1: Logistic Regression (a Support Vector Machine section was started but not finished) |
| Train accuracy | 0.847 |
| Test accuracy | **0.836** (61 test patients) |
| Test recall for heart disease † | 0.786 (22 of 28 patients with disease found) |
| Test AUROC † | 0.895 |
| Accuracy of always predicting "healthy" † | 0.541 |

Figures marked † were calculated for this README by re-running the notebook's pipeline; they are not printed in the notebook. The test set is small (61 patients), so the scores are rough. See [Known Issues and Limitations](#known-issues-and-limitations).

---

## Dataset

`Heart_Disease_Dataset.csv` has **303 rows and 14 columns**, one row per patient. The notebook renames the columns after loading; both names are shown below.

| Original column | Notebook name | Description | Values in the data |
|-----------------|---------------|-------------|--------------------|
| `age` | `age` | Age in years | 29 to 77 (mean 54.4) |
| `sex` | `sex` | Sex | 1 = male: 206; 0 = female: 97 † |
| `cp` | `chest_pain_type` | Chest pain type | 1 typical angina: 23; 2 atypical angina: 50; 3 non-anginal pain: 86; 4 asymptomatic: 144 † |
| `trestbps` | `resting_blood_pressure` | Resting blood pressure (mm Hg) | 94 to 200 (mean 131.7) |
| `chol` | `cholestrol` | Cholesterol (mg/dl) | 126 to 564 (mean 246.7) |
| `fbs` | `fasting_blood_sugar` | Fasting blood sugar above 120 mg/dl | 1 = true: 45; 0 = false: 258 † |
| `restecg` | `rest_ecg` | Resting ECG result | 0 normal: 151; 1 ST-T wave abnormality: 4; 2 probable or definite left ventricular hypertrophy: 148 † |
| `thalach` | `max_heart_rate_achieved` | Maximum heart rate achieved (bpm) | 71 to 202 (mean 149.6) |
| `exang` | `exercise_induced_angina` | Exercise-induced angina | 1 = yes: 99; 0 = no: 204 † |
| `oldpeak` | `st_depression` | ST depression induced by exercise relative to rest | 0.0 to 6.2 (mean 1.04) |
| `slope` | `st_slope` | Slope of the peak exercise ST segment | 1 upsloping: 142; 2 flat: 140; 3 downsloping: 21 † |
| `ca` | `num_major_vessels` | Major vessels (0 to 3) coloured by fluoroscopy | 0: 177; 1: 67; 2: 38; 3: 21 † |
| `thal` | `thalassemia` | Thallium stress test result | 3 normal: 167; 6 fixed defect: 19; 7 reversible defect: 117 † |
| `target` | `target` | **Target:** heart disease | 0: 164; 1: 139 |

The data source is not stated in the notebook.

---

## Data Preparation

- **Missing values:** none in any column. All columns are numeric.
- **Duplicates:** none.
- **Column names:** the 14 columns were renamed by position to longer names (for example `cp` became `chest_pain_type`). The original spelling `cholestrol` is kept in the notebook.
- **Encoding:** none. The categorical columns (`chest_pain_type`, `rest_ecg`, `st_slope`, `thalassemia`) keep their original numeric codes and are fed to the model as numbers.
- **Scaling:** all 13 features were standardised with `StandardScaler` **before** the train/test split.
- **Split:** 80% train, 20% test, stratified on `target`, `random_state=2`: 242 training and 61 test patients. The test set holds 33 healthy patients and 28 with heart disease †.

---

## Exploratory Data Analysis

Figures marked † were calculated from the CSV for this README. The rest are printed in the notebook.

**Disease rate by group** † (share of patients in each group with `target = 1`)

| Feature | Disease rate by group |
|---------|-----------------------|
| `sex` | female 25.8% (25 of 97); male 55.3% (114 of 206) |
| `chest_pain_type` | typical angina 30.4%; atypical angina 18.0%; non-anginal pain 20.9%; asymptomatic 72.9% |
| `exercise_induced_angina` | no 30.9%; yes 76.8% |
| `st_slope` | upsloping 25.4%; flat 65.0%; downsloping 57.1% (21 patients) |
| `thalassemia` | normal 22.2%; fixed defect 68.4% (19 patients); reversible defect 76.1% |
| `num_major_vessels` | 0 vessels 26.0%; 1 vessel 65.7%; 2 vessels 81.6%; 3 vessels 85.7% |
| `fasting_blood_sugar` | not above 120: 45.3%; above 120: 48.9% |
| `rest_ecg` | normal 37.1%; ST-T abnormality 75.0% (only 4 patients); hypertrophy 54.1% |
| `age` | 39 and under 26.7% (15 patients); 40 to 49 30.6%; 50 to 59 48.0%; 60 and over 58.2% |
| `max_heart_rate_achieved` | up to 120 bpm 81.1%; 121 to 140 64.8%; 141 to 160 47.1%; over 160 23.6% |
| `st_depression` | 0: 26.3%; 0.1 to 1.0: 38.3%; 1.1 to 2.0: 53.4%; over 2.0: 86.0% |

**Average by outcome** (as printed in the notebook)

| Feature | Healthy | Heart disease |
|---------|--------:|--------------:|
| Age | 52.6 | 56.6 |
| Share male | 0.56 | 0.82 |
| Resting blood pressure | 129.3 | 134.6 |
| Cholesterol | 242.6 | 251.5 |
| Max heart rate achieved | 158.4 | 139.3 |
| Exercise-induced angina (share) | 0.14 | 0.55 |
| ST depression | 0.59 | 1.57 |
| Major vessels coloured | 0.28 | 1.15 |

The notebook also prints average codes for `chest_pain_type`, `st_slope` and `thalassemia`. These are not meaningful averages because the numbers are category codes (for example, thalassemia codes 3, 6 and 7), so they are left out here.

**Correlation of each feature with `target`** † (the notebook shows these in a heatmap)

| Feature | Correlation | Feature | Correlation |
|---------|------------:|---------|------------:|
| `thalassemia` | 0.528 | `sex` | 0.277 |
| `num_major_vessels` | 0.461 | `age` | 0.223 |
| `exercise_induced_angina` | 0.432 | `rest_ecg` | 0.169 |
| `st_depression` | 0.425 | `resting_blood_pressure` | 0.151 |
| `max_heart_rate_achieved` | -0.417 | `cholestrol` | 0.085 |
| `chest_pain_type` | 0.414 | `fasting_blood_sugar` | 0.025 |
| `st_slope` | 0.339 | | |

**Key observations**
- **Class balance:** the classes are fairly even (45.9% disease), so accuracy is less misleading here than in a heavily unbalanced problem, though it still hides which kind of mistake is made.
- **Chest pain:** patients with *no* chest pain symptoms (asymptomatic, type 4) have by far the highest disease rate (72.9%). This is the opposite of what one might expect, so the label should not be read as "healthy".
- **Exercise tests:** exercise-induced angina, a low maximum heart rate, larger ST depression, a flat or downsloping ST slope, and more coloured vessels all go with a higher disease rate.
- **Sex:** men have more than twice the disease rate of women (55.3% vs 25.8%). Only 97 women are in the data, so the female figure rests on few patients.
- **Age:** the disease rate rises with age, from about 27% to 31% below age 50 to 58% at 60 and over. The notebook says disease peaks at age 58. That holds for the **number** of patients with disease (12 at age 58, the highest of any age) but not the rate: single-age rates are noisy (at most 19 patients per age) and are higher at ages 60 and 61.
- **Age and other measurements:** blood pressure (correlation 0.285 †) and cholesterol (0.209 †) tend to rise with age, and maximum heart rate falls (-0.394 †), as the notebook's scatter plots show.
- **Fasting blood sugar:** it shows almost no relationship with heart disease in this data (correlation 0.025; 45.3% vs 48.9%).
- **Cholesterol:** a weak link with disease (0.085). Four patients have cholesterol above 400, with a maximum of 564.
- **Small groups:** `rest_ecg = 1` has only 4 patients, `st_slope = 3` has 21 and `thalassemia = 6` has 19, so rates for those groups are unreliable.

---

## Modelling

The model predicts `target` from the 13 scaled features.

| Model | Settings | Test patients |
|-------|----------|--------------:|
| Logistic Regression | Default scikit-learn settings | 61 |

A Support Vector Machine section appears at the end of the notebook but contains only an unfinished import line (`from sklearn.`), which raises a `SyntaxError`. No other model was trained.

---

## Model Results

| Metric | Train | Test |
|--------|------:|-----:|
| Accuracy | 0.8471 | 0.8361 |

**Test confusion matrix †** (61 patients)

| | Predicted healthy | Predicted heart disease |
|---|---:|---:|
| **Actually healthy** | 29 | 4 |
| **Actually heart disease** | 6 | 22 |

| Test metric † | Value |
|---------------|------:|
| Precision (heart disease) | 0.846 |
| Recall (heart disease) | 0.786 |
| F1 | 0.815 |
| AUROC | 0.895 |

**More reliable estimates †.** Because 61 patients is a small test set, I also ran the same pipeline (scaler fitted inside each training fold) with 5-fold cross-validation repeated 10 times:

| Metric † | Mean | Std |
|----------|-----:|----:|
| Accuracy | 0.836 | 0.041 |
| Recall (heart disease) | 0.788 | 0.060 |
| AUROC | 0.903 | 0.033 |

**Standardised coefficients †** (the notebook does not print these; they come from re-fitting its pipeline). Because the features were standardised, the sizes can be compared roughly.

| Raises predicted risk | Coefficient | Lowers predicted risk | Coefficient |
|-----------------------|------------:|-----------------------|------------:|
| `num_major_vessels` | 1.048 | `max_heart_rate_achieved` | -0.360 |
| `thalassemia` | 0.711 | `fasting_blood_sugar` | -0.357 |
| `chest_pain_type` | 0.663 | `age` | -0.123 |
| `sex` | 0.541 | | |
| `rest_ecg` | 0.442 | | |
| `resting_blood_pressure` | 0.432 | | |
| `st_depression` | 0.392 | | |
| `exercise_induced_angina` | 0.359 | | |
| `st_slope` | 0.309 | | |
| `cholestrol` | 0.081 | | |

**How to read these numbers**
- **Single split:** each test patient is worth 1.6 percentage points of accuracy. When I repeated the stratified split with 10 different seeds, test accuracy ranged from 0.754 to 0.918. The notebook's 0.836 is a reasonable middle value, but the true accuracy is uncertain by several points.
- **Overfitting:** the gap between train (0.847) and test (0.836) accuracy is small, so there is no sign of overfitting.
- **Against the baseline:** always predicting "healthy" would score 0.541, so the model is clearly learning something.
- **Missed patients:** 6 of the 28 test patients with heart disease (21%) were predicted healthy. For a screening tool this matters more than the overall accuracy.
- **Coefficients:** the negative coefficients for `age` and `fasting_blood_sugar` look odd given their positive correlations with disease (0.223 and 0.025). They reflect the effect of each feature after the others are accounted for, and they are small and may not be stable with 303 patients. They are not evidence that age or blood sugar protect against heart disease.

---

## Known Issues and Limitations

These affect how far the results can be trusted and are listed so that anyone reusing the code knows about them.

1. **The notebook is unfinished.** It ends with a half-written Support Vector Machine cell that raises a `SyntaxError`, so only Logistic Regression was evaluated and no models were compared. Any statement that Logistic Regression is the "best" model would be unsupported.
2. **The scaler is fitted on all the data before the split.** `scaler.fit_transform(X)` runs on every row, so information from the test patients leaks into the scaling. On this split the effect was nil: fitting the scaler on training data only gave the same test accuracy (0.836), the same confusion matrix and the same AUROC †. It should still be fixed (for example with a scikit-learn `Pipeline`), because it can matter with other models or splits.
3. **Small test set, single split.** Results rest on 61 patients and one split. Repeated cross-validation (above) is a better guide: about 0.84 accuracy, with a standard deviation of about 0.04.
4. **Only accuracy was reported.** The notebook has no confusion matrix, recall or AUROC. Missing a patient with heart disease is usually costlier than a false alarm, so recall should be tracked.
5. **Category codes are treated as numbers.** `chest_pain_type`, `rest_ecg`, `st_slope` and `thalassemia` (coded 3, 6, 7) are nominal categories used as if they were ordered quantities, which does not suit a linear model in principle. In my check, one-hot encoding them gave cross-validated accuracy of 0.839 against 0.836 †, so it made almost no difference here.
6. **The strongest predictors need specialist tests.** `num_major_vessels` (fluoroscopy), `thalassemia` (thallium stress test) and the exercise ECG measurements are only available after a clinical work-up. A model using them is not a cheap screening tool for people who have not yet been examined.
7. **Missing values were not discussed.** This file has none. If it comes from the well-known Cleveland heart disease data, my understanding is that the original has a few missing values in `ca` and `thal`, so it is worth checking how they were filled before this file was created. The notebook does not say where the data came from or how the target was defined.
8. **Notebook errors and labelling mistakes.**
   - The `df1` filter asks for a column named `cholesterol`, but the column is `cholestrol`, so cholesterol is silently left out of `df1` (the output shows 5 columns instead of 6). `df1` is not used later.
   - The scatter plot titled "age vs cholestrol" actually plots maximum heart rate.
   - The heading above the thalassemia cells reads "Maximum heart rate achieved (ordinal)".
   - Slope value 1 is described as "Unsloping"; it means upsloping.
   - Two sex-based count plots set their legends with `plt.legend({'female':0})`, which is unlikely to label the bars correctly.
9. **File name mismatch.** The notebook reads `'Heart Disease Dataset.csv'` (with spaces), but the file in this repository is `Heart_Disease_Dataset.csv` (with underscores). Rename the file or edit the `read_csv` line.
10. **Not a medical tool.** The data covers 303 patients and the model was not validated on any other group. It is a learning exercise and must not be used to diagnose or rule out heart disease.

---

## Key Takeaways

- The task is a binary classification problem. Logistic Regression reached 0.836 test accuracy (0.847 on training), well above the 0.541 baseline. Cross-validation puts accuracy at about 0.84 and AUROC at about 0.90.
- The model missed about one in five patients with heart disease on the test set (6 of 28), so accuracy alone overstates how useful it would be for screening.
- Heart disease goes with exercise-induced angina, low maximum heart rate, larger ST depression, more coloured vessels, thallium defects and male sex.
- Patients with no chest pain symptoms (type 4) have the highest disease rate (72.9%).
- Fasting blood sugar and cholesterol show little link with disease in this data.
- The notebook is incomplete (no second model), uses a leaky scaling step that happened not to change the result, and was evaluated on a very small test set. Fix these before quoting the scores or comparing models.

---

## Getting Started

**1. Install the libraries**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**2. Place the dataset next to the notebook.** The notebook loads the data with `pd.read_csv('Heart Disease Dataset.csv')`, so rename `Heart_Disease_Dataset.csv` to match, or change the file name in that cell.

**3. Open the notebook**
```bash
jupyter notebook Heart_Disease_Risk_Prediction.ipynb
```

**4. Run the cells in order.** The last cell (the unfinished Support Vector Machine import) will fail; everything before it runs.

---

## Repository Structure

```
.
├── Heart_Disease_Risk_Prediction.ipynb   # Cleaning, EDA, scaling and Logistic Regression
├── Heart_Disease_Dataset.csv             # 303 patients x 14 columns
└── README.md
```

---

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn (StandardScaler, train_test_split, Logistic Regression, accuracy_score), Jupyter Notebook.
