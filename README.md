# MACHINE-LEARNING-PROJECTS

A collection of machine learning and deep learning notebooks covering classification, regression, and computer vision tasks done as part of IIT Madras BS Degree Course

## Projects

| Notebook | Task | Type |
|---|---|---|
| [`25f1000979-notebook-26t2.ipynb`] | Mechanical parts tray — pixel-level segmentation into 6 fastener classes | Semantic Segmentation |
| [`Flight_Price_Prediction.ipynb`] | Predicting flight ticket prices from airline, route, and booking features | Regression |
| [`Mushroom_EdiblePoisonous_Prediction.ipynb`] | Classifying mushrooms as edible or poisonous from physical attributes | Classification |
| [`Taxi_Fare_Prediction.ipynb`] | Predicting taxi fare amounts from trip details | Regression |

---

## Mechanical Parts Segmentation

Pixel-level semantic segmentation of a cluttered parts tray into six fastener classes: `hex_nut`, `washer`, `bolt`, `ball_bearing`, `spring`, `o_ring`.

**Constraint:** trained fully from scratch — no pretrained weights of any kind.

**Approach:**
- Custom U-Net (GroupNorm, Kaiming init, deep-supervision heads)
- Combined weighted cross-entropy + macro (per-class) soft Dice loss
- Augmentation: flips/rotations, highlight-preserving color jitter, copy-paste augmentation for occlusion diversity
- Test-time augmentation (flip averaging) at inference

**Metric:** mean Dice coefficient, averaged over every (image, class) pair.

## Flight Price Prediction

Predicting flight ticket prices from features like airline, source/destination, number of stops, duration, and days left before departure.

**Workflow:**
1. Load data, check dtypes and basic stats
2. Handle missing values (median for numeric, mode for categorical — computed on train only, to avoid leakage) and check duplicates
3. EDA: price is strongly right-skewed with a clear Business-class cluster above Economy; Vistara/Air_India (the carriers offering Business class) show higher median prices; prices rise sharply as `days_left` shrinks
4. Outliers kept deliberately — high prices are legitimate Business-class long-haul fares, not data errors, and dropping them would stop the model learning to predict expensive tickets at all
5. One-hot encoding for low-cardinality categoricals (`airline`, `source`, `departure`, `stops`, `arrival`, `destination`, `class`); `flight` dropped (near-unique ID, not useful); `StandardScaler` on numeric columns
6. Trained and compared: Linear, Ridge, Lasso, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost
7. Tuned Random Forest, Gradient Boosting, and XGBoost via `RandomizedSearchCV`

**Result:** tuned tree ensembles clearly outperformed the linear models (price doesn't scale linearly with features — e.g. the Economy→Business jump is a step change, not linear). Best model: **tuned XGBoost, R² ≈ 0.968** on the validation set.

**Metric:** R² score

## Mushroom Edible/Poisonous Prediction


Binary classification predicting whether a mushroom is edible (`e`) or poisonous (`p`) from 22 categorical physical traits (cap shape, odor, gill color, spore print color, habitat, etc.) plus 2 numerical features (`number_of_bruises`, `ring-number`).

**Workflow:**
1. Identify column types: 2 numerical, 22 categorical, plus `ID`/`mushroom_id` as row identifiers
2. Handle missing values — mode imputation for categoricals (`odor` ~46% missing, `stalk-root` ~2.7%, `ring-type` ~0.5%), median imputation for the numeric `ring-number` (~0.5% missing); imputing rather than dropping to avoid losing ~46% of the data over `odor` alone
3. Check duplicates
4. Outliers in `number_of_bruises` detected via IQR but **retained** — tree ensembles are robust to them, and extreme bruise counts may carry genuine biological signal
5. EDA: `odor` and `spore-print-color` stand out as the most discriminative features between edible and poisonous mushrooms; habitat and bruising also show visible class separation
6. Label Encoding for all 22 categorical features (one-hot would blow up dimensionality with 22 high-cardinality columns); `StandardScaler` on the 2 numeric features
7. Trained and compared 7 classifiers: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, SVM, KNN
8. Tuned Random Forest, Gradient Boosting, and Decision Tree via `GridSearchCV` (3-fold CV)

**Result:** tree-based ensembles (Random Forest, Gradient Boosting) consistently outperformed linear/distance-based models. Best model: **tuned Random Forest**, achieving near-perfect validation accuracy — confirming mushroom toxicity is largely determined by observable physical traits.

**Metric:** Accuracy

## Taxi Fare Prediction

Predicting total taxi fare (`total_amount`) from NYC-style trip data: pickup/dropoff timestamps, trip distance, passenger count, rate code, pickup/dropoff zone IDs, payment type, and surcharge fields.

**Workflow:**
1. Load data, check dtypes and null counts across all 17 columns
2. Handle missing values (`passenger_count`, `RatecodeID`, `store_and_fwd_flag`, `congestion_surcharge`, `Airport_fee`, ~3.7% missing each) — median for numeric, mode for categorical
3. Check duplicates (none found)
4. Outlier handling — removed ~39% of rows with physically invalid values (negative fares/durations), then IQR-capped remaining outliers in `total_amount` and `trip_distance`
5. Feature engineering — `trip_duration_mins`, `pickup_hour`, `pickup_dayofweek`, `pickup_month`, `is_weekend`, `is_rush_hour`, `is_night`, `speed_mph`, `dist_per_min`
6. EDA: `total_amount` is right-skewed (most rides $10–$35); `trip_distance` and `trip_duration_mins` are the strongest correlates of fare; rush hours (7–10am, 4–7pm) and weekends show higher average fares; credit-card payments associate with longer, higher-fare trips
7. Label-encoded `payment_type`, `StandardScaler` on 23 numeric features
8. Trained and compared 7 models: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, Extra Trees
9. Tuned Decision Tree, Random Forest, and Gradient Boosting via `GridSearchCV`

**Result:** tree-based ensembles clearly beat the linear models (RMSE ~5.22 for linear/Ridge/Lasso vs. ~3.85–4.06 for tuned trees). Best model: **tuned Gradient Boosting — RMSE 3.85, MAE 2.41, R² 0.949**.

**Metric:** RMSE (R² and MAE also reported)

---

## Tech Stack

- Python, NumPy, Pandas, Seaborn/Matplotlib
- scikit-learn, XGBoost
- PyTorch (segmentation project)
- Jupyter Notebooks (developed on Kaggle)
