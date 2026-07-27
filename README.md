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

<!-- TODO: fill in — dataset source, features used, model(s) tried, accuracy/F1 achieved -->

## Taxi Fare Prediction

<!-- TODO: fill in — dataset source, features used (distance, time, location), model(s) tried, RMSE/MAE achieved -->

---

## Tech Stack

- Python, NumPy, Pandas, Seaborn/Matplotlib
- scikit-learn, XGBoost
- PyTorch (segmentation project)
- Jupyter Notebooks (developed on Kaggle)

## How to Run

1. Clone this repo:
```bash
   git clone <your-repo-url>
   cd <your-repo-name>
```
2. Open the notebook of interest in Jupyter, Kaggle, or Google Colab.
3. Each notebook is self-contained — install dependencies as prompted at the top of the notebook, and update the dataset path if running outside Kaggle.

## Author

<!-- your name / GitHub profile / LinkedIn -->
