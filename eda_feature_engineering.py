from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

# the csv has to be in the same folder as this file
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Dataset for Data Analytics - Sheet1.csv"
OUTPUT_PATH = BASE_DIR / "orders_cleaned.csv"
TARGET = "TotalPrice"


# ---------- load and explore ----------
def load_and_explore(path):
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print(df.head())
    print(df.info())
    print(df.describe())
    print("\nMissing values (%):")
    print((df.isnull().mean() * 100).round(2))
    print("\nDuplicate rows:", df.duplicated().sum())
    print("Duplicate OrderIDs:", df["OrderID"].duplicated().sum())

    # TotalPrice should be quantity * unit price
    expected = df["Quantity"] * df["UnitPrice"]
    mismatch = ((expected - df["TotalPrice"]).abs() > 0.01).sum()
    print("Rows where TotalPrice != Quantity x UnitPrice:", mismatch)
    return df


# ---------- missing values ----------
def handle_missing(df):
    df = df.copy()

    report = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_pct": (df.isnull().mean() * 100).round(2),
    })
    print("\nColumns with missing values:")
    print(report[report["missing_count"] > 0])

    # only CouponCode has missing values (25.75%, above the 20% rule).
    # it is text so KNN can't be used. no coupon = no code, so label it
    df["CouponCode"] = df["CouponCode"].fillna("NoCoupon")

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    print("Rows:", len(df))
    return df


# ---------- outliers (IQR, then clip) ----------
def cap_outliers_iqr(df, columns):
    df = df.copy()
    print("\nOutlier capping (IQR):")
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        before_max = df[col].max()
        df[col] = np.clip(df[col], lower, upper)
        print(f"{col}: bounds [{lower:.2f}, {upper:.2f}] | capped {n_outliers} "
              f"values | max {before_max:.2f} -> {df[col].max():.2f}")
    return df


def summarize_capping(df_before, df_after, col=TARGET):
    summary = pd.DataFrame({
        "before capping": df_before[col].describe(),
        "after capping": df_after[col].describe(),
    }).round(2)
    print(f"\n{col} before vs after capping:")
    print(summary)


# ---------- new features ----------
def engineer_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # parts of the date
    df["OrderYear"] = df["Date"].dt.year
    df["OrderMonth"] = df["Date"].dt.month
    df["OrderDayOfWeek"] = df["Date"].dt.day_of_week   # monday = 0
    df["IsWeekend"] = (df["OrderDayOfWeek"] >= 5).astype(int)

    # did the customer use a coupon
    df["HasCoupon"] = (df["CouponCode"] != "NoCoupon").astype(int)

    # group the products into two categories
    df["ProductCategory"] = df["Product"].map({
        "Laptop": "Electronics", "Phone": "Electronics", "Tablet": "Electronics",
        "Monitor": "Electronics", "Printer": "Electronics",
        "Chair": "Furniture", "Desk": "Furniture",
    })

    # unit price split into 3 equal groups
    df["PriceTier"] = pd.qcut(df["UnitPrice"], q=3,
                              labels=["Budget", "Mid", "Premium"])

    # 4 or more units of the same product
    df["IsBulkOrder"] = (df["Quantity"] >= 4).astype(int)

    new_cols = ["OrderYear", "OrderMonth", "OrderDayOfWeek", "IsWeekend",
                "HasCoupon", "ProductCategory", "PriceTier", "IsBulkOrder"]
    print("\nNew features preview:")
    print(df[new_cols].head(10))

    # average total price for each group
    print(f"\nAverage {TARGET} by new features:")
    check_cols = ["ProductCategory", "PriceTier", "IsBulkOrder",
                  "HasCoupon", "IsWeekend"]
    for col in check_cols:
        print(f"\n{col}:")
        print(df.groupby(col, observed=True)[TARGET].mean().round(1))
    return df


# ---------- one-hot encoding and correlated features ----------
def encode_and_remove_collinear(df, target=TARGET, threshold=0.80):
    # ids, addresses and the raw date are not useful as they are
    id_cols = ["OrderID", "CustomerID", "ShippingAddress",
               "TrackingNumber", "Date"]
    df_model = df.drop(columns=id_cols)

    cat_cols = ["Product", "ProductCategory", "PaymentMethod", "OrderStatus",
                "CouponCode", "ReferralSource", "PriceTier"]
    df_model = pd.get_dummies(df_model, columns=cat_cols,
                              drop_first=True, dtype=int)
    print("\nColumns after encoding:", list(df_model.columns))

    X = df_model.drop(columns=[target])
    y = df_model[target]

    # pairs of features with correlation above the threshold
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack()
    pairs = pairs[pairs > threshold]

    # from each pair drop the one that is less correlated with the target
    to_drop = set()
    print(f"\nHighly correlated pairs (> {threshold}):")
    for (a, b), val in pairs.items():
        if a in to_drop or b in to_drop:
            continue
        corr_a = abs(X[a].corr(y))
        corr_b = abs(X[b].corr(y))
        drop = a if corr_a < corr_b else b
        to_drop.add(drop)
        print(f"{a} vs {b}: r={val:.2f} | corr with target "
              f"{corr_a:.2f} vs {corr_b:.2f} -> drop {drop}")
    if not to_drop:
        print("none found")

    X_final = X.drop(columns=list(to_drop))
    df_final = pd.concat([X_final, y], axis=1)

    print("\nDropped features:", sorted(to_drop))
    print("Final shape:", df_final.shape)
    print(df_final.head())
    return df_final


# ---------- compare imputation methods ----------
# the data has no missing numbers, so I hide 10% of UnitPrice,
# fill it back with each method and compare with the real values
def knn_impute_numpy(features, values, hidden, k=5):
    # scale the columns so none of them dominates the distance
    Z = ((features - features.mean()) / features.std()).to_numpy()
    known_Z, miss_Z = Z[~hidden], Z[hidden]
    known_vals = values.to_numpy()[~hidden]

    # distance from each hidden row to every known row
    dist = np.sqrt(((miss_Z[:, None, :] - known_Z[None, :, :]) ** 2).sum(axis=2))
    nearest = np.argpartition(dist, k, axis=1)[:, :k]
    filled = known_vals[nearest].mean(axis=1)

    out = values.copy()
    out[hidden] = filled
    return out


def compare_imputation(df, col="UnitPrice", frac=0.10, seed=42):
    rng = np.random.default_rng(seed)
    true_vals = df[col].copy()
    hidden = rng.random(len(df)) < frac
    masked = true_vals.where(~hidden)

    results = {}
    results["Mean"] = masked.fillna(masked.mean())
    results["Median"] = masked.fillna(masked.median())
    results["Group median (by Product)"] = masked.fillna(
        masked.groupby(df["Product"]).transform("median"))
    results["KNN (k=5)"] = knn_impute_numpy(
        df[["Quantity", "ItemsInCart", "TotalPrice"]], masked, hidden, k=5)

    rows = []
    for name, series in results.items():
        err = series[hidden] - true_vals[hidden]
        rows.append({
            "Method": name,
            "RMSE on hidden values": round(float(np.sqrt((err ** 2).mean())), 2),
            "Mean": round(series.mean(), 2),
            "Std": round(series.std(), 2),
        })
    table = pd.DataFrame(rows).set_index("Method")

    print(f"\nImputation comparison on {col} ({hidden.sum()} values hidden)")
    print(f"True {col}: mean {true_vals.mean():.2f}, std {true_vals.std():.2f}")
    print(table)
    print("Lower RMSE = closer to the real values.")
    print("Std close to the true std = distribution preserved.")


def main():
    df = load_and_explore(CSV_PATH)

    df = handle_missing(df)
    df_before_cap = df.copy()
    numeric_cols = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]
    df = cap_outliers_iqr(df, numeric_cols)
    summarize_capping(df_before_cap, df)

    df = engineer_features(df)
    df_final = encode_and_remove_collinear(df)

    df_final.to_csv(OUTPUT_PATH, index=False)
    print("\nSaved: orders_cleaned.csv")

    compare_imputation(df_before_cap)


if __name__ == "__main__":
    main()