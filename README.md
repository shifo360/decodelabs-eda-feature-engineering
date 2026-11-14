# DecodeLabs Data Science - Project 1: Advanced EDA & Feature Engineering

I cleaned an e-commerce orders dataset and built new features from it,
using only pandas and numpy.

The dataset comes from DecodeLabs (1,200 orders, 14 columns). It is not
included in this repo.

## How to run

1. Install the libraries: `pip install pandas numpy`
2. Put `Dataset for Data Analytics - Sheet1.csv` in the same folder as the script
3. Run `python eda_feature_engineering.py`

The script prints every step and saves the result as `orders_cleaned.csv`
(1,200 rows, 31 columns). I wrote and ran it with pandas 3.

## What the script does

**1. Explore.** Checks shape, types, missing values and duplicates. There are no
duplicates, and `TotalPrice` always equals `Quantity x UnitPrice`.

**2. Missing values.** Only `CouponCode` has missing values (309 rows, 25.75%).
The slides suggest KNN above 20%, but KNN needs numbers and this column is
text. An empty coupon just means no coupon was used, so I filled it with
`NoCoupon`.

**3. Outliers.** IQR bounds on the four numeric columns, values capped with
`np.clip` so no rows are lost. Only `TotalPrice` had outliers: 8 orders were
capped at 3330.41 (the max was 3456.40). They are all real orders of 5 units
at about $667-$691, so the change is small (mean 1053.97 to 1053.64).

**4. New features (8).** `OrderYear`, `OrderMonth`, `OrderDayOfWeek`,
`IsWeekend`, `HasCoupon`, `ProductCategory`, `PriceTier`, `IsBulkOrder`.

**5. Encoding and correlated features.** One-hot encoding for the text columns.
Pairs with correlation above 0.80 are reduced by dropping the feature that is
less correlated with `TotalPrice`. This removed `IsBulkOrder`,
`PriceTier_Premium`, and one of `HasCoupon` / `CouponCode_NoCoupon` (they are
exact opposites).

**6. Imputation comparison (extra).** The data has no missing numbers, so I hid
10% of `UnitPrice` at random and filled it back with four methods. KNN is
written in numpy.

| Method | RMSE on hidden values | Std |
|---|---|---|
| Mean | 199.25 | 187.63 |
| Median | 199.05 | 187.64 |
| Group median (by Product) | 200.87 | 187.83 |
| KNN (k=5) | 16.34 | 196.78 |

The real std of `UnitPrice` is 197.18. Mean and median shrink the spread,
KNN keeps it.

## Things to keep in mind

- `TotalPrice` is only an example target. It is exactly `Quantity x UnitPrice`,
  so its correlations with those columns are mechanical.
- KNN does so well partly because it uses `Quantity` and `TotalPrice`, which
  together give `UnitPrice` exactly.
- The correlation step removed two of the most useful new features, and left
  only `PriceTier_Mid` from the price tiers.
- The coupon, weekend and product category features show almost no relation to
  `TotalPrice`. The data looks synthetic.
- 487 cancelled or pending orders still have tracking numbers.
