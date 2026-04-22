# =========================================
# 📌 VIDEO GAME DATA ANALYSIS PROJECT (ULTIMATE MASTER)
# =========================================

# ---------- 1) IMPORT LIBRARIES ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8,5)


# ---------- 2) LOAD DATA ----------
df = pd.read_csv(r"C:\Users\hp\Downloads\vgsales.csv")

print("Initial Shape:", df.shape)


# =========================================
# 📊 BASIC EDA (VERY IMPORTANT)
# =========================================

print("\n===== BASIC INFORMATION =====")

print("\nFirst 5 Rows:\n", df.head())

print("\nLast 5 Rows:\n", df.tail())

print("\nColumns:\n", df.columns)

print("\nData Types:\n", df.dtypes)

print("\nMissing Values:\n", df.isnull().sum())

print("\nUnique Values Per Column:\n", df.nunique())

print("\nBasic Statistics:\n", df.describe())

# =========================================
# 🧹 DATA CLEANING (DETAILED)
# =========================================

print("\n===== DATA CLEANING STARTED =====")

# 1. Remove duplicate rows
duplicates = df.duplicated().sum()
print("Duplicate rows:", duplicates)

df = df.drop_duplicates()


# 2. Handle missing values
print("\nMissing values before:\n", df.isnull().sum())

# Drop rows where important columns are missing
df = df.dropna(subset=["Year", "Publisher"])

# Fill numeric columns with median
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical columns with mode
cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    df[col] = df[col].astype(str).str.lower().str.strip()


print("\nMissing values after:\n", df.isnull().sum())


# 3. Fix data types
df["Year"] = df["Year"].astype(int)


# 4. Remove invalid or unrealistic data
df = df[(df["Year"] >= 1980) & (df["Year"] <= 2020)]


# 5. Standardize text data
for col in cat_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()


# 6. Outlier handling (IQR method)
Q1 = df["Global_Sales"].quantile(0.25)
Q3 = df["Global_Sales"].quantile(0.75)
IQR = Q3 - Q1

df = df[(df["Global_Sales"] >= Q1 - 1.5*IQR) & 
        (df["Global_Sales"] <= Q3 + 1.5*IQR)]


# 7. Feature Engineering (NEW 🔥)

# Total regional sales check
df["Total_Regional_Sales"] = df["NA_Sales"] + df["EU_Sales"] + df["JP_Sales"]

# Difference between global and regional
df["Other_Sales"] = df["Global_Sales"] - df["Total_Regional_Sales"]


# 8. Reset index after cleaning
df = df.reset_index(drop=True)

print("\nFinal Shape after cleaning:", df.shape)
print("===== DATA CLEANING COMPLETED =====\n")


# =========================================
# 📊 KPI SUMMARY
# =========================================

print("\n===== KPI SUMMARY =====")
print("Total Games:", len(df))
print("Total Sales:", df["Global_Sales"].sum())
print("Average Sales:", df["Global_Sales"].mean())
print("Top Genre:", df["Genre"].mode()[0])
print("Top Platform:", df["Platform"].mode()[0])


# =========================================
# 📊 DISTRIBUTION VISUALS
# =========================================

# Histogram
sns.histplot(df["Global_Sales"], bins=50)
plt.title("Global Sales Distribution")
plt.show()

# Boxplot
sns.boxplot(x=df["Global_Sales"])
plt.title("Outliers in Sales")
plt.show()

# KDE Plot
sns.kdeplot(df["Global_Sales"], fill=True)
plt.title("Density Curve of Sales")
plt.show()


# =========================================
# 🥧 PIE CHARTS (NEW)
# =========================================

# Genre share
genre_counts = df["Genre"].value_counts().head(6)

genre_counts.plot.pie(autopct='%1.1f%%')
plt.title("Top Genre Share")
plt.ylabel("")
plt.show()

# Platform share
platform_counts = df["Platform"].value_counts().head(6)

platform_counts.plot.pie(autopct='%1.1f%%')
plt.title("Top Platform Share")
plt.ylabel("")
plt.show()


# =========================================
# 📊 TOP ANALYSIS
# =========================================

# Top 10 games
top_games = df.nlargest(10, "Global_Sales")

sns.barplot(x="Global_Sales", y="Name", data=top_games)
plt.title("Top 10 Games")
plt.show()

# Top publishers
top_pub = df.groupby("Publisher")["Global_Sales"].sum().nlargest(10)

top_pub.plot(kind="bar")
plt.title("Top Publishers")
plt.show()


# =========================================
# 📊 CATEGORY ANALYSIS
# =========================================

# Genre count
sns.countplot(y="Genre", data=df)
plt.title("Games by Genre")
plt.show()

# Genre sales
genre_sales = df.groupby("Genre")["Global_Sales"].sum()

genre_sales.plot(kind="barh")
plt.title("Sales by Genre")
plt.show()

# Platform sales
platform_sales = df.groupby("Platform")["Global_Sales"].sum().nlargest(10)

platform_sales.plot(kind="bar")
plt.xticks(rotation=45)
plt.title("Top Platforms")
plt.show()


# =========================================
# 📈 TIME SERIES
# =========================================

year_sales = df.groupby("Year")["Global_Sales"].sum()

sns.lineplot(x=year_sales.index, y=year_sales.values)
plt.title("Year-wise Sales Trend")
plt.show()


# =========================================
# 🌍 REGIONAL ANALYSIS
# =========================================

regions = ["NA_Sales", "EU_Sales", "JP_Sales"]

df[regions].sum().plot(kind="bar")
plt.title("Regional Sales Comparison")
plt.show()


# =========================================
# 📊 RELATIONSHIPS
# =========================================

# Scatter plots
sns.scatterplot(x="NA_Sales", y="Global_Sales", data=df)
plt.title("NA vs Global Sales")
plt.show()

sns.scatterplot(x="EU_Sales", y="Global_Sales", data=df)
plt.title("EU vs Global Sales")
plt.show()


# =========================================
# 🔥 ADVANCED VISUALS
# =========================================

# Heatmap
corr = df[["NA_Sales","EU_Sales","JP_Sales","Global_Sales"]].corr()

sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df[["NA_Sales","EU_Sales","JP_Sales","Global_Sales"]])
plt.show()


# =========================================
# 📊 STATISTICAL TESTS
# =========================================

corr, p = stats.pearsonr(df["NA_Sales"], df["Global_Sales"])
print("\nCorrelation NA vs Global:", corr, "P-value:", p)


# =========================================
# 🤖 MACHINE LEARNING
# =========================================

df_ml = df.select_dtypes(include=["float64","int64"])

X = df_ml.drop(columns=["Global_Sales"])
y = df_ml["Global_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nMODEL PERFORMANCE")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Actual vs predicted
plt.scatter(y_test, y_pred)
plt.title("Actual vs Predicted")
plt.show()


# =========================================
# 📌 FINAL INSIGHTS
# =========================================

print("\nKEY INSIGHTS:")
print("✔ Majority games have low sales")
print("✔ Few games dominate revenue")
print("✔ Action genre is most popular")
print("✔ NA region leads in sales")
print("✔ Strong regional correlation")
