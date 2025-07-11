import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load dataset
file_path = 'Cleaned_Bank_Client_Dataset.csv'
df = pd.read_csv(file_path)

# Basic sanity check
print(f"📦 Total records: {len(df)}")
print("🎯 Unique credit scores:", df['credit_score'].nunique())

# 1. Average credit score by default status
avg_credit_by_default = df.groupby('defaulted')['credit_score'].mean()
print("\n📊 Average Credit Score by Default Status:")
print(avg_credit_by_default)

# 2. Credit score ranges and default rate
bins = [300, 500, 600, 700, 800, 900]
labels = ['300–499', '500–599', '600–699', '700–799', '800–900']
df['credit_bucket'] = pd.cut(
    df['credit_score'], bins=bins, labels=labels, right=False)

# Default rate per credit score bucket
bucket_summary = df.groupby('credit_bucket')[
    'defaulted'].agg(['count', 'sum', 'mean'])
bucket_summary.rename(
    columns={'count': 'Total', 'sum': 'Defaults', 'mean': 'Default Rate'}, inplace=True)

print("\n📈 Default Rate by Credit Score Bucket:")
print(bucket_summary)

# Optional: Drop helper column after analysis
# df.drop('credit_bucket', axis=1, inplace=True)


# Load dataset
file_path = 'Cleaned_Bank_Client_Dataset.csv'
df = pd.read_csv(file_path)

# Create loan-to-income ratio column
df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']

# Optional: check if any division by zero occurred
zero_income = df[df['annual_income'] == 0]
if not zero_income.empty:
    print("⚠️ Warning: Some rows have zero income. Consider handling them:")
    print(zero_income)

# Preview the new column
print("\n📌 Sample rows with new 'loan_to_income_ratio' column:")
print(df[['annual_income', 'loan_amount', 'loan_to_income_ratio']].head())


# --- Statistics for loan_to_income_ratio ---

print("\n📊 Descriptive Statistics for 'loan_to_income_ratio':")
print(df['loan_to_income_ratio'].describe())

# Optional: Percentile distribution
percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
print("\n📈 Loan-to-Income Ratio Percentiles:")
print(df['loan_to_income_ratio'].quantile(percentiles))

# Optional: How many are over common thresholds?
thresholds = [0.2, 0.3, 0.5, 0.7, 1.0]
print("\n🔍 Loan-to-Income Ratio Counts over Thresholds:")
for t in thresholds:
    count = (df['loan_to_income_ratio'] > t).sum()
    print(f" > {t:.1f}: {count} rows ({count/len(df)*100:.2f}%)")

# Optional: Visualize with histogram (if matplotlib is available)
try:
    import matplotlib.pyplot as plt
    plt.hist(df['loan_to_income_ratio'], bins=50,
             color='skyblue', edgecolor='black')
    plt.axvline(df['loan_to_income_ratio'].mean(), color='red',
                linestyle='dashed', linewidth=1, label='Mean')
    plt.title("Distribution of Loan-to-Income Ratio")
    plt.xlabel("Loan Amount / Annual Income")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
except ImportError:
    print("\n📉 Plot skipped (matplotlib not installed)")


# --- STEP 1: Risk flag based on ratio ---
df['high_risk_ratio_flag'] = df['loan_to_income_ratio'] > 0.5  # mark if ratio > 50%

# --- STEP 2: Distribution plot by default status ---
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='loan_to_income_ratio', hue='defaulted',
             bins=50, kde=True, palette='Set1', alpha=0.6)
plt.axvline(0.5, color='black', linestyle='--', label='Risk Threshold (0.5)')
plt.title("Loan-to-Income Ratio Distribution by Default Status")
plt.xlabel("Loan Amount / Annual Income")
plt.ylabel("Count")
plt.legend(title='Defaulted')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='credit_score',
    y='loan_to_income_ratio',
    hue='defaulted',
    palette={0: 'green', 1: 'red'},
    alpha=0.6
)
plt.title("📊 Credit Score vs Loan-to-Income Ratio (Colored by Default Status)")
plt.xlabel("Credit Score")
plt.ylabel("Loan Amount / Annual Income")
plt.legend(title="Defaulted", labels=["No (0)", "Yes (1)"])
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# Bin the credit score and loan-to-income ratio
df['credit_bin'] = pd.cut(df['credit_score'], bins=[300, 580, 670, 740, 800, 850],
                          labels=["Poor", "Fair", "Good", "Very Good", "Excellent"])
df['ratio_bin'] = pd.cut(df['loan_to_income_ratio'], bins=[0, 0.2, 0.4, 0.6, 1.0, float('inf')],
                         labels=["<20%", "20-40%", "40-60%", "60-100%", ">100%"])

# Pivot table for heatmap
heatmap_data = df.pivot_table(
    index='credit_bin', columns='ratio_bin', values='defaulted', aggfunc='mean')

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='Reds', fmt=".2f")
plt.title("🔥 Default Rate Heatmap by Credit Score & Loan-to-Income Ratio Bins")
plt.xlabel("Loan-to-Income Ratio")
plt.ylabel("Credit Score Category")
plt.tight_layout()
plt.show()


print("📈 Correlation Matrix:")
print(df[['loan_to_income_ratio', 'credit_score', 'defaulted']].corr())
