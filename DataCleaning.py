import pandas as pd

# Load the dataset
df = pd.read_csv('Bank_Client_Dataset.csv')

# Compute loan-to-income ratio
df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']


# --- Strategy 1: Fixing false positives (likely to default) ---
df.loc[(df['credit_score'] < 500) & (
    df['loan_to_income_ratio'] > 0.1), 'defaulted'] = 1
df.loc[(df['credit_score'].between(500, 600, inclusive='left'))
       & (df['loan_to_income_ratio'] > 0.175), 'defaulted'] = 1
df.loc[(df['credit_score'].between(600, 650, inclusive='left'))
       & (df['loan_to_income_ratio'] > 0.25), 'defaulted'] = 1
df.loc[(df['credit_score'].between(650, 700, inclusive='left'))
       & (df['loan_to_income_ratio'] > 0.4), 'defaulted'] = 1
df.loc[(df['credit_score'].between(700, 750, inclusive='left'))
       & (df['loan_to_income_ratio'] > 0.6), 'defaulted'] = 1
df.loc[(df['credit_score'].between(750, 800, inclusive='left'))
       & (df['loan_to_income_ratio'] > 1.0), 'defaulted'] = 1

# --- Strategy 2: Fixing false negatives (should NOT be defaulted) ---
df.loc[(df['credit_score'] > 800) & (
    df['loan_to_income_ratio'] < 0.8), 'defaulted'] = 0
df.loc[(df['credit_score'].between(750, 800, inclusive='left'))
       & (df['loan_to_income_ratio'] < 0.65), 'defaulted'] = 0
df.loc[(df['credit_score'].between(700, 750, inclusive='left'))
       & (df['loan_to_income_ratio'] < 0.4), 'defaulted'] = 0
df.loc[(df['credit_score'].between(650, 700, inclusive='left'))
       & (df['loan_to_income_ratio'] < 0.25), 'defaulted'] = 0
df.loc[(df['credit_score'].between(600, 650, inclusive='left'))
       & (df['loan_to_income_ratio'] < 0.15), 'defaulted'] = 0
df.loc[(df['credit_score'].between(500, 600, inclusive='left'))
       & (df['loan_to_income_ratio'] < 0.07), 'defaulted'] = 0

# Save the cleaned dataset
df.to_csv('Cleaned_Bank_Client_Dataset.csv', index=False)

# Print summary
print("📁 Cleaned file saved as: Cleaned_Bank_Client_Dataset.csv")
