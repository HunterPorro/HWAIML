import pandas as pd
df = pd.read_csv('Cars_HW_data.csv')
template = pd.read_excel('Cars_HW_template.xlsx')
print(f"Cars_HW_data shape: {df.shape}")
print(f"Missing prices in Cars_HW_data: {df['Price'].isna().sum()}")
print(f"Template shape: {template.shape}")
