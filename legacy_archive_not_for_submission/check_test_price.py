import pandas as pd
df = pd.read_csv('Cars_HW_data.csv')
template = pd.read_excel('Cars_HW_template.xlsx')
unlabeled_ids = set(template['ID'])
test_submit = df[df['ID'].isin(unlabeled_ids)].copy()
print(test_submit['Price'].head())
print(f"Number of test prices: {test_submit['Price'].notnull().sum()}")
