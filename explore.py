import pandas as pd

df = pd.read_csv('Cars_HW_data.csv')
temp = pd.read_excel('Cars_HW_template.xlsx')

print(f"Data shape: {df.shape}")
print(f"Template shape: {temp.shape}")

# Check overlap of IDs
overlap = set(temp['ID']).intersection(set(df['ID']))
print(f"Number of template IDs in data: {len(overlap)}")

if len(overlap) > 0:
    overlap_df = df[df['ID'].isin(template_ids)] if 'template_ids' in locals() else df[df['ID'].isin(temp['ID'])]
    print(f"Number of overlapping rows with non-null Price: {overlap_df['Price'].notnull().sum()}")
