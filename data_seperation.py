import pandas as pd

# Load the file that has predicted_match column
df_test = pd.read_csv("results/new_data_output.csv")

# Filter and save predicted_match = 0
df_pred_0 = df_test[df_test['predicted_match'] == 0]
df_pred_0.to_csv("seperated/predicted_0.csv", index=False)
print(f"✅ Saved {len(df_pred_0)} records with predicted_match = 0 to 'predicted_0.csv'")

# Filter and save predicted_match = 1
df_pred_1 = df_test[df_test['predicted_match'] == 1]
df_pred_1.to_csv("seperated/predicted_1.csv", index=False)
print(f"✅ Saved {len(df_pred_1)} records with predicted_match = 1 to 'predicted_1.csv'")

print(f" ratio for 1:0 is  {len(df_pred_1)}:{len(df_pred_0)} and true value percentage is {(len(df_pred_1)/len(df_pred_0))*100}%'")
