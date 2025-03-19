import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load the dataset
breast_cancer_data = load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)
df["label"] = breast_cancer_data.target  # Malignant: 0, Benign: 1

# Save to CSV
df.to_csv("data/breast_cancer.csv", index=False)
print("✅ Dataset saved as data/breast_cancer.csv")
