# Dataset source and limitations

The training script downloads the public `food_wastage_data.csv` file from:

- Repository: https://github.com/Ashutosh-dutta07/Smart-Food-Waste-Intelligence
- Raw CSV: https://raw.githubusercontent.com/Ashutosh-dutta07/Smart-Food-Waste-Intelligence/main/food_wastage_data.csv

The public CSV describes restaurant/event food wastage using columns such as food type, guest count, event type, quantity prepared, storage conditions, seasonality and wastage amount. Review the upstream repository and any applicable terms before redistributing the dataset. This project downloads the CSV from the source rather than bundling a copy.

**Limitations:** The dataset does not document a reliable physical unit for `Quantity of Food` or `Wastage Food Amount`. Accordingly, the dashboard labels predictions as *dataset units*, not kilograms. Its held-out test metrics describe only this dataset split and do not validate predictions for a specific restaurant, donor or local NGO. The model is a student prototype, not a food-safety tool.
