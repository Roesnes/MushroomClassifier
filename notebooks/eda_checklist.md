# EDA Checklist

Use this outline to guide your exploratory data analysis of the mushroom dataset.

1. **Data Loading**
   - Inspect the first rows and schema.
   - Verify delimiter handling (`sep=';'`).
   - Count missing values per column.

2. **Target Distribution**
   - Plot class balance (`poisonous` vs `edible`).
   - Compute base rates.

3. **Numeric Features**
   - Summary statistics (mean, std, quartiles).
   - Histograms / KDE plots per class.
   - Outlier detection (boxplots).

4. **Categorical Features**
   - Cardinality per feature.
   - Bar plots / heatmaps for top categories vs. target.

5. **Correlations**
   - Correlation matrix for numeric variables.
   - Cramér's V or mutual information for categorical variables (optional).

6. **Feature Interactions**
   - Pairwise plots or grouped aggregations for interesting interactions.

7. **Data Quality**
   - Investigate rare labels (frequency < 0.5%).
   - Document features with high missingness for imputation strategy.

Document key findings in the project report to justify preprocessing and modeling choices.
