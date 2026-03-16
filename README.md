# Yelp Business Analysis: Indiana & Pennsylvania
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project-Academic-orange)
![Data](https://img.shields.io/badge/Data-Yelp%20Academic%20Dataset-lightgrey)

## Overview

This project analyzes Yelp business data with a focused academic scope on **Indiana (IN)** and **Pennsylvania (PA)**.  

The goal is to:
- Filter the full Yelp dataset
- Analyze business distributions
- Visualize rating trends
- Identify dominant business categories
- Compare filtered data to the full dataset

---

## Dataset

Source: Yelp Academic Dataset
This visualization demonstrates the reduction in dataset size after filtering for Indiana and Pennsylvania. The comparison highlights the methodological narrowing of scope for focused regional analysis.
We:
1. Loaded the full business dataset
2. Filtered businesses located in Indiana and Pennsylvania
3. Stored the filtered dataset
4. Generated statistical visualizations
![Dataset Comparison](assets/images/dataset_comparison.png)
---

## Methodology

1. Load full dataset
2. Filter for IN and PA
3. Perform exploratory data analysis
4. Generate visualizations using Matplotlib
5. Save outputs to `assets/images/`

All visualizations are reproducible by running:

```bash


---

## Visualizations

### 1. Dataset Size Comparison

Compares total businesses in full dataset vs IN & PA subset.

![Dataset Comparison](assets/images/dataset_comparison.png)

---

### 2. Business Distribution by State

Shows relative business density between Indiana and Pennsylvania.

![State Distribution](assets/images/state_distribution.png)

---

### 3. Star Rating Distribution

Displays frequency of Yelp star ratings in IN & PA.

![Star Distribution](assets/images/star_distribution.png)

---

### 4. Top 10 Business Categories

Identifies the most common industries within the filtered dataset.

![Top Categories](assets/images/top_categories.png)

---
Review Count Distribution

Pennsylvania restaurants generally exhibit higher review volume compared to Indiana, indicating stronger Yelp engagement density in PA.

Rating vs Review Count

The correlation between review count and rating provides insight into whether popularity aligns with perceived quality.

Average Reviews by State

On average, Pennsylvania restaurants receive more reviews per establishment compared to Indiana, suggesting greater platform usage or urban density effects.

## Academic Focus

This project emphasizes:

- Reproducible analysis
- Clear data filtering methodology
- Structured visualization
- Clean repository organization
- Separation of raw data and generated outputs

---
## Data Setup

Due to file size limitations, raw Yelp datasets are not included in this repository.

Please manually place:
- yelp_academic_dataset_business.json
- yelp_in_pa_business.json

Inside the `data/` directory before running:
python3 src/main.py


## Key Findings

- **Pennsylvania contains a higher concentration of Yelp businesses** compared to Indiana within the filtered dataset.
- **Most businesses cluster around 3.5 to 4.5 star ratings**, indicating generally positive consumer experiences.
- The **Top 10 business categories dominate a significant share** of the dataset, highlighting strong industry clustering in food, retail, and service sectors.
- Filtering from the full Yelp dataset to only Indiana and Pennsylvania significantly reduced dataset size, demonstrating the impact of geographic scoping in data analysis.
- The statistical summary shows relatively stable rating variance, suggesting consistent review behavior across both states.

These findings support the importance of regional filtering when performing targeted business analysis.
---

Restaurant Analysis Project

This project analyzes restaurant data from Indianapolis (Indiana)
and Philadelphia (Pennsylvania).

Data Source
Processed Yelp restaurant dataset.

Visualizations
1. Restaurant counts by city
2. Top restaurant categories
3. Rating distributions
4. Total reviews comparison

Indiana is represented in blue.
Pennsylvania is represented in red.


## Contributors

- Jay Kelley, Yamini Hariharan and Divya Patel
- INFO-H501 Team 3

---

## Data Setup

Due to file size limitations, raw Yelp datasets are not included in this repository.

Place the following files inside the `data/` directory before running:

- yelp_academic_dataset_business.json
- yelp_in_pa_business.json

Run:
python3 src/main.py

