# Report: House Prices AI HOMEWORK BY GRISELD HOLLIDRI

## Cross-Method Comparison Questions
### Do clustering results align with classification labels?
Yes. The K-Means algorith found 3 distinct clusters (Starter, Average, Premium homes) that directly mirrored the Cheap, Medium, and Expensive price tiers from the Classification task. It did this completely unsupervised, without ever looking at the actual sale prices.

### Do regression-important features match classification-important features?
Yes. Both the Random Forest Regressor and the Random Forest Classifier identified OverallQual and GrLivArea as the two most dominant features for predicting house prices.

### What does each method “see” differently in the data?
* Regression: Predicts an exact dollar amount on a continuous sliding scale.
* Classification: Sorts houses into strict, predefined categories regardless of how close it is to the boundary.
* Clustering: sees the natural, geometric shapes of the data. It ignores human labels and simply groups houses that physically look alike based on their features.



## Deliverables
### Key Insights
The housing market is highly predictable for average homes. Buyers usually pay premium prices for space and overall quality. Furthermore, the market naturally segments itself into distinct groups, such as starter homes and luxury developments, that can be identified purely by the physical attributes of the houses without even knowing their price tags.

### Surprising Findings
The most surprising finding was how perfectly the unsupervised clustering method which was hidden from the SalePrice data managed to recreate the same price tiers we manually built in the classification task. The physical features of a home are a near-perfect proxy for its financial value.

### Limitations of Each Method
* Regression Limitations: It struggles heavily with extreme outliers. It consistently underestimated the price of ultra-luxury homes because high-end real estate is driven by subjective, custom features not easily captured by basic math.
* Classification Limitations: It forces prices into strict, artificial buckets. Because the boundaries are purely mathematical, a microscopic price difference can inaccurately change a home's entire category.
* Clustering Limitations: You have to guess the correct number of clusters using estimates like the Elbow Method. Also, it doesn't predict a target variable; it only groups data, leaving the human to interpret what those groups actually mean.