# Step 1: Regression Task Interpretation
### Feature Selection & Engineering Explanation
To keep the model simple but accurate, I focused on five key features that strongly affect house prices: OverallQual, GrLivArea, GarageCars, TotalBsmtSF, and YearBuilt. To fix any missing data, I just replaced the empty spots with the median value for that column.

### 1. Which features influence price the most?
Based on our model, the two most important features are Overall Quality and Above Ground Living Area (Square Footage). These two factors carry the most weight when determining a home's final price.

### 2. Are relationships linear or not?
They are mostly linear, but not completely. We know this because the Random Forest model, made better predictions than the basic Linear Regression model. For example, a home's age might affect its price differently depending on how big the house is.

### 3. Where does the model perform poorly and why?
Both models struggle with extremely expensive homes (over $400,000), almost always guessing a price that is too low. 
* Why? The models consistently underestimate the price of these luxury homes. Luxury homes get their high prices from unique features that aren't in our basic data—like fancy architecture, a prestigious neighborhood, or a custom pool. Plus, there are simply fewer of these mansions in our dataset to train the model on.

### 4. What does this tell us about the housing market?
For normal homes, the market is very logical: buyers are mainly paying for space and good condition. However, the high-end luxury market is much more subjective, emotional, and harder to predict with simple math.


