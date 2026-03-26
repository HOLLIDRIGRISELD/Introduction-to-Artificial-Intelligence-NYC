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



# Step 2: Classification Task
### Model Performance
The Logistic Regression model achieved an accuracy of 81.85% with an F1 score of 0.8140. The Random Forest model performed with an accuracy of 81.16% and an F1 score of 0.8062.

### 1. Which features distinguish expensive houses?
Just like in the regression task, Overall Quality and Square Footage (Above Ground Living Area) are the main drivers. A newer age (Year Built) also helps push a house from the Medium tier into the Expensive tier.

### 2. What mistakes does the model make?
Looking at the Confusion Matrix, It mostly makes "adjacent" mistakes, like confusing a Medium house for a Cheap or Expensive one. It almost never makes extreme errors, like predicting a Cheap house is an Expensive one.

### 3. Is the classification boundary meaningful or arbitrary?
TIt is completely arbitrary. Because we forced the prices into strict 33% buckets, a $1 difference could flip a house from "Medium" to "Expensive." Real-world buyers don't view a $1 difference as a whole new pricing tier.

### 4. Compare interpretability vs. regression
Classification is much easier to explain to a general audience ("this house is in the top tier"), but Regression is ultimately more useful because it predicts exact dollar amounts instead of throwing away detailed pricing data.