import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 55

# Loading the dataset
df = pd.read_csv("../utils/heart.csv")
df.head()

# One-hot encoding categorical variables
cat_variables = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df = pd.get_dummies(data = df, prefix = cat_variables, columns = cat_variables)

# Removing target feature 'HeartDisease'
features = [i for i in df.columns if i not in 'HeartDisease']

# Splitting dataset
x_train, x_val, y_train, y_val = train_test_split(df[features], df['HeartDisease'],
                                                  train_size=0.8, random_state=RANDOM_STATE,
                                                  shuffle=True)

# Training trees with different 'min_samples_split' values
min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700]

accuracy_list_train = []
accuracy_list_val = []
for min_samples_split in min_samples_split_list:
    model = DecisionTreeClassifier(min_samples_split=min_samples_split, random_state=RANDOM_STATE).fit(x_train, y_train)

    # Predicted values for both samples
    predictions_train = model.predict(x_train)
    predictions_val = model.predict(x_val)

    # Accuracy values
    accuracy_train = accuracy_score(predictions_train, y_train)
    accuracy_val = accuracy_score(predictions_val, y_val)

    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

# Plotting results, most optimal min_samples_split value is: 30
plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
plt.show()


# Training trees with different 'max_depth' values
max_depth_list = [1, 2, 3, 4, 8, 16, 32, 64, None]

accuracy_list_train.clear()
accuracy_list_val.clear()
for max_depth in max_depth_list:
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE).fit(x_train, y_train)

    # Predicted values for both samples
    predictions_train = model.predict(x_train)
    predictions_val = model.predict(x_val)

    # Accuracy values
    accuracy_train = accuracy_score(predictions_train, y_train)
    accuracy_val = accuracy_score(predictions_val, y_val)

    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

# Plotting results, most optimal max_depth value is: 4
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
plt.show()


# Training a final decision tree
decision_tree_model = DecisionTreeClassifier(min_samples_split = 30, max_depth = 4, random_state = RANDOM_STATE).fit(x_train,y_train)
print(f"Metrics train for final decision tree:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(x_train),y_train):.4f}\n"
      f"Metrics validation for final decision tree:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(x_val),y_val):.4f}\n")



# Training random forest with different 'min_samples_split' values
accuracy_list_train.clear()
accuracy_list_val.clear()
for min_samples_split in min_samples_split_list:
    model = RandomForestClassifier(min_samples_split=min_samples_split, random_state=RANDOM_STATE).fit(x_train, y_train)

    # Predicted values for both samples
    predictions_train = model.predict(x_train)
    predictions_val = model.predict(x_val)

    # Accuracy values
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)

    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

# Plotting results, most optimal min_samples_list value is: 10
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.show()


# Training random forest with different 'max_depth' values
accuracy_list_train.clear()
accuracy_list_val.clear()
for max_depth in max_depth_list:
    model = RandomForestClassifier(max_depth=max_depth, random_state=RANDOM_STATE).fit(x_train, y_train)

    # Predicted values for both samples
    predictions_train = model.predict(x_train)
    predictions_val = model.predict(x_val)

    # Accuracy values
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)

    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

# Plotting results, most optimal max_depth value is: 16
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.show()


# Training random forest with different 'n_estimators' values
n_estimators_list = [10,50,100,500] #number of Decision Trees that make up the Random Forest

accuracy_list_train.clear()
accuracy_list_val.clear()
for n_estimators in n_estimators_list:
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE).fit(x_train, y_train)

    # Predicted values for both samples
    predictions_train = model.predict(x_train)
    predictions_val = model.predict(x_val)

    # Accuracy values
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)

    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

# Plotting results, most optimal n_estimators value is: 100
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.show()

# Training final random forest
random_forest_model = RandomForestClassifier(n_estimators = 100, max_depth = 16, min_samples_split = 10).fit(x_train,y_train)
print(f"Metrics train for final random forest:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(x_train),y_train):.4f}\n"
      f"Metrics validation for final random forest:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(x_val),y_val):.4f}\n")



# Training XGBoost model
xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.1,verbosity=1, random_state=RANDOM_STATE)
xgb_model.fit(x_train,y_train, eval_set=[(x_val,y_val)])


# The best round of training was 16, with a log loss of 4.3948
print(f"Metrics train for final xgb model:\n\tAccuracy score: {accuracy_score(xgb_model.predict(x_train),y_train):.4f}\n"
      f"Metrics test for final xgb model:\n\tAccuracy score: {accuracy_score(xgb_model.predict(x_val),y_val):.4f}\n")
