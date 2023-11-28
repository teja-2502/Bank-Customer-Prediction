
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import joblib
data = pd.read_csv(r"C:\Users\DELL\Desktop\Bank churn prediction project files\bank data.csv")
data.head()
data.describe()
data.shape
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
data.head()
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data = pd.get_dummies(data, columns=['Geography'])
data
X = data.drop('Exited', axis=1)
y = data['Exited']
data['Exited'].value_counts()
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
y_resampled.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm = svm.SVC(random_state=43)
svm.fit(X_train,y_train)
y_pred_svc = svm.predict(X_test)
accuracy_score(y_test,y_pred_svc)
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)
accuracy_score(y_test,y_pred_knn)
dt = DecisionTreeClassifier(random_state=43)
dt.fit(X_train,y_train)
y_pred_dt = dt.predict(X_test)
accuracy_score(y_test,y_pred_dt)
rf = RandomForestClassifier(random_state=43)
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
accuracy_score(y_test,y_pred_rf)
Model_Acc=pd.DataFrame({'Models':["SVC","KNN","DT","RF"],
                         "Accuracy":[accuracy_score(y_test,y_pred_svc)*100,
                                     accuracy_score(y_test,y_pred_knn)*100,
                                     accuracy_score(y_test,y_pred_dt)*100,
                                     accuracy_score(y_test,y_pred_rf)*100]})
print(Model_Acc)
X_res=scaler.fit_transform(X_resampled)
rf.fit(X_resampled,y_resampled)
joblib.dump(rf,'churn_predict_model')
model = joblib.load('churn_predict_model')
data.columns
def predict_churn(X_resampled):
    result = rf.predict([X_resampled])
    if result == 1:
        return "Customer is about to leave"
    else:
        return "Customer is not willing to leave"

def get_user_input():
    input_data = []
    print("Please enter customer data:")
    try:
        input_data.append(float(input("CreditScore: ")))
        input_data.append(float(input("Gender (0 for Female, 1 for Male): ")))
        input_data.append(float(input("Age: ")))
        input_data.append(float(input("Tenure: ")))
        input_data.append(float(input("Balance: ")))
        input_data.append(float(input("NumOfProducts: ")))
        input_data.append(float(input("HasCrCard (0 for No, 1 for Yes): ")))
        input_data.append(float(input("IsActiveMember (0 for No, 1 for Yes): ")))
        input_data.append(float(input("EstimatedSalary: ")))
        input_data.append(float(input("Geography_France (0 for No, 1 for Yes): ")))
        input_data.append(float(input("Geography_Germany (0 for No, 1 for Yes): ")))
        input_data.append(float(input("Geography_Spain (0 for No, 1 for Yes): ")))
        return input_data
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return None

user_input = get_user_input()
if user_input:
    result = predict_churn(user_input)
    print(result)
