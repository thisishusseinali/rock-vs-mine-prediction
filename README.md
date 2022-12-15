Rock vs Mine Prediction Using ML models
========================
### **Project Structure  :**

```
├── datasets
├── models
│   │── dt_model.sav
│   │── kn_model.sav
│   │── lg_model.sav
│   │── nb_model.sav
│   │── rf_model.sav
│   └── sv_model.sav
├── notebooks
│   └── rock_vs_mine_prediction.ipynb
├── main.py
├── app.py
└── README.md
```

### **Dataset Overview :**
**The data set was used by Gorman and Sejnowski in their study of the classification of sonar signals using a neural network .***
**The project is to train a network to discriminate between sonar signals bounced off a Mines and those bounced off a rock.**
 **Though the dataset is small but still it has approximately 60 features(attributes).**
 **This is a good project to start with, because It is a classification problem, allowing you to practice with perhaps an easier type of supervised learning algorithm.**
**It is a multi-class classification problem (multi-nominal) that may require some specialized handling.**
**It only has 60 attributes and 207 rows, meaning it is small and easily fits into memory .**
**All of the numeric attributes are in the same units and the same scale, not requiring any special scaling or transforms to get started.**

### **Notebook Overview  :**
1. Importing useful libraries
2. Collection of Data
3. Data Preprocessing
4. Split the data into Test Data and Train Data
5. Make Trained ML Models
6. Feed Test Data into our Trained ML Models to predict
