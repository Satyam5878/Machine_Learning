print("Test")

from sklearn.preprocessing import LabelEncoder
def test_analysis():
        """One at time"""
        #sns.countplot  (x='occupation',hue='income',data=adult_data)
        #sns.countplot(x='education',hue='income',data=adult_data)
        #sns.countplot(x='hours_per_week',hue='income',data=adult_data)
        #sns.countplot(x='race',hue='income',data=adult_data)
        #sns.countplot(x='sex',hue='income',data=adult_data)
        #sns.countplot(x='age',hue='income',data=adult_data)



        #plt.xticks(rotation=45)
        #plt.show()
        #print("Done")
        
def test_unique_category():
        category = {column: list(adult_data[column].unique())for column in adult_data.columns if adult_data[column].dtype == 'object'}

        for item in category.keys():print(item+" "+str(category[item]))
        

def test_label_encoder():
        lbl_enc = LabelEncoder()
        lbl_enc.fit(Code.X["workclass"])
        print(lbl_enc.classes_)
        print("Done")
