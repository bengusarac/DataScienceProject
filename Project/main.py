
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing,metrics
from sklearn.preprocessing import MinMaxScaler,StandardScaler,normalize
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.cluster import  KMeans
from scipy.stats import skew

data = pd.read_csv("disney_plus_shows.csv")
data = data.drop(columns="writer")
data = data.drop(columns="awards")

data.columns = ['imdb_id', 'title',
              'plot', 'type',
              'rating', 'years',
              'release_year', 'date_added',
              'duration', 'genre',
              'director', 'cast','language','country',
              'meta_score','imdb_ratings','imdb_votes']


#Dimensionality Reduction

#Recursive Feature Elimination
data.duplicated().value_counts()
data.drop_duplicates(inplace= True)
data.duplicated().value_counts()

#Data Cleaning

#Missing value
data["director"]=data["director"].fillna("Unknown")
data["country"]=data["country"].fillna("Unknown")
data["plot"]=data["plot"].fillna("Unknown")
data["type"]=data["type"].fillna("Unknown")
data["genre"]=data["genre"].fillna("Unknown")
data["cast"]=data["cast"].fillna("Unknown")
data["language"]=data["language"].fillna("Unknown")
data["rating"]=data["rating"].fillna("Unknown")

data["duration"].replace("min","",regex=True,inplace=True)
data["duration"].replace("1 h","60",regex=True,inplace=True)
data.duration = pd.to_numeric(data.duration)
duration_median = data["duration"].median()
data["duration"]=data["duration"].fillna(duration_median)
data["duration"]=data["duration"].astype("int64")

meta_median = data["meta_score"].median()
data["meta_score"].fillna(meta_median,inplace=True)

imdb_ratings_median = data["imdb_ratings"].median()
data["imdb_ratings"].fillna(imdb_ratings_median,inplace=True)

data.imdb_votes = data.imdb_votes.str.replace(",","")
data.imdb_votes = pd.to_numeric(data.imdb_votes)
imdb_votes_median = data["imdb_votes"].median()
data["imdb_votes"].fillna(imdb_votes_median,inplace=True)
data["imdb_votes"]=data["imdb_votes"].astype("int64")

#Feature Engineering

#Deal with date
data["release_year"] = pd.to_datetime(data["release_year"])
data["year"] = data["release_year"].dt.year
data["month"] = data["release_year"].dt.month
data["day"] = data["release_year"].dt.day

#Combining/Splitting Features
data = data.drop(columns="years")
data.dropna(subset=['imdb_id','title','release_year','year','month','day'],inplace=True)

marvel_keywords = ['marvel', 'black widow', 'captain america', 'iron man','hawkeye','falcon','spider-man','avengers'
    ,'doctor strange','thor','black panther','agent carter','runaways','inhumans','rocket & groot'
    ,'x-men','shang-chi','ant-man','guardians of the galaxy','loki','wanda vision','fantastic four','hulk']
pattern = '|'.join(marvel_keywords)

data['marvel'] = data['title'].str.lower().str.contains(pattern)

print("Marvel Shows in Disney+")
data['marvel'].value_counts()

marvel= data[data['marvel'] == True]
print(marvel.shape)

#Normalization
print("Normalize of imdb ratings")
mi = np.array(data['imdb_ratings'])
normalized_mi = preprocessing.normalize([mi])
print(normalized_mi)

print("Normalize of meta score")
ms = np.array(data['meta_score'])
normalized_ms = preprocessing.normalize([ms])
print(normalized_ms)

#Min-max normalization
minmaxn = marvel.iloc[:,[13,14]]
print("Min-max normalization  of Marvel imdb ratings and meta score")
scaler = MinMaxScaler()
model=scaler.fit(minmaxn)
scaled_data=model.transform(minmaxn)
print(scaled_data)

fig, a = plt.subplots(1,2)
classes = np.random.randint(0,10,54)
a[0].scatter(minmaxn['meta_score'],minmaxn['imdb_ratings'], c = classes)
a[0].set_title("Original data")
a[1].scatter(scaled_data[:,0],scaled_data[:,1],c = classes)
a[1].set_title("MinMax scaled data")
print(plt.show())

#EDA PART

print(data.isnull().sum())
print(data.info())

sns.histplot(data=data, x='imdb_ratings', kde=True)
print(plt.show())
d = data.iloc[:,14]
print(skew(d, axis=0, bias=True))

print('Disney+ Shows Added Month')
data.month.value_counts()
sns.countplot(y = data.month, palette = 'rainbow')
plt.title('Disney+ Shows Added Month')
plt.xlabel('Frequency')
plt.ylabel('Month')
plt.show()

print("Duration")
data.duration.describe()
sns.histplot(data.duration, color= "violet")
plt.title('Duration')
plt.show()
sns.boxplot(data.duration, color= "skyblue")
plt.show()

sns.countplot(y = marvel.rating)
plt.title("Rating of Disney+'s Marvel Shows")
plt.show()

sns.countplot(y = marvel.imdb_ratings)
plt.title("Imdb Rating of Disney+'s Marvel Shows")
plt.show()

plt.scatter(marvel["imdb_ratings"],marvel["release_year"])
plt.title("Imdb rating of Disney+'s Marvel Shows")
plt.show()

sns.lineplot(x = 'meta_score', y = 'release_year', data = marvel)
plt.title("Meta score of Disney+'s Marvel Shows")
plt.show()

print('Types of Marvel Shows')
marvel.type.value_counts()
sns.countplot(x = marvel.type, palette = 'rainbow')
plt.title('Types of Marvel Shows')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()

dataNew = data.copy()

def LABEL_ENCODING(c1):
    label_encoder = preprocessing.LabelEncoder()
    dataNew[c1]= label_encoder.fit_transform(dataNew[c1])
    dataNew[c1].unique()

LABEL_ENCODING("imdb_id")
LABEL_ENCODING("title")
LABEL_ENCODING("plot")
LABEL_ENCODING("type")
LABEL_ENCODING("rating")
LABEL_ENCODING("release_year")
LABEL_ENCODING("date_added")
LABEL_ENCODING("genre")
LABEL_ENCODING("director")
LABEL_ENCODING("cast")
LABEL_ENCODING("language")
LABEL_ENCODING("country")
LABEL_ENCODING("imdb_ratings")
LABEL_ENCODING("rating status")
LABEL_ENCODING("marvel")

dataNew["meta_score"]=dataNew["meta_score"].astype("int64")
dataNew["year"]=dataNew["year"].astype("int64")
dataNew["month"]=dataNew["month"].astype("int64")
dataNew["day"]=dataNew["day"].astype("int64")

print(dataNew)
print(dataNew.info())

scaler = StandardScaler()
scaler.fit(dataNew.drop('meta_score',axis = 1))

scaled_features = scaler.transform(dataNew.drop('meta_score',axis = 1))
dn_feat = pd.DataFrame(scaled_features,columns = dataNew.columns.drop('meta_score'))
print(dn_feat.head())

#Logistic Regression

X = dn_feat
y = dataNew['meta_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train,y_train)
pred = logmodel.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred,zero_division=1))

log_reg_acc = logmodel.score(X_test,y_test)
print('Accuracy:',log_reg_acc)

#K-Means

data1 = dataNew.iloc[:,13:16]

scaling=StandardScaler()
scaled=scaling.fit_transform(data1)
scaled_df=pd.DataFrame(scaled,columns=data1.columns)

a = []
K = range(1, 10)
for i in K:
    kmean = KMeans(n_clusters=i)
    kmean.fit(data1)
    a.append(kmean.inertia_)

plt.plot(K, a, marker='o')
plt.title('Elbow Method', fontsize=15)
plt.xlabel('Number of clusters', fontsize=15)
plt.ylabel('Sum of Squared distance', fontsize=15)
plt.show()

kmeans = KMeans(n_clusters = 3,random_state = 111)
kmeans.fit(scaled_df)
print(pd.Series(kmeans.labels_).value_counts())


cluster_labels = kmeans.fit_predict(scaled_df)
preds = kmeans.labels_
kmeans_df = pd.DataFrame(dataNew)
kmeans_df['KMeans_Clusters'] = preds
print(kmeans_df.head(10))


high_rating=kmeans_df[kmeans_df['KMeans_Clusters']==0]['meta_score']
rating=kmeans_df[kmeans_df['KMeans_Clusters']==1]['meta_score']
low_rating=kmeans_df[kmeans_df['KMeans_Clusters']==2]['meta_score']

print("Number of movies with high ratings",len(high_rating))
print("Number of movies with ratings ",len(rating))
print("Number of movies with low ratings",len(low_rating))


#KNN
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(pred)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred,zero_division=1))

error_rate= []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize = (10,6))
plt.plot(range(1,40),error_rate,color = 'blue',linestyle = '--',marker = 'o',markerfacecolor='red',markersize = 10)
plt.title('Error Rate vs K')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

knn_acc = metrics.accuracy_score(y_test, pred)
print("Accuracy:",metrics.accuracy_score(y_test, pred))

