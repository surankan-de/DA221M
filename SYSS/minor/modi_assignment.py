##.ipynb file written and modified such that it can be used for deployment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import warnings
warnings.filterwarnings('ignore')


df1 = pd.read_csv("tmdb_5000_credits.csv")
df2 = pd.read_csv("tmdb_5000_movies.csv")
main_df = df2
main_df['cast'] = df1["cast"]
main_df['crew'] = df1['crew']
main_df = main_df[["overview","popularity","title","vote_average","vote_count","cast","crew","keywords","genres"]]
main_df.fillna(" ",inplace= True)
demodf = main_df[main_df["vote_count"]>main_df['vote_count'].quantile(0.95)]


##weighted rating formula
def weighted_rating(df):
    v=df['vote_count']
    m=main_df['vote_count'].quantile(q=0.95)
    r=df['vote_average']
    c=main_df['vote_average'].mean()
    wr = (r*v+c*m)/(v+m)
    return wr
demodf["weighted_rating"] = demodf.apply(weighted_rating,axis =1)
## scaling popularity to same range
demodf["popularity"] = 10*(demodf["popularity"]/max(demodf["popularity"]))

##return statements are changed
def show_by_wr(n):
    s = ""
    dft = demodf.sort_values(by = 'weighted_rating',ascending = False).head(min(n,len(demodf)))
    toshow = list(dft['title'])
    for i in range(len(toshow)):
        s+=str(i+1)
        s+=" "
        s+=toshow[i]
        s+="\n"
    return s+"\n"

def show_by_wr_and_pop(n):
    demodf1 = demodf
    demodf1["sums"] = demodf["popularity"]+demodf["weighted_rating"]
    dft = demodf1.sort_values(by = 'sums',ascending = False).head(min(n,len(demodf)))
    s = ""
    toshow = list(dft['title'])
    for i in range(len(toshow)):
        s+=str(i+1)
        s+=" "
        s+=toshow[i]
        s+="\n"
    return s+"\n"


tf_vect = TfidfVectorizer(stop_words ="english") ##to remove words like and the etc
tf_matrix = tf_vect.fit_transform(main_df['overview'])

similarities = cosine_similarity(tf_matrix)

keys = {} ## to keep track of the indexes
for i in range(len(main_df['title'])):
    titl = main_df['title'][i]
    keys[titl.lower().replace(' ','')] = i

def contentbasedreco(title,n):
    s = ""
    idx = keys[title.lower().replace(' ','')]
    check = list(enumerate(similarities[idx]))
    check.sort(key = lambda x:x[1],reverse = True)
    s+="the movie recommendations according to the title searched are:\n"
    for i in range(1,min(n+1,len(check)),1):
        s+=str(i)
        s+=" "
        s+=main_df.iloc[check[i][0]]['title']
        s+="\n"
    return s+"\n"

new = ['cast' ,"crew","keywords","genres"]
for col in new:
    main_df[col] = main_df[col].apply(json.loads)


## functions for crew mebers fetch and normal name check
def director(x):
    for a in x:
        if a['job']=='Director':
            return a['name'].lower().replace(' ','')
    return 'NaN'.lower().replace(' ','')

def somecrew(x):
    new=[]
    for a in x[:min(5,len(x))]:
        new.append(a['name'].lower().replace(' ','')) 
    return new

    return []


main_df['director']=main_df['crew'].apply(lambda x: director(x))
main_df['actor']=main_df['cast'].apply(lambda x:somecrew(x))
main_df['genres']=main_df['genres'].apply(lambda x:somecrew(x))
main_df['keywords']=main_df['keywords'].apply(lambda x:somecrew(x))

def metadata(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['actor']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

main_df['meta'] = main_df.apply(metadata, axis=1)

tf_matrix_2 = tf_vect.fit_transform(main_df['meta'])
similarities_2 = cosine_similarity(tf_matrix_2)

def crewbasedreco(title,n):
    s = ""
    idx = keys[title.lower().replace(' ','')]
    check = list(enumerate(similarities_2[idx]))
    check.sort(key = lambda x:x[1],reverse = True)
    s+="the movie recommendations according to the related crews,casts are:\n"
    for i in range(1,min(n+1,len(check)),1):
        s+=str(i)
        s+=" "
        s+=main_df.iloc[check[i][0]]['title']
        s+="\n"
    return s

def overallreco(title,n):
    if(title=="-1"):
        
        s1="according to ratings\n"
        s1  += show_by_wr(n)
        print(" ")
        s2 = "according to both ratings and popularity\n"
        s2 +=show_by_wr_and_pop(n)
        return s1+"\n"+s2+"\n"
    else:
        try:
            s1 = contentbasedreco(title,n)           
            s2 = crewbasedreco(title,n)
            return s1+"\n"+s2+"\n"
        except Exception as e:
            return "not much movies related to this movie! try something else\n"


colldf = pd.read_csv("ratings_small.csv")
colldf.head()
ar = colldf['movieId'].value_counts().index
di = {}
ide = {}
for i in range(len(ar)):
    di[ar[i]] = i+1
    ide[i+1] = ar[i]

for i in range(len(colldf['movieId'])):
    if(i%10000==0):
        print(i)
    colldf['movieId'][i] = di[colldf['movieId'][i]]


r_m = np.ndarray(shape = (np.max(colldf['movieId'].values),np.max(colldf['userId'])))
r_m[colldf['movieId'].values-1,colldf['userId'].values-1] = colldf['rating'].values
r_m = r_m - np.asarray([np.mean(r_m,1)]).T
u,s,v = np.linalg.svd(r_m)
checkdf = pd.read_csv("links_small.csv")

def collabreco(movie_id,n):
    rm = v.T[:,:50]
    dt = rm[movie_id-1,:]
    s = ""
    sim = np.dot(dt,rm.T)
    idxs = np.argsort(-sim)
    s+="some movies that are liked by fans of this movie are\n"
    i =0
    for idx in idxs[:min(n,len(idxs))]:
        
        
        try:
            chidx = ide[idx]
            dfidx = checkdf[checkdf.movieId==chidx].tmdbId.values[0]
            s+=str(i+1)
            s+=" "
            s+=df1[df1.movie_id==dfidx].title.values[0]
            s+="\n"
            
            i =i +1
        except Exception as e:
            pass

    return s+"\n"

def recommend(movie_name,n):
    s = ""
    try:
        s = overallreco(movie_name,n)
    except Exception as e:
        s = "not much data about the movie"
    try:
        id1 = keys[movie_name.lower().replace(' ','')]
        id2 = df1.iloc[id1]['movie_id']
        id3 = checkdf[checkdf.tmdbId==id2].movieId.values[0]
        id4 = di[id3]
        s+="\n"+collabreco(id4,n)
    except Exception as e:
        pass
    return s+"\n"

##code taken from surprise documentation for performance measures

"""from surprise import Dataset, SVD, Reader
from surprise.model_selection import cross_validate

reader = Reader()
data = Dataset.load_from_df(colldf[['userId', 'movieId', 'rating']], reader)

algo = SVD()

cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
import pickle
with open('x.pkl', 'rb') as f:
    data = pickle.load(f)"""





