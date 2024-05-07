from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import sqlite3
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


movies = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv')
credits = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_credits.csv')

movies.head()

credits.head()

conn = sqlite3.connect('../data/movies_db.db')

movies.to_sql('movies_table',conn,if_exists='replace',index=False)
credits.to_sql('credits_table',conn,if_exists='replace',index=False)

query = conn.execute('select * from movies_table')
rows = query.fetchall()

consulta = """ 
            SELECT *
            FROM movies_table
            INNER JOIN credits_table
            ON movies_table.title = credits_table.title;"""

df = pd.read_sql_query(consulta,conn)
conn.close()

df = df.loc[:,~df.columns.duplicated()]
df

df['genres'] = df['genres'].apply(lambda x: [genre['name'] for genre in json.loads(x)] if pd.notna(x) else None)

df['keywords'] = df['keywords'].apply(lambda x: [kw['name'] for kw in json.loads(x)] if pd.notna(x) else None)
df['cast'] = df['cast'].apply(lambda x: [protagonista['name'] for protagonista in json.loads(x)][:3] if pd.notna(x) else None)
df['crew'] = df['crew'].apply(lambda x: " ".join([cm['name'] for cm in json.loads(x) if cm['job'] == 'Director']))
df['overview'] = df['overview'].apply(lambda x: [x])

df['overview'] = df['overview'].apply(lambda x: [str(x)])
df['genres'] = df['genres'].apply(lambda x: [str(genre) for genre in x])
df['keywords'] = df['keywords'].apply(lambda x: [str(kw) for kw in x])
df['cast'] = df['cast'].apply(lambda x: [str(protagonista) for protagonista in x])
df['crew'] = df['crew'].apply(lambda x: [str(director) for director in x])

df['tags'] = df['overview']+df['genres']+df['keywords']+df['cast']+df['crew']
df.tags

df['tags'] = df['tags'].apply(lambda x: ",".join(x).replace(","," "))
df.iloc[0].tags

vectorizer = TfidfVectorizer()
matriz_tfidf = vectorizer.fit_transform(df['tags'])

model = NearestNeighbors(n_neighbors=6,algorithm="brute",metric='cosine')
model.fit(matriz_tfidf)

def peliculas_similares(titulo_de_pelicula):
    indice = df[df['title'] == titulo_de_pelicula].index[0]
    distances, indices = model.kneighbors(matriz_tfidf[indice])
    peliculas_similares = [(df['title'][i]) for _,i in enumerate(indices[0])]
    return peliculas_similares[1:]

print(peliculas_similares('The Dark Knight Rises'))