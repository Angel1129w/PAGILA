import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine
import plotly.graph_objects as go
import plotly.io as pio
np.random.seed(0)
def connect_to_db():
    try:
        engine = create_engine('postgresql+psycopg2://postgres:070223@localhost:5432/pagila')
        print("Conexión exitosa")
        return engine
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None
def extract_data(engine):
    query = """
    SELECT a.actor_id, a.first_name, a.last_name, f.film_id, f.title, c.customer_id,
    c.first_name AS customer_first_name, c.last_name AS customer_last_name,
    r.rental_date, r.last_update
    FROM actor a
    JOIN film_actor fa ON a.actor_id = fa.actor_id
    JOIN film f ON fa.film_id = f.film_id
    JOIN inventory i ON f.film_id = i.film_id
    JOIN rental r ON i.inventory_id = r.inventory_id
    JOIN customer c ON r.customer_id = c.customer_id;
    """
    df = pd.read_sql(query, engine)
    print(f"Datos extraídos: {len(df)} registros")
    return df
def transform_data(df):
    df['first_name'] = df['first_name'].str.upper()
    df['last_name'] = df['last_name'].str.upper()
    df['customer_first_name'] = df['customer_first_name'].str.upper()
    df['customer_last_name'] = df['customer_last_name'].str.upper()
    df['rental_date'] = pd.to_datetime(df['rental_date']).dt.date
    df = df.drop_duplicates(subset=['customer_id', 'rental_date'])
    df_customers_films = df.groupby(['customer_id', 'customer_first_name', 'customer_last_name']).agg({
        'film_id': lambda x: len(x.unique()),
        'title': lambda x: ', '.join(x.unique()),
        'rental_date': lambda x: ', '.join(x.astype(str).unique())
    }).reset_index()
    df_actors_films = df.groupby(['actor_id', 'first_name', 'last_name']).agg({
        'film_id': lambda x: len(x.unique()),
        'title': lambda x: ', '.join(x.unique())
    }).reset_index()
    df_rentals = df.groupby(['customer_id', 'title']).agg({
        'rental_date': 'count'
    }).rename(columns={'rental_date': 'number_of_rentals'}).reset_index()
    print(f"Datos transformados: {len(df_customers_films)} clientes, {len(df_actors_films)} actores, {len(df_rentals)} alquileres.")
    return df_customers_films, df_actors_films, df_rentals
def save_to_database(df, engine, table_name):
    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Datos guardados en la tabla '{table_name}'.")
    except Exception as e:
        print(f"Error al guardar en la base de datos: {e}")
def create_plots(df_customers_films, df_actors_films, df_rentals):
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=df_customers_films['customer_first_name'] + ' ' + df_customers_films['customer_last_name'],
        y=df_customers_films['film_id'],
        name='Número de Películas'
    ))
    fig1.update_layout(title='Número de Películas por Cliente', xaxis_title='Cliente', yaxis_title='Número de Películas')
    pio.write_html(fig1, file='clientes_peliculas.html', auto_open=False)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df_actors_films['first_name'] + ' ' + df_actors_films['last_name'],
        y=df_actors_films['film_id'],
        name='Número de Películas'
    ))
    fig2.update_layout(title='Número de Películas por Actor', xaxis_title='Actor', yaxis_title='Número de Películas')
    pio.write_html(fig2, file='actores_peliculas.html', auto_open=False)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=df_rentals['title'],
        y=df_rentals['number_of_rentals'],
        name='Número de Alquileres'
    ))
    fig3.update_layout(title='Número de Alquileres por Película', xaxis_title='Película', yaxis_title='Número de Alquileres')
    pio.write_html(fig3, file='alquileres.html', auto_open=False)
def save_table_to_html(df, file_name):
    try:
        html = df.to_html(index=False)
        with open(file_name, 'w') as file:
            file.write(html)
        print(f"Tabla guardada en '{file_name}'.")
    except Exception as e:
        print(f"Error al guardar la tabla en HTML: {e}")
def preprocess_data(df_rentals):
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    df_rentals['user'] = user_encoder.fit_transform(df_rentals['customer_id'])
    df_rentals['movie'] = movie_encoder.fit_transform(df_rentals['title']) 
    return df_rentals, user_encoder, movie_encoder
def create_matrices(df_rentals, num_users, num_movies):
    user_movie_matrix = np.zeros((num_users, num_movies)) 
    for row in df_rentals.itertuples():
        user_movie_matrix[row.user, row.movie] = row.number_of_rentals    
    return user_movie_matrix
def build_model(num_users, num_movies, n_components=20):
    svd = TruncatedSVD(n_components=n_components)
    return svd
def train_model(svd, user_movie_matrix):
    svd.fit(user_movie_matrix)
    return svd
def recommend_movies(svd, user_movie_matrix, user_encoder, movie_encoder, user_id, top_n=20):
    try:
        user_idx = user_encoder.transform([user_id])[0]
    except ValueError:
        print(f"Error al transformar el user_id: {user_id}. Asegúrate de que el user_id sea válido y esté en el encoder.")
        return []
    movie_ids = list(movie_encoder.classes_)
    movie_indices = np.array(movie_encoder.transform(movie_ids))
    user_predictions = svd.transform(user_movie_matrix)[user_idx]
    top_indices = np.argsort(user_predictions)[-top_n:]
    top_movies = [movie_ids[i] for i in top_indices]
    return top_movies
def generate_recommendations_html(recommendations, file_name):
    try:
        html_content = "<h3>Recomendaciones de Películas</h3><ul>"
        for idx, title in enumerate(recommendations):
            html_content += f"<li><strong>{idx + 1}.</strong> {title}</li>"
        html_content += "</ul>"
        with open(file_name, 'w') as file:
            file.write(html_content)
        print(f"Recomendaciones guardadas en '{file_name}'.")
    except Exception as e:
        print(f"Error al guardar recomendaciones en HTML: {e}")
def main():
    engine = connect_to_db()
    if engine:
        df = extract_data(engine)
        df_customers_films, df_actors_films, df_rentals = transform_data(df)
        save_to_database(df_customers_films, engine, table_name='customers_films')
        save_to_database(df_actors_films, engine, table_name='actors_films')
        save_to_database(df_rentals, engine, table_name='rentals')
        create_plots(df_customers_films, df_actors_films, df_rentals)
        save_table_to_html(df_rentals, 'alquileres_tabla.html')
        save_table_to_html(df_actors_films, 'actores_tabla.html')
        save_table_to_html(df_customers_films, 'clientes_tabla.html')
        df_rentals, user_encoder, movie_encoder = preprocess_data(df_rentals)
        num_users = len(user_encoder.classes_)
        num_movies = len(movie_encoder.classes_)
        user_movie_matrix = create_matrices(df_rentals, num_users, num_movies)
        svd = build_model(num_users, num_movies)  
        svd = train_model(svd, user_movie_matrix)    
        user_id = 1 
        recommendations = recommend_movies(svd, user_movie_matrix, user_encoder, movie_encoder, user_id)        
        generate_recommendations_html(recommendations, 'recomendaciones.html')
        engine.dispose()
if __name__ == "__main__":
    main()
