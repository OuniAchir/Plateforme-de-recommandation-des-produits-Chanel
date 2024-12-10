import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from sentence_transformers import SentenceTransformer
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Charger ResNet50
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Charger les DataFrames
df = pd.read_parquet("hf://datasets/DBQ/Chanel.Product.prices.Germany/data/train-00000-of-00001-d681c47b79d4401f.parquet")
df1 = pd.read_parquet('df_with_all_embeddings1.parquet')
df2 = pd.read_parquet('textual_embeddings_sentence_bert.parquet')
df3 = pd.read_parquet('df_with_embeddings2.parquet')

# Ajouter les embeddings au DataFrame principal
df['text_embeddings'] = df2['embeddings_sentence_bert']
df['image_embeddings'] = df3['image_embedding2']

def ensure_numpy_array(column):
    return column.apply(lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x))

df['text_embeddings'] = ensure_numpy_array(df['text_embeddings'])
df['image_embeddings'] = ensure_numpy_array(df['image_embeddings'])

default_image_vector = np.zeros(51200)

df['image_embeddings'] = df['image_embeddings'].apply(
    lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) and len(x) == 51200 else default_image_vector
)

# Convertir en tableaux NumPy
text_embeddings_array = np.array(df['text_embeddings'].tolist())
image_embeddings_array = np.array(df['image_embeddings'].tolist())

# Fonction pour extraire les embeddings d'une image
def extract_image_embedding(image):
    image = image.resize((150, 150))  # Redimensionner pour ResNet50
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter une dimension batch
    image_array = preprocess_input(image_array)  # Prétraitement spécifique à ResNet50
    embedding = resnet_model.predict(image_array)  # Extraire les features
    return embedding.flatten()

# Fonction pour obtenir les k éléments les plus similaires
def get_top_k_similar(embeddings, query_embedding, k=10):
    similarities = cosine_similarity([query_embedding], embeddings).flatten()
    top_indices = similarities.argsort()[-k:][::-1]
    return df.iloc[top_indices]

# Interface Streamlit
st.title("Système de Recommandation d'Articles")

option = st.selectbox(
    "Choisissez une option de recherche",
    ["Recherche par image", "Recherche par texte", "Recherche combinée"]
)

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

if option == "Recherche par image":
    uploaded_image = st.file_uploader("Chargez une image pour la recherche visuelle", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        # Afficher l'image chargée
        image = Image.open(uploaded_image)
        st.image(image, caption="Image chargée", use_column_width=True)

        with st.spinner("Extraction des embeddings pour l'image..."):
            query_image_embedding = extract_image_embedding(image)

        st.success("Embeddings extraits avec succès !")

        # Calculer les similarités et obtenir les recommandations
        results = get_top_k_similar(image_embeddings_array, query_image_embedding)
        st.write("Articles recommandés :")
        for _, row in results.iterrows():
            st.image(row['imageurl'], caption=row['title'])
            st.write(f"Description : {row['category2_code']}")
            st.write("---")


# Recherche par texte
elif option == "Recherche par texte":
    text_query = st.text_input("Saisissez une description pour la recherche textuelle")
    if text_query:
        # Générer l'embedding pour le texte
        st.write("Extraction des embeddings pour le texte...")
        query_text_embedding = model.encode(text_query)

        # Calculer les similarités et obtenir les recommandations
        results = get_top_k_similar(text_embeddings_array, query_text_embedding)
        st.write("Articles recommandés :")
        for _, row in results.iterrows():
            st.image(row['imageurl'], caption=row['title'])
            st.write(f"Description : {row['category2_code']}")
            st.write("---")

# Recherche combinée
elif option == "Recherche combinée":
    uploaded_image = st.file_uploader("Chargez une image pour la recherche visuelle", type=["jpg", "png", "jpeg"])
    text_query = st.text_input("Saisissez une description pour la recherche textuelle")

    if uploaded_image and text_query:
        # Afficher l'image chargée
        image = Image.open(uploaded_image)
        st.image(image, caption="Image chargée", use_column_width=True)

        # Extraire l'embedding de l'image
        st.write("Extraction des embeddings pour l'image...")
        query_image_embedding = extract_image_embedding(image)

        # Générer l'embedding pour le texte
        st.write("Extraction des embeddings pour le texte...")
        query_text_embedding = model.encode(text_query)

        # Calculer les similarités visuelles
        st.write("Calcul des similarités visuelles...")
        image_similarities = cosine_similarity([query_image_embedding], image_embeddings_array).flatten()

        # Calculer les similarités textuelles
        st.write("Calcul des similarités textuelles...")
        text_similarities = cosine_similarity([query_text_embedding], text_embeddings_array).flatten()

        # Combiner les similarités
        alpha = 0.5  # Pondération entre image et texte
        combined_similarities = alpha * image_similarities + (1 - alpha) * text_similarities

        # Obtenir les indices des articles les plus similaires
        top_indices = combined_similarities.argsort()[-10:][::-1]

        # Afficher les résultats
        st.write("Articles recommandés :")
        for i in top_indices:
            st.image(df.iloc[i]['imageurl'], caption=df.iloc[i]['title'])
            st.write(f"Description : {df.iloc[i]['category2_code']}")
            st.write(f"Score combiné : {combined_similarities[i]:.4f}")
            st.write("---")