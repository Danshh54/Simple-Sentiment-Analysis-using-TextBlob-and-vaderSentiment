import streamlit as st
from transformers import pipeline
import pandas as pd
import altair as alt


# @st.cache_resource untuk menyimpan cache
@st.cache_resource 
def load_model():    
    model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"    
    sentiment_model = pipeline(        
        "sentiment-analysis",        
        model=model_name,        
        tokenizer=model_name,
        # top_k=None menampilkan probabilitas dari ke-3 label sekaligus.
        top_k=None
    )    
    return sentiment_model


# Mengubah teks ke indo karna model bawaan pake teks inggris
def translate_label(label):    
    label = label.upper()    
    if "POSITIVE" in label:        
        return "Positif"    
    elif "NEGATIVE" in label:        
        return "Negatif"    
    elif "NEUTRAL" in label:        
        return "Netral"    
    else:        
        return label


# Fungsi yang memproses seluruh teks kalimat/paragraf)
def analyze_sentiment(text, sentiment_model):    
    results = sentiment_model(text)[0] 
    
    # menentukan sentimen utama dr yang dominan
    best_result = max(results, key=lambda x: x['score'])
    main_label = translate_label(best_result['label'])
    
    # Mengumpulkan ketiga skor ke sebuah list bisa tampil sbg grafik altar
    all_scores = []
    for res in results:
        all_scores.append({
            'sentiment': translate_label(res['label']),
            'score': res['score']
        })
        
    return main_label, all_scores

# Mengubah list data sentimen jadi DataFrame & nama kolomnya agar dikenali Altair
def convert_to_df(all_scores):    
    df = pd.DataFrame(all_scores)
    df.rename(columns={'sentiment': 'metric', 'score': 'value'}, inplace=True)
    return df

# Memecah kalimat per kata lalu menebak sentimen masing" kata
def analyze_token_sentiment(text, sentiment_model):
    pos_list = []
    neg_list = []
    neu_list = []
    
    # Memecah teks berdasarkan spasi (split)
    for word in text.split():
        # Membuang tanda baca
        clean_word = word.strip(".,!?;:\"'()[]{}")
        if clean_word == "":
            continue
            
        try:
            # Menganalisis 1 kata tunggal
            res = sentiment_model(clean_word)[0]
            best_res = max(res, key=lambda x: x['score'])
            label = translate_label(best_res['label'])
            
            # Membulatkan nilai jadi 3 angka di belakang koma
            score = round(best_res['score'], 3)
            
            # Memasukkan kata & skor ke list
            if label == "Positif":
                pos_list.append(clean_word)
                pos_list.append(score)
            elif label == "Negatif":
                neg_list.append(clean_word)
                neg_list.append(score)
            else:
                neu_list.append(clean_word)
                
        except Exception:
            # Jika ada error, anggap sebagai netral
            neu_list.append(clean_word)
            
    return {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}


# tampilan webnya
def main():
    st.title("Aplikasi Analisis Sentimen Bahasa Indonesia")
    st.subheader("Streamlit Project")
    
    # Memanggil model
    sentiment_model = load_model()

    menu = ["Home", "Tentang"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        
        with st.form("nlpForm"):
            raw_text = st.text_area("Masukkan Teks Di Sini")
            submit_button = st.form_submit_button(label='Analisis')

        if submit_button and raw_text:
            # layar jadi kiri kanan
            col1, col2 = st.columns(2)
            
            main_label, all_scores = analyze_sentiment(raw_text, sentiment_model)
            
            with col1:
                st.info("Hasil")
                
                if main_label == "Positif":
                    st.markdown("Sentimen:: Positif :smiley:")
                elif main_label == "Negatif":
                    st.markdown("Sentimen:: Negatif :angry:")
                else:
                    st.markdown("Sentimen:: Netral 😐")
                    
                result_df = convert_to_df(all_scores)
                st.dataframe(result_df, use_container_width=True)
                
                chart = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric'
                )
                st.altair_chart(chart, use_container_width=True)
                
            with col2:
                st.info("Sentimen Per Kata (Token)")
                
                # Mengeksekusi fungsi pemecah kata dan menampilkannya sebagai Dictionary
                token_sentiments = analyze_token_sentiment(raw_text, sentiment_model)
                st.write(token_sentiments)
                
    else:
        st.subheader("Tentang")
        st.write("Aplikasi pemrosesan bahasa alami (NLP) untuk analisis sentimen teks berbahasa Indonesia.")

if __name__ == '__main__':
    main()