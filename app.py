import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av # Nueva librería necesaria para recv
#api_key = st.secrets.get("ROBOFLOW_API_KEY", "TU_API_KEY") # Opcional si usas secretos
from ultralytics import YOLO
import numpy as np
from PIL import Image

# 1. Configuración de pantalla y PWA
st.set_page_config(page_title="EcoDetect", page_icon="♻️", layout="wide")

# Inyectar PWA (Meta tags, Manifest y Service Worker)
st.markdown("""
    <!-- Meta tags para PWA -->
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="EcoDetect">
    <link rel="apple-touch-icon" href="/static/logo.png">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="theme-color" content="#2e7d32">
    <meta name="description" content="IA para reconocimiento de residuos sólidos en Abancay, Perú.">
    
    <!-- Manifest -->
    <link rel="manifest" href="/manifest.json">
    
    <!-- Service Worker -->
    <script>
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', () => {
                navigator.serviceWorker.register('/static/sw.js')
                    .then(reg => console.log('SW registered:', reg.scope))
                    .catch(err => console.log('SW registration failed:', err));
            });
        }
    </script>
    
    <style>
        /* Ocultar toolbar de Streamlit en móvil */
        .stToolbar { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# 2. Inicializar memoria (Session State) para el conteo
if "counts" not in st.session_state:
    st.session_state.counts = {"Plástico": 0, "Papel": 0, "Orgánico": 0, "Vidrio": 0}

# 3. Cargar el modelo
@st.cache_resource
def get_model():
    return YOLO("best.pt")

model = get_model()

# --- INTERFAZ DE USUARIO ---
st.title("♻️ Sistema de Reconocimiento de Residuos")
st.write("Reconoce en tiempo real o sube una fotografía para analizar.")

# Creamos dos pestañas: Cámara y Carga de Archivo
tab1, tab2 = st.tabs(["🎥 Cámara en Vivo", "📁 Subir Imagen"])

# --- TAB 1: CÁMARA EN VIVO ---
with tab1:
    col_v1, col_v2 = st.columns([2, 1])
    
    with col_v1:
        # Función moderna 'recv' (reemplaza a transform)
        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Inferencia rápida (imgsz=320 para baja latencia)
            results = model.predict(img, conf=0.45, imgsz=640, verbose=False)
            
            local_counts = {"Plástico": 0, "Papel": 0, "Orgánico": 0, "Vidrio": 0}
            for box in results[0].boxes:
                label = model.names[int(box.cls)].lower()
                if "plastico" in label: local_counts["Plástico"] += 1
                elif "papel" in label: local_counts["Papel"] += 1
                elif "organico" in label: local_counts["Orgánico"] += 1
                elif "vidrio" in label: local_counts["Vidrio"] += 1
            
            # Guardamos el conteo en session_state para que la web lo vea
            st.session_state.counts = local_counts
            
            # Dibujamos y devolvemos el frame procesado
            annotated_img = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

        webrtc_streamer(
            key="waste-live",
            video_frame_callback=video_frame_callback,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_v2:
        st.subheader("Conteo en Vivo")
        if st.button("🔄 Actualizar Reporte"):
            st.bar_chart(st.session_state.counts)
            st.table(st.session_state.counts)

# --- TAB 2: SUBIR IMAGEN ---
with tab2:
    st.subheader("Analizar Fotografía")
    uploaded_file = st.file_uploader("Elige una imagen de tus archivos...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convertir archivo a imagen de OpenCV
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Procesar con YOLO
        st.write("🔍 Analizando...")
        results_upload = model.predict(img_array, conf=0.40)
        
        # Mostrar resultado
        res_plotted = results_upload[0].plot()
        st.image(res_plotted, caption="Resultado de la reconocimiento", width="stretch")# use_container_width=True
        
        # Conteo de la imagen subida
        upload_counts = {"Plástico": 0, "Papel": 0, "Orgánico": 0, "Vidrio": 0}
        for box in results_upload[0].boxes:
            label = model.names[int(box.cls)].lower()
            if "plastico" in label: upload_counts["Plástico"] += 1
            elif "papel" in label: upload_counts["Papel"] += 1
            elif "organico" in label: upload_counts["Orgánico"] += 1
            elif "vidrio" in label: upload_counts["Vidrio"] += 1
            
        st.write("### Objetos encontrados:")
        st.json(upload_counts)

# Barra lateral informativa
st.sidebar.image("static/logo.png", width=120)
st.sidebar.markdown("---")
st.sidebar.markdown("**Proyecto:** IA de Reconocimiento de Residuos\n**Sede:** Abancay")