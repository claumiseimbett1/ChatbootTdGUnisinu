# unisinu_thesis_chatbot.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import pickle
import json
import hashlib
from datetime import datetime
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

@st.cache_resource
def load_llama_model():
    """Carga el modelo con optimizaciones"""
    try:
        # Usar modelo más compatible y liviano
        model_name = "microsoft/DialoGPT-small"  # Más liviano y estable
        
        print(f"📥 Descargando modelo: {model_name}")
        
        # Cargar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configurar argumentos del modelo
        model_kwargs = {
            "torch_dtype": torch.float32,
        }
        
        # Cargar modelo
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        print("✅ Modelo cargado exitosamente")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error cargando el modelo: {str(e)}")
        st.error(f"Error cargando el modelo: {str(e)}")
        return None, None

@st.cache_resource
def setup_rag_system(pdf_folder="pdfs"):
    """Configura el sistema RAG con los PDFs de trabajo de grado"""
    try:
        # Embeddings en español
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Verificar si existe vectorstore guardado
        vectorstore_path = "unisinu_vectorstore"
        if os.path.exists(f"{vectorstore_path}.faiss"):
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            return vectorstore, embeddings
        
        # Si no existe, crear nuevo vectorstore
        all_documents = []
        if os.path.exists(pdf_folder):
            for pdf_file in os.listdir(pdf_folder):
                if pdf_file.endswith('.pdf'):
                    try:
                        pdf_path = os.path.join(pdf_folder, pdf_file)
                        loader = PyPDFLoader(pdf_path)
                        documents = loader.load()
                        
                        for doc in documents:
                            doc.metadata.update({
                                "source": pdf_file,
                                "doc_type": identify_doc_type(pdf_file)
                            })
                        
                        all_documents.extend(documents)
                    except Exception as e:
                        st.warning(f"Error cargando {pdf_file}: {str(e)}")
                        continue
        
        if all_documents:
            # Dividir documentos en chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            texts = text_splitter.split_documents(all_documents)
            vectorstore = FAISS.from_documents(texts, embeddings)
            
            # Guardar para uso futuro
            vectorstore.save_local(vectorstore_path)
        else:
            vectorstore = None
        
        return vectorstore, embeddings
    except Exception as e:
        st.error(f"Error configurando sistema RAG: {str(e)}")
        return None, None

def identify_doc_type(filename):
    """Identifica el tipo de documento"""
    filename_lower = filename.lower()
    if "reglamento" in filename_lower or "opcion" in filename_lower:
        return "reglamento_grado"
    elif "practica" in filename_lower:
        return "practica_profesional"
    elif "procedimiento" in filename_lower:
        return "procedimiento"
    else:
        return "general"

class RedisCache:
    """Maneja el cache de respuestas con Redis"""
    def __init__(self):
        self.redis_client = None
        self.cache_available = False
        self._connect()
    
    def _connect(self):
        """Conecta a Redis con fallback"""
        if not REDIS_AVAILABLE:
            print("⚠️ Módulo redis no instalado - cache deshabilitado")
            self.cache_available = False
            return
            
        try:
            self.redis_client = redis.Redis(
                host='localhost', 
                port=6379, 
                db=0, 
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            # Probar conexión
            self.redis_client.ping()
            self.cache_available = True
            print("✅ Redis conectado exitosamente")
        except Exception as e:
            print(f"⚠️ Redis no disponible: {e}")
            self.cache_available = False
    
    def _generate_key(self, user_input):
        """Genera clave única para el cache"""
        normalized_input = user_input.lower().strip()
        return f"thesis_chatbot_response:{hashlib.md5(normalized_input.encode()).hexdigest()}"
    
    def get_response(self, user_input):
        """Obtiene respuesta del cache"""
        if not self.cache_available:
            return None
        
        try:
            key = self._generate_key(user_input)
            cached_data = self.redis_client.get(key)
            if cached_data:
                response_data = json.loads(cached_data)
                print(f"🔄 Cache HIT para: {user_input[:50]}...")
                return response_data['response']
        except Exception as e:
            print(f"Error obteniendo del cache: {e}")
        
        return None
    
    def set_response(self, user_input, response, ttl=3600):
        """Guarda respuesta en cache"""
        if not self.cache_available:
            return False
        
        try:
            key = self._generate_key(user_input)
            response_data = {
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'input': user_input
            }
            self.redis_client.setex(key, ttl, json.dumps(response_data))
            print(f"💾 Cache SAVE para: {user_input[:50]}...")
            return True
        except Exception as e:
            print(f"Error guardando en cache: {e}")
            return False

class UnusinuThesisBot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.vectorstore = None
        self.embeddings = None
        self.conversation_history = []
        self.cache = RedisCache()
    
    def search_documents(self, query, k=2):
        """Busca en los documentos PDF"""
        try:
            if not self.vectorstore:
                print("⚠️ No hay vectorstore disponible")
                return ""
            
            print(f"🔍 Buscando: {query}")
            docs = self.vectorstore.similarity_search(query, k=k)
            context = ""
            for doc in docs:
                context += f"\n[{doc.metadata.get('doc_type', 'documento')}]: {doc.page_content}\n"
            
            print(f"📄 Documentos encontrados: {len(docs)}")
            return context
        except Exception as e:
            print(f"❌ Error en búsqueda de documentos: {e}")
            return ""
    
    def get_fallback_response(self, user_input):
        """Respuesta de emergencia usando la información hardcodeada"""
        user_lower = user_input.lower()
        
        # Modalidades de trabajo de grado
        if any(word in user_lower for word in ["modalidad", "modalidades", "tipos", "opciones", "trabajo de grado", "opcion de grado"]):
            return """📋 **MODALIDADES DE TRABAJO DE GRADO - UNISINU:**

**1. 🔬 TRABAJOS INVESTIGATIVOS:**
• Participación activa en un Grupo de Investigación
• Presentación y desarrollo de un Proyecto de Investigación

**2. 🏢 PRÁCTICAS DE EXTENSIÓN:**
• Práctica con Proyección Empresarial o Social
• Internado Rotatorio en Medicina
• Judicatura (Programa de Derecho)

**3. 🎓 CO-TERMINAL:**
• Cursar asignaturas de primer semestre de Posgrado

**4. 📚 CURSOS DE PERFECCIONAMIENTO:**
• Diplomados especializados

📝 **Nota mínima:** 3.5 (excepto trabajos investigativos: 3.0 y diplomados: 3.8)
⏰ **Duración:** Mínimo 4 meses, máximo 12 meses

📞 **Contacto Universidad del Sinú:**
📧 Email: admisiones@unisinu.edu.co
📍 Montería, Córdoba"""
        
        # Requisitos generales
        elif any(word in user_lower for word in ["requisitos", "requerimientos", "necesito", "debo cumplir"]):
            return """📋 **REQUISITOS PARA TRABAJO DE GRADO - UNISINU:**

**✅ REQUISITOS GENERALES:**
• Haber terminado todas las materias del pensum académico
• Presentar propuesta al Comité de Trabajos de Grado
• Tener director de trabajo de grado asignado
• Cumplir con los pre-requisitos del programa

**📝 DOCUMENTACIÓN:**
• Formulario de inscripción diligenciado
• Anteproyecto o propuesta según modalidad
• Certificados médicos (si aplica)
• Documentos específicos según modalidad

**⏰ PLAZOS:**
• Inscripción: Primeros 15 días del ciclo lectivo
• Desarrollo: 4-12 meses desde aprobación
• Sustentación: Según cronograma del comité

📞 **Más información:**
📧 Email: admisiones@unisinu.edu.co
📍 Universidad del Sinú - Montería"""
        
        # Proceso de inscripción
        elif any(word in user_lower for word in ["inscripcion", "inscripción", "como inscribir", "proceso", "pasos"]):
            return """📝 **PROCESO DE INSCRIPCIÓN TRABAJO DE GRADO - UNISINU:**

**🔹 PASO 1: PREPARACIÓN**
• Definir modalidad de trabajo de grado
• Identificar director/tutor
• Preparar anteproyecto

**🔹 PASO 2: INSCRIPCIÓN**
• Llenar formulario de inscripción
• Presentar al Comité de Trabajos de Grado
• Plazo: Primeros 15 días del ciclo lectivo

**🔹 PASO 3: EVALUACIÓN**
• El comité evalúa la propuesta (máximo 20 días hábiles)
• Asignación de director/supervisor
• Aprobación o solicitud de modificaciones

**🔹 PASO 4: DESARROLLO**
• Seguimiento con director asignado
• Entregas periódicas según cronograma
• Evaluación continua

**🔹 PASO 5: FINALIZACIÓN**
• Entrega de informe final
• Sustentación pública
• Calificación final

📞 **Contacto:**
📧 Email: admisiones@unisinu.edu.co"""
        
        # Duración y plazos
        elif any(word in user_lower for word in ["duracion", "duración", "tiempo", "plazo", "cuanto tiempo"]):
            return """⏰ **DURACIÓN Y PLAZOS - TRABAJO DE GRADO UNISINU:**

**📅 DURACIÓN GENERAL:**
• **Mínimo:** 4 meses
• **Máximo:** 12 meses
• **Conteo:** Desde fecha de aprobación del proyecto

**🔍 POR MODALIDAD:**
• **Trabajos Investigativos:** 4-12 meses
• **Prácticas Empresariales:** 4-6 meses (tiempo completo) o 9-12 meses (medio tiempo)
• **Judicatura:** 9 meses (ad-honorem) o 1 año (remunerada)
• **Co-terminal:** 1 semestre

**⚠️ PLAZOS IMPORTANTES:**
• Inscripción: Primeros 15 días del ciclo
• Evaluación propuesta: Máximo 20 días hábiles
• Vigencia propuesta: 1 año
• Asesorías: 16-32 horas por semestre

**🔄 PRÓRROGA:**
• Máximo 2 meses adicionales
• Requiere 75% de avance demostrado
• Aprobación del Comité de Trabajos de Grado

📞 **Consultas:**
📧 Email: admisiones@unisinu.edu.co"""
        
        # Calificaciones
        elif any(word in user_lower for word in ["calificacion", "calificación", "nota", "notas", "minima", "mínima", "aprobar"]):
            return """📊 **CALIFICACIONES TRABAJO DE GRADO - UNISINU:**

**🎯 NOTAS MÍNIMAS POR MODALIDAD:**
• **Trabajos Investigativos:** 3.0/5.0
• **Prácticas de Extensión:** 3.5/5.0
• **Co-terminal:** 3.5/5.0
• **Diplomados:** 3.8/5.0

**🏆 ESCALA DE EVALUACIÓN:**
• **Laureado:** 5.0
• **Meritorio:** 4.5 - 4.9
• **Sobresaliente:** 4.0 - 4.4
• **Satisfactorio:** 3.5 - 3.9
• **Suficiente:** 3.0 - 3.4
• **Insuficiente:** < 3.0

**⚠️ IMPORTANTES:**
• Si se reprueba, debe cambiar de modalidad
• Reconocimientos Meritorio/Laureado requieren aprobación del Consejo Académico
• La calificación incluye: documento escrito + sustentación + relación entre ambos

📞 **Más información:**
📧 Email: admisiones@unisinu.edu.co
📍 Universidad del Sinú - Montería"""
        
        # Comité de trabajos de grado
        elif any(word in user_lower for word in ["comite", "comité", "quien evalua", "quién evalúa", "evaluacion", "evaluación"]):
            return """👥 **COMITÉ DE TRABAJOS DE GRADO - UNISINU:**

**🔹 COMPOSICIÓN:**
• Decano de la Facultad (presidente)
• Jefe de programa
• Coordinador de Práctica de la Facultad
• Coordinador de Investigaciones de la Facultad
• Jefes de área del programa

**🔹 FUNCIONES PRINCIPALES:**
• Evaluar propuestas de trabajo de grado
• Designar directores y supervisores
• Asignar jurados calificadores
• Definir fechas de sustentación
• Resolver temas del reglamento

**🔹 PROCESO DE EVALUACIÓN:**
• Máximo 20 días hábiles para respuesta
• Reuniones cada 15 días máximo
• Seguimiento de cronogramas
• Supervisión de calidad académica

**📞 **Contacto:**
📧 Email: admisiones@unisinu.edu.co
📍 Universidad del Sinú - Montería"""
        
        # Prácticas profesionales
        elif any(word in user_lower for word in ["practica", "práctica", "pasantia", "pasantía", "empresa", "instituciones"]):
            return """🏢 **PRÁCTICAS PROFESIONALES - UNISINU:**

**📋 TIPOS DE PRÁCTICA:**
• Práctica Empresarial
• Práctica Social
• Judicatura (solo Derecho)

**⏰ DURACIÓN:**
• **Tiempo completo:** 4-6 meses
• **Medio tiempo:** 9-12 meses

**✅ REQUISITOS:**
• Haber aprobado todas las asignaturas
• Autorización de la Facultad
• Carta de aceptación de la empresa/institución
• Plan de trabajo avalado
• EPS vigente

**🏛️ SEDES VÁLIDAS:**
• Empresas privadas legalmente constituidas
• Instituciones públicas
• ONGs reconocidas
• No se acepta: empresas familiares hasta 4to grado

**📊 EVALUACIÓN:**
• Informe parcial (8va semana)
• Informe final
• Evaluación del supervisor externo
• Taller de socialización

📞 **Más información:**
📧 Email: admisiones@unisinu.edu.co"""
        
        # Director de trabajo de grado
        elif any(word in user_lower for word in ["director", "tutor", "asesor", "supervisor", "quien dirige", "quién dirige"]):
            return """👨‍🏫 **DIRECTOR DE TRABAJO DE GRADO - UNISINU:**

**🔹 SELECCIÓN:**
• Sugerido por el estudiante con orientación del consejero académico
• Si no hay sugerencia, el Comité asigna uno
• Debe tener expertise en el área del proyecto

**🔹 FUNCIONES:**
• Avalar la propuesta del estudiante
• Crear agenda de seguimiento
• Asesorar en la elaboración del proyecto
• Revisar contenido, metodología y presentación
• Asistir a reuniones del Comité
• Asistir a la sustentación final
• Informar incumplimientos del estudiante

**🔹 ASESORÍAS:**
• 16-32 horas por semestre
• Hasta por 2 períodos académicos (1 año)
• Seguimiento continuo del cronograma

**⚠️ CAMBIO DE DIRECTOR:**
• El estudiante puede solicitar cambio justificado
• Requiere aprobación del Comité de Trabajos de Grado

📞 **Consultas:**
📧 Email: admisiones@unisinu.edu.co"""
        
        return None

    def generate_response(self, user_input):
        """Genera respuesta usando fallback primero, luego PDFs si es necesario"""
        print(f"🔍 INPUT: {user_input}")
        
        # Intentar obtener respuesta del cache primero
        cached_response = self.cache.get_response(user_input)
        if cached_response:
            print("📋 Usando respuesta del cache")
            return cached_response
        
        # Primero intentar respuesta de fallback
        fallback = self.get_fallback_response(user_input)
        if fallback:
            print("✅ Respuesta hardcodeada encontrada")
            self.cache.set_response(user_input, fallback, ttl=7200)
            return fallback
        
        # Si no hay respuesta hardcodeada, buscar en PDFs
        print("📚 No hay respuesta hardcodeada, buscando en PDFs...")
        document_context = self.search_documents(user_input)
        print(f"📄 Contexto encontrado: {len(document_context) if document_context else 0} caracteres")
        
        if document_context and len(document_context.strip()) > 50:
            pdf_response = f"""📋 **Información encontrada en documentos oficiales UNISINU:**

{document_context}

📞 **Para más información:**
📧 Email: admisiones@unisinu.edu.co
📍 Universidad del Sinú - Elías Bechara Zainúm
🏛️ Montería, Córdoba, Colombia

💡 **Consulta también:** Oficina de Admisiones y Registro Académico"""
            
            print("✅ Respuesta generada desde PDFs")
            self.cache.set_response(user_input, pdf_response, ttl=3600)
            return pdf_response
        
        # Si no encontramos nada en PDFs, dar respuesta genérica
        print("❌ No se encontró información específica")
        generic_response = f"""🎓 **Universidad del Sinú - Elías Bechara Zainúm**

Lo siento, no tengo información específica sobre tu consulta en este momento.

📞 **Para información detallada contacta:**
📧 Email: admisiones@unisinu.edu.co
🏛️ Oficina de Admisiones y Registro Académico
📍 Universidad del Sinú - Montería, Córdoba

**🔍 Consultas frecuentes:**
• Modalidades de trabajo de grado
• Requisitos y proceso de inscripción
• Duración y plazos
• Calificaciones y evaluación
• Prácticas profesionales

¡Estaremos felices de ayudarte! 📚"""
        
        self.cache.set_response(user_input, generic_response, ttl=1800)
        return generic_response

# Aplicación Streamlit
def main():
    st.set_page_config(
        page_title="Asistente Trabajo de Grado UNISINU",
        page_icon="🎓",
        layout="wide"
    )
    
    # CSS personalizado
    st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .header-container {
        background: linear-gradient(135deg, #dc3545 0%, #8b1538 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.2);
        text-align: center;
    }
    
    .logo-space {
        width: 80px;
        height: 80px;
        background-color: #ffffff;
        border-radius: 50%;
        margin: 0 auto 15px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 40px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
    }
    
    .main-title {
        color: #ffffff !important;
        font-size: 28px;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #dc3545 0%, #8b1538 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 12px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 3px 8px rgba(220, 53, 69, 0.3);
        width: 100%;
        margin-bottom: 8px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #c82333 0%, #721c24 100%);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(220, 53, 69, 0.4);
    }
    
    .frequent-queries {
        background: linear-gradient(rgba(255, 255, 255, 0.95), rgba(255, 255, 255, 0.95));
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        border: 3px solid #dc3545;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .section-title {
        color: #dc3545 !important;
        background-color: rgba(255, 255, 255, 0.95);
        font-size: 24px;
        font-weight: 900;
        margin-bottom: 15px;
        text-align: center;
        padding: 8px 16px;
        border-radius: 8px;
        display: inline-block;
        width: 100%;
        box-sizing: border-box;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header personalizado con logo
    col_logo, col_title = st.columns([1, 4])
    
    with col_logo:
        try:
            st.image("logo/LOGO_UNISINU.png", width=120)
        except:
            st.markdown("""
            <div class="logo-space">🎓</div>
            """, unsafe_allow_html=True)
    
    with col_title:
        st.markdown("""
        <div class="header-container">
            <h1 class="main-title">Asistente Virtual - Trabajo de Grado UNISINU</h1>
            <p style="color: #ffffff; margin: 0;">Universidad del Sinú - Elías Bechara Zainúm</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Mensaje de bienvenida
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                padding: 20px; 
                border-radius: 15px; 
                margin: 20px auto; 
                border-left: 5px solid #dc3545;
                box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
                max-width: 800px;
                text-align: center;">
        <h3 style="color: #dc3545; margin-bottom: 15px; font-weight: bold;">¡Bienvenido al Asistente de Trabajo de Grado!</h3>
        <p style="color: #495057; font-size: 16px; margin: 0; line-height: 1.5;">
            Estoy aquí para ayudarte con todas tus consultas sobre <strong>trabajos de grado, modalidades, requisitos, 
            prácticas profesionales y procedimientos</strong> de la Universidad del Sinú. 
            <strong>¿En qué puedo asistirte hoy?</strong> 📚
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar bot
    if "bot" not in st.session_state:
        with st.spinner("Inicializando sistema de trabajo de grado..."):
            st.session_state.bot = UnusinuThesisBot()
            st.session_state.bot.vectorstore, st.session_state.bot.embeddings = setup_rag_system()
        st.success("✅ Sistema listo para consultas!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Botones de consultas frecuentes
    st.markdown("""
    <div class="frequent-queries">
        <div class="section-title">Consultas Frecuentes</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Modalidades de Trabajo de Grado"):
            user_input = "¿Cuáles son las modalidades de trabajo de grado?"
            process_message(user_input)
            
        if st.button("⏰ Duración y Plazos"):
            user_input = "¿Cuánto tiempo dura un trabajo de grado?"
            process_message(user_input)
            
        if st.button("👨‍🏫 Director de Trabajo de Grado"):
            user_input = "¿Cómo se asigna el director de trabajo de grado?"
            process_message(user_input)
    
    with col2:
        if st.button("✅ Requisitos Generales"):
            user_input = "¿Qué requisitos debo cumplir para el trabajo de grado?"
            process_message(user_input)
            
        if st.button("📊 Calificaciones y Notas"):
            user_input = "¿Cuáles son las notas mínimas para aprobar?"
            process_message(user_input)
            
        if st.button("🏢 Prácticas Profesionales"):
            user_input = "¿Cómo funcionan las prácticas profesionales?"
            process_message(user_input)
        
    with col3:
        if st.button("📝 Proceso de Inscripción"):
            user_input = "¿Cómo me inscribo para el trabajo de grado?"
            process_message(user_input)
        
        if st.button("👥 Comité de Trabajos de Grado"):
            user_input = "¿Quién evalúa mi trabajo de grado?"
            process_message(user_input)
            
        if st.button("📞 Información de Contacto"):
            user_input = "¿Dónde puedo obtener más información?"
            process_message(user_input)
    
    # Mostrar historial de chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input del usuario
    if prompt := st.chat_input("Escribe tu pregunta sobre trabajo de grado aquí..."):
        process_message(prompt)

def process_message(user_input):
    """Procesa un mensaje del usuario"""
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generar respuesta
    with st.spinner("Consultando reglamentos y documentos..."):
        response = st.session_state.bot.generate_response(user_input)
    
    # Agregar respuesta al historial
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun para mostrar los nuevos mensajes
    st.rerun()

# Información adicional en el sidebar
def show_sidebar_info():
    """Muestra información adicional en el sidebar"""
    st.sidebar.markdown("""
    ## 🎓 Universidad del Sinú
    ### Elías Bechara Zainúm
    
    ---
    
    ### 📋 Modalidades Disponibles:
    
    **🔬 Trabajos Investigativos**
    - Participación en Grupo de Investigación
    - Proyecto de Investigación Individual
    
    **🏢 Prácticas de Extensión**
    - Práctica Empresarial
    - Práctica Social
    - Judicatura (Derecho)
    
    **🎓 Co-terminal**
    - Asignaturas de Posgrado
    
    **📚 Diplomados**
    - Cursos de Perfeccionamiento
    
    ---
    
    ### ⏰ Datos Importantes:
    - **Duración:** 4-12 meses
    - **Nota mínima:** 3.0-3.8 (según modalidad)
    - **Inscripción:** Primeros 15 días del ciclo
    
    ---
    
    ### 📞 Contacto:
    **Email:** admisiones@unisinu.edu.co  
    **Ubicación:** Montería, Córdoba  
    **Web:** www.unisinu.edu.co
    
    ---
    
    ### 📄 Documentos Cargados:
    - Reglamento de Opción de Grado
    - Procedimiento Trabajo de Grado
    - Reglamento Prácticas Profesionales
    """)

if __name__ == "__main__":
    # Mostrar información en sidebar
    show_sidebar_info()
    
    # Ejecutar aplicación principal
    main()
    
    # Footer
    st.markdown("""
    ---
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #dc3545 0%, #8b1538 100%); 
                border-radius: 10px; margin-top: 30px;">
        <p style="color: white; margin: 0; font-weight: bold;">
            🎓 Universidad del Sinú - Elías Bechara Zainúm<br>
            📍 Montería, Córdoba, Colombia<br>
            📧 admisiones@unisinu.edu.co
        </p>
    </div>
    """, unsafe_allow_html=True)