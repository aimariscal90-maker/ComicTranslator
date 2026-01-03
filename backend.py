# ============================================================================
# IMPORTACIONES
# ============================================================================
# Librerías estándar de Python
import os  # Para trabajar con archivos y rutas del sistema
import io  # Para leer archivos como bytes
import math  # Para operaciones matemáticas (raíz cuadrada, etc.)
import textwrap  # Para dividir texto en líneas
import unicodedata
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
# Librerías de procesamiento de imágenes
import cv2  # OpenCV - para procesamiento de imágenes y visión por computadora
import numpy as np  # NumPy - para arrays y operaciones matemáticas con imágenes
from PIL import Image, ImageDraw, ImageFont  # Pillow - para dibujar texto en imágenes

# Librerías de APIs externas
from dotenv import load_dotenv  # Para cargar variables de entorno desde .env
from openai import OpenAI  # Cliente de OpenAI para traducción
from google.cloud import vision  # Google Cloud Vision API para detección de texto

# Tipos de Python para documentación
from typing import List, Tuple, Optional


# ============================================================================
# CLASE PRINCIPAL: ComicTranslator
# ============================================================================
class ComicTranslator:
    """
    Traductor de cómics que detecta texto en imágenes de cómics,
    lo traduce de inglés a español (España) y lo renderiza manteniendo
    el estilo visual original.
    
    Características principales:
    - Usa Google Cloud Vision para detectar bloques de texto completos
    - Usa OpenAI para traducir manteniendo mayúsculas/minúsculas originales
    - Detecta la forma orgánica de los globos de diálogo (ovalados)
    - Preserva la integridad de las palabras (no las parte)
    - Ajusta el tamaño de fuente dinámicamente para que quepa
    """
    
    def __init__(self, font_path: Optional[str] = None, debug_visuals: bool = False):
        """
        Inicializa el traductor de cómics.
        
        PASO 1: Cargar variables de entorno (.env)
        PASO 2: Configurar cliente de OpenAI
        PASO 3: Configurar cliente de Google Cloud Vision
        PASO 4: Buscar y cargar la fuente de cómic
        PASO 5: Configurar parámetros de detección
        PASO 6: Activar visualización de debug opcional (cajas coloreadas)
        
        Args:
            font_path: Ruta al archivo de fuente TTF (si es None, busca automáticamente)
            debug_visuals: Si es True, pinta overlays de colores para ver padding/wrapping
        """
        # Cargar variables de entorno desde el archivo .env
        load_dotenv()
        
        # ====================================================================
        # PASO 1: Configurar OpenAI para traducción
        # ====================================================================
        # Obtener la clave API de OpenAI desde las variables de entorno
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY no encontrada en el archivo .env")
        
        # Crear el cliente de OpenAI con la clave
        self.client = OpenAI(api_key=api_key)
        
        # Caché de traducciones para ahorrar llamadas repetidas
        self.translation_cache = {}
        
        # ====================================================================
        # PASO 2: Configurar Google Cloud Vision para detección de texto
        # ====================================================================
        # Ruta al archivo de credenciales de Google Cloud
        self.credentials_path = "service_account.json"
        
        # Verificar que el archivo de credenciales existe
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(
                "❌ CRÍTICO: 'service_account.json' no encontrado. "
                "Por favor descárgalo desde Google Cloud Console."
            )
        
        # Inicializar el cliente de Vision con las credenciales
        try:
            self.vision_client = vision.ImageAnnotatorClient.from_service_account_json(
                self.credentials_path
            )
        except Exception as e:
            raise RuntimeError(
                f"Error al inicializar Google Cloud Vision: {e}. "
                "Por favor verifica tu archivo service_account.json."
            )
        
        # ====================================================================
        # PASO 3: Buscar y cargar la fuente de cómic
        # ====================================================================
        if font_path is None:
            # Si no se especifica una ruta, buscar automáticamente
            possible_paths = ["comic_font.ttf", "comic_font.ttf.ttf"]
            self.font_path = None
            
            # Intentar encontrar la fuente en las rutas posibles
            for path in possible_paths:
                if os.path.exists(path):
                    self.font_path = path
                    break
            
            # Si no se encuentra, usar fuente por defecto del sistema
            if self.font_path is None:
                print("Advertencia: Archivo de fuente de cómic no encontrado. Usando fuente por defecto.")
                self.font_path = None
        else:
            # Si se especifica una ruta, verificar que existe
            if not os.path.exists(font_path):
                raise FileNotFoundError(f"Archivo de fuente no encontrado: {font_path}")
            self.font_path = font_path
        
        # ====================================================================
        # PASO 4: Configurar parámetros de detección
        # ====================================================================
        # Máximo: 70% del área de la página (solo filtra títulos de página completa)
        self.max_bubble_area_ratio = 0.70
        
        # Mínimo: 0.01% del área (permite texto muy pequeño como "Ouch", "Eh?")
        self.min_bubble_area_ratio = 0.0001
        
        # Debug: overlays de colores para inspección visual
        self.debug_visuals = debug_visuals
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Carga una imagen desde un archivo.
        
        Args:
            image_path: Ruta al archivo de imagen
            
        Returns:
            Imagen como array de NumPy en formato BGR (para OpenCV)
        """
        # Leer la imagen usando OpenCV
        image = cv2.imread(image_path)
        
        # Verificar que se cargó correctamente
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen desde {image_path}")
        
        return image
    
    def get_bbox_rect(self, bbox: List[List[int]]) -> Tuple[int, int, int, int]:
        """
        Convierte un bounding box (caja delimitadora) de formato de lista de puntos
        a formato de rectángulo (x_min, y_min, x_max, y_max).
        
        El bounding box viene como: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        Lo convertimos a: (x_min, y_min, x_max, y_max)
        
        Args:
            bbox: Coordenadas del bounding box como lista de puntos
            
        Returns:
            Tupla con (x_min, y_min, x_max, y_max)
        """
        # Extraer todas las coordenadas X e Y
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        # Encontrar los valores mínimos y máximos
        return int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))
    
    def detect_text_google(self, image_path: str) -> List[Tuple[List[List[int]], str, float]]:
        """
        Detecta texto usando Google Cloud Vision con:
        - Validación y redimensionado previo (límite 4096px / 20MB).
        - Reintentos con backoff exponencial.
        - Filtrado por confianza y tamaño mínimo razonable.
        - Validación de texto imprimible.
        """
        from PIL import Image as PILImage

        # ============================================================
        # Validación y posible redimensionado antes de enviar a Google
        # ============================================================
        try:
            with PILImage.open(image_path) as img:
                width, height = img.size
                file_size_mb = os.path.getsize(image_path) / (1024 * 1024)

                if width > 4096 or height > 4096 or file_size_mb > 20:
                    max_dim = max(width, height)
                    scale = 4096 / max_dim if max_dim > 0 else 1.0
                    new_w = int(width * scale)
                    new_h = int(height * scale)
                    img_resized = img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
                    temp_path = image_path + "_resized.jpg"
                    img_resized.save(temp_path, "JPEG", quality=85)
                    image_path = temp_path
                    logging.info(f"Imagen redimensionada: {width}x{height} -> {new_w}x{new_h}")

            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
        except Exception as e:
            raise IOError(f"Error al leer/preparar la imagen: {e}")

        # ============================================================
        # Reintentos con backoff exponencial
        # ============================================================
        max_retries = 3
        retry_delay = 1
        response = None

        for attempt in range(max_retries):
            try:
                image = vision.Image(content=content)
                response = self.vision_client.document_text_detection(image=image)
                if response.error.message:
                    raise Exception(response.error.message)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logging.warning(f"Error Vision (intento {attempt+1}): {e}. Reintentando en {wait_time}s")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Google Vision falló tras {max_retries} intentos: {e}")

        detections = []
        min_confidence = 0.3
        min_area = 20  # área mínima razonable

        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                block_text = ""

                # Reconstrucción del texto palabra a palabra para no perder separadores
                word_confidences = []
                for paragraph in block.paragraphs:
                    para_text = ""
                    for word in paragraph.words:
                        word_text = ''.join([symbol.text for symbol in word.symbols])
                        para_text += word_text + " "
                        if hasattr(word, 'confidence'):
                            word_confidences.append(word.confidence)
                    block_text += para_text

                block_text = block_text.strip()
                if not block_text:
                    continue

                # Confianza promedio de palabras o del bloque
                if word_confidences:
                    confidence = sum(word_confidences) / len(word_confidences)
                else:
                    confidence = block.confidence if hasattr(block, 'confidence') else 0.5

                # Filtrado por confianza: descartamos bloques poco fiables
                if confidence < min_confidence:
                    continue

                # Bounding box: lo guardamos para procesarlo después
                vertices = block.bounding_box.vertices
                bbox = [[v.x, v.y] for v in vertices]

                x_coords = [v.x for v in vertices]
                y_coords = [v.y for v in vertices]
                w = max(x_coords) - min(x_coords)
                h = max(y_coords) - min(y_coords)

                # Filtrado de tamaño mínimo (área)
                if w < 1 or h < 1 or (w * h) < min_area:
                    continue

                # Validar que tenga caracteres alfanuméricos
                if not any(c.isalnum() for c in block_text):
                    continue

                detections.append((bbox, block_text, confidence))

        # Limpiar archivo temporal si se creó
        if image_path.endswith("_resized.jpg") and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass

        return detections
    
    def get_contour_mask(self, image: np.ndarray, bbox: List[List[int]]) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
        """
        Versión mejorada:
        - Padding adaptativo según tamaño del globo.
        - Selección de contorno con sistema de puntuación (área, centro, circularidad, aspecto).
        - Dilatación adaptativa para cubrir texto sin invadir bordes.
        """
        x_min, y_min, x_max, y_max = self.get_bbox_rect(bbox)
        height, width = image.shape[:2]

        # Padding adaptativo: 2% del tamaño mayor del globo, mínimo 5px, máximo 30px
        box_w = x_max - x_min
        box_h = y_max - y_min
        box_size = max(box_w, box_h)
        padding = max(5, min(30, int(box_size * 0.02)))

        roi_x_min = max(0, x_min - padding)
        roi_y_min = max(0, y_min - padding)
        roi_x_max = min(width, x_max + padding)
        roi_y_max = min(height, y_max + padding)

        roi = image[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        if roi.size == 0:
            return None, (0, 0, 0, 0)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Blur fijo (podría hacerse adaptativo si se quiere)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Umbralización Otsu
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, (roi_x_min, roi_y_min, roi_x_max, roi_y_max)

        # Centro del texto relativo al ROI
        orig_cx = (x_min + x_max) // 2 - roi_x_min
        orig_cy = (y_min + y_max) // 2 - roi_y_min

        best_contour = None
        best_score = -1

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:
                continue

            score = 0

            # Factor área: prioriza contornos grandes (globos suelen ser amplios)
            score += area * 0.001

            # Factor centro: mejor si el texto queda dentro o muy cerca del contorno
            dist = cv2.pointPolygonTest(contour, (orig_cx, orig_cy), True)
            if dist >= 0:
                score += 1000
            elif dist > -20:
                score += 500 - (abs(dist) * 10)

            # Factor circularidad: globos suelen ser ovalados, evitamos formas raras
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                if 0.3 <= circularity <= 1.5:
                    score += 200

            # Factor proporción ancho/alto: penaliza contornos extremadamente alargados
            rect = cv2.boundingRect(contour)
            aspect_ratio = rect[2] / max(rect[3], 1)
            if 0.2 <= aspect_ratio <= 5.0:
                score += 100

            if score > best_score:
                best_score = score
                best_contour = contour

        if best_contour is None:
            best_contour = max(contours, key=cv2.contourArea)

        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [best_contour], -1, 255, -1)

        # Dilatación adaptativa: más grande para globos grandes, max 3 iteraciones
        dilation_iterations = max(1, min(3, int(box_size / 200)))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)

        return mask, (roi_x_min, roi_y_min, roi_x_max, roi_y_max)
    
    def get_bubble_color_from_mask(self, image: np.ndarray, mask: np.ndarray, 
                                   roi_coords: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
        """
        Detecta el color dominante del globo usando K-means (más robusto que la mediana).
        - Filtra píxeles oscuros (texto) antes de clústerizar.
        - Si hay pocos píxeles, cae a mediana.
        """
        x1, y1, x2, y2 = roi_coords
        roi = image[y1:y2, x1:x2]
        
        if roi.shape[:2] != mask.shape[:2]:
            return (255, 255, 255)
        
        masked_pixels = roi[mask == 255]
        if len(masked_pixels) < 10:
            return (255, 255, 255)
        
        # Filtrar texto (píxeles oscuros)
        gray_pixels = 0.299 * masked_pixels[:, 2] + 0.587 * masked_pixels[:, 1] + 0.114 * masked_pixels[:, 0]
        bright_mask = gray_pixels > 50  # Umbral de brillo
        background_pixels = masked_pixels[bright_mask] if bright_mask.any() else masked_pixels
        
        if len(background_pixels) < 10:
            background_pixels = masked_pixels
        
        # K-means para color dominante
        if len(background_pixels) >= 3:
            pixels_float = background_pixels.reshape(-1, 3).astype(np.float32)
            k = min(3, len(background_pixels))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            try:
                _, labels, centers = cv2.kmeans(pixels_float, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                unique, counts = np.unique(labels, return_counts=True)
                dominant_idx = unique[np.argmax(counts)]
                dominant_color = centers[dominant_idx]
                return (int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2]))
            except Exception:
                pass  # Si falla K-means, caemos a mediana
        
        avg_color = np.median(background_pixels, axis=0)
        return (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))
    
    def translate_text(self, text: str) -> str:
        """
        Traduce texto de inglés a español (España) usando OpenAI API.
        - Usa caché para evitar llamadas repetidas.
        - Maneja textos largos partiéndolos en chunks.
        - Reintenta ante errores temporales.
        """
        # Caché
        if text in self.translation_cache:
            return self.translation_cache[text]

        # Textos vacíos
        if not text or len(text.strip()) == 0:
            return text

        # Si es muy largo, dividir en chunks aproximados
        max_chunk_length = 400
        if len(text) > max_chunk_length:
            # Partimos por frases aproximando a 400 chars para que el modelo no recorte
            sentences = text.split('. ')
            chunks = []
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) < max_chunk_length:
                    current += sentence + ". "
                else:
                    if current:
                        chunks.append(current.strip())
                    current = sentence + ". "
            if current:
                chunks.append(current.strip())

            translated_chunks = [self._translate_chunk(chunk) for chunk in chunks]
            result = " ".join(translated_chunks)
            self.translation_cache[text] = result
            return result

        result = self._translate_chunk(text)
        self.translation_cache[text] = result
        return result

    def _translate_chunk(self, text: str) -> str:
        """Helper para traducir un chunk con retry y validación."""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Eres un traductor profesional de cómics EN->ES (España). "
                                "Reglas CRÍTICAS:\n"
                                "1. Mantén el significado exacto y el tono.\n"
                                "2. NO traduzcas Nombres Propios de personajes (ej: 'Immortus', 'Death's Head', 'Stark', 'Maker'). Déjalos tal cual.\n"
                                "3. Si el texto parece cortado o incompleto, tradúcelo lo mejor posible basándote en el contexto.\n"
                                "4. Mantén MAYÚSCULAS/minúsculas originales.\n"
                                "5. Usa '...' en lugar de '…'.\n"
                                "6. Devuelve solo el texto traducido, nada más."
                            )
                        },
                        {
                            "role": "user",
                            "content": text
                        }
                    ],
                    temperature=0.2,
                    max_tokens=500
                )

                translated = response.choices[0].message.content.strip()

                # Validación básica: no devolver traducciones vacías o demasiado cortas
                if not translated or len(translated) < max(3, int(len(text) * 0.1)):
                    raise ValueError("Traducción sospechosamente corta o vacía")

                return translated

            except Exception as e:
                # Reintento con backoff sencillo ante fallos puntuales de red/servicio
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logging.warning(f"Error de traducción (intento {attempt+1}): {e}. Reintentando en {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logging.error(f"❌ Error de traducción tras {max_retries} intentos: {e}")
                    return text  # Fall back al original
    
    def render_text_in_mask(self, image: np.ndarray, text: str, bbox: List[List[int]], 
                           font_path: Optional[str], text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        Renderizado robusto y general.
        - Maximiza el uso del espacio (0.90 ancho).
        - Evita solapamiento vertical usando métricas reales de fuente.
        - Centrado inteligente.
        """
        # --- SANITIZACIÓN GENERAL (PROFESIONAL) ---
        
        # 1. Normalización Unicode (NFKC):
        # Esto convierte variantes raras de caracteres en sus formas estándar.
        # Ejemplo: Convierte ligaduras raras o símbolos matemáticos en letras equivalentes si existen.
        # Es vital para inputs que vienen de OCR (Google Vision).
        text = unicodedata.normalize('NFKC', text)

        # 2. Filtro de "Imprimibles":
        # Elimina cualquier carácter de control, basura binaria o símbolos invisibles
        # que el OCR haya podido colar. Esto es la solución GENERAL a "ruido extraño".
        text = "".join(c for c in text if c.isprintable())

        # 3. Normalización de Cómics:
        # La elipsis (…) es un solo carácter, pero en fuentes de cómic suele fallar.
        # Lo estándar es convertirlo a tres puntos (...). Esto SÍ es una regla de negocio general válida.
        text = text.replace('…', '...')

        # 4. Limpieza final de espacios
        text = text.strip()

        # Validación de seguridad: Si después de limpiar no queda nada, nos vamos.
        if not text: return image

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)

        x_min, y_min, x_max, y_max = self.get_bbox_rect(bbox)
        box_width, box_height = x_max - x_min, y_max - y_min

        # Márgenes mínimos: aprovechamos casi todo el globo para que el texto quede grande
        safe_width = int(box_width * 0.998)
        safe_height = int(box_height * 0.998)

        # Preparación del bucle
        font_size = 80 # Empezar más grande para priorizar tamaño de letra
        min_font_size = 12
        final_font = None
        final_lines = []
        final_line_height = 0
        final_line_spacing = 0

        # Calcular palabra más larga para no romperla
        words = text.split()
        longest_word = max(words, key=len) if words else text

        while font_size >= min_font_size:
            try:
                font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
            except:
                font = ImageFont.load_default()

            # --- CÁLCULO VERTICAL ROBUSTO ---
            ascent, descent = font.getmetrics()
            # Altura real de la línea + un pequeño respiro (no compresivo, estándar)
            line_height = ascent + descent 
            # Espaciado mínimo para que no se peguen (más compacto)
            line_spacing = max(1, int(font_size * 0.02))

            # --- CÁLCULO HORIZONTAL ---
            # Usamos 'x' para asegurar que caben las mayúsculas anchas
            avg_char_w = font.getlength("x")
            if avg_char_w == 0: avg_char_w = 1 # Evitar div por cero
            
            # Estimación inicial de caracteres por línea
            # Añadimos margen positivo para evitar wrapping agresivo
            chars_per_line = max(1, int((safe_width / avg_char_w) + 8))
            
            # Ajuste: Si la palabra más larga es más ancha que safe_width, esta fuente NO vale
            if font.getlength(longest_word) > safe_width:
                font_size -= 2
                continue

            # Wrapping
            lines = textwrap.wrap(text, width=chars_per_line, break_long_words=False)
            
            # Verificar dimensiones reales
            max_w_line = 0
            for line in lines:
                w = font.getlength(line)
                if w > max_w_line: max_w_line = w
            
            total_text_height = (line_height * len(lines)) + (line_spacing * (len(lines) - 1))

            # ¿Cabe?
            if total_text_height <= safe_height and max_w_line <= safe_width:
                final_font = font
                final_lines = lines
                final_line_height = line_height
                final_line_spacing = line_spacing
                break # ¡Encontrado!
            
            font_size -= 2

        # Fallback si el bucle falla (usar fuente mínima)
        if final_font is None:
            final_font = ImageFont.truetype(font_path, min_font_size) if font_path else ImageFont.load_default()
            final_lines = textwrap.wrap(text, width=30)
            ascent, descent = final_font.getmetrics()
            final_line_height = ascent + descent
            final_line_spacing = 1

        # --- DIBUJADO ---
        # Calcular altura total final para centrado vertical dentro del bbox (el safe area es casi todo el bbox)
        total_h = (final_line_height * len(final_lines)) + (final_line_spacing * (len(final_lines) - 1))
        start_y = y_min + (box_height - total_h) // 2
        
        current_y = start_y
        if self.debug_visuals:
            # Borde exterior del bbox original (rojo):
            # Es la caja de texto que devuelve Vision para el globo detectado.
            draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=2)
            # Área segura usada para cálculo de texto (azul):
            # Es el bbox con un pequeño margen (0.2%) donde estimamos wrapping y centrado.
            safe_x1 = x_min + (box_width - safe_width) // 2
            safe_y1 = y_min + (box_height - safe_height) // 2
            safe_x2 = safe_x1 + safe_width
            safe_y2 = safe_y1 + safe_height
            draw.rectangle([safe_x1, safe_y1, safe_x2, safe_y2], outline=(0, 0, 255), width=2)
        
        for line in final_lines:
            w = final_font.getlength(line)
            # Centrado horizontal exacto respecto al bbox completo
            x = x_min + (box_width - w) // 2
            
            draw.text((x, current_y), line, font=final_font, fill=text_color)
            if self.debug_visuals:
                # Caja de cada línea renderizada (verde):
                # Muestra exactamente la anchura de la línea colocada.
                draw.rectangle([x, current_y, x + w, current_y + final_line_height], outline=(0, 200, 0), width=1)
            current_y += final_line_height + final_line_spacing

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def process_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Función principal: detecta, traduce y renderiza texto.
        Mejoras:
        - Procesamiento en paralelo de globos.
        - Manejo de errores por globo (uno fallido no detiene el resto).
        - Logging informativo.
        """
        logger = logging.getLogger(__name__)

        original_image = self.load_image(image_path)
        processed_image = original_image.copy()

        try:
            text_results = self.detect_text_google(image_path)
        except Exception as e:
            logger.error(f"Error en detección de texto: {e}")
            return original_image, processed_image

        if not text_results:
            return original_image, processed_image

        # Función interna para procesar un globo
        def process_single_bubble(bubble_data):
            bbox, text, confidence = bubble_data
            try:
                height, width = original_image.shape[:2]
                x_min, y_min, x_max, y_max = self.get_bbox_rect(bbox)
                box_area = (x_max - x_min) * (y_max - y_min)
                total_area = height * width

                # Filtrar solo globos gigantes (títulos de página completa)
                if box_area > (total_area * self.max_bubble_area_ratio):
                    return None

                # Traducir
                translated_text = self.translate_text(text)
                if not translated_text.strip():
                    translated_text = text

                # Máscara: detecta contorno real del globo y color de fondo
                mask, (rx1, ry1, rx2, ry2) = self.get_contour_mask(original_image, bbox)
                target_bbox = bbox
                text_color = (0, 0, 0)

                if mask is not None:
                    fill_color = self.get_bubble_color_from_mask(original_image, mask, (rx1, ry1, rx2, ry2))
                    brightness = (fill_color[2] * 0.299 + fill_color[1] * 0.587 + fill_color[0] * 0.114)
                    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

                    # Aplicar relleno
                    roi_target = processed_image[ry1:ry2, rx1:rx2].copy()
                    roi_target[mask == 255] = fill_color

                    # Actualizar bbox según la máscara
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        mx, my, mw, mh = cv2.boundingRect(largest)
                        target_bbox = [
                            [rx1 + mx, ry1 + my],
                            [rx1 + mx + mw, ry1 + my],
                            [rx1 + mx + mw, ry1 + my + mh],
                            [rx1 + mx, ry1 + my + mh]
                        ]

                    return (target_bbox, translated_text, text_color, roi_target, (ry1, ry2, rx1, rx2))
                else:
                    # Fallback: relleno blanco
                    x1, y1, x2, y2 = self.get_bbox_rect(bbox)
                    roi_target = processed_image[y1:y2, x1:x2].copy()
                    cv2.rectangle(roi_target, (0, 0), (x2 - x1, y2 - y1), (255, 255, 255), -1)
                    return (bbox, translated_text, text_color, roi_target, (y1, y2, x1, x2))

            except Exception as e:
                logger.error(f"Error procesando globo '{text[:50]}...': {e}")
                return None

        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_bubble = {executor.submit(process_single_bubble, bubble): bubble for bubble in text_results}
            for future in as_completed(future_to_bubble):
                result = future.result()
                if result is not None:
                    results.append(result)

        # Aplicar resultados
        for target_bbox, translated_text, text_color, roi_target, (ry1, ry2, rx1, rx2) in results:
            try:
                processed_image[ry1:ry2, rx1:rx2] = roi_target
                processed_image = self.render_text_in_mask(
                    processed_image, translated_text, target_bbox, self.font_path, text_color
                )
            except Exception as e:
                logger.error(f"Error aplicando resultado: {e}")
                continue

        return original_image, processed_image

