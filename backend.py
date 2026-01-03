# ============================================================================
# IMPORTACIONES
# ============================================================================
# Librerías estándar de Python
import os  # Para trabajar con archivos y rutas del sistema
import io  # Para leer archivos como bytes
import math  # Para operaciones matemáticas (raíz cuadrada, etc.)
import textwrap  # Para dividir texto en líneas
import unicodedata
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
    
    def __init__(self, font_path: Optional[str] = None):
        """
        Inicializa el traductor de cómics.
        
        PASO 1: Cargar variables de entorno (.env)
        PASO 2: Configurar cliente de OpenAI
        PASO 3: Configurar cliente de Google Cloud Vision
        PASO 4: Buscar y cargar la fuente de cómic
        PASO 5: Configurar parámetros de detección
        
        Args:
            font_path: Ruta al archivo de fuente TTF (si es None, busca automáticamente)
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
        Detecta texto en la imagen usando Google Cloud Vision API.
        
        Google Vision detecta BLOQUES completos de texto (párrafos),
        lo que resuelve el problema de fragmentación (texto dividido en palabras).
        
        Proceso:
        1. Lee la imagen como bytes
        2. Envía a Google Cloud Vision
        3. Procesa la respuesta (Páginas -> Bloques -> Párrafos -> Palabras)
        4. Reconstruye el texto completo de cada bloque
        5. Filtra detecciones muy pequeñas (ruido)
        
        Args:
            image_path: Ruta al archivo de imagen
            
        Returns:
            Lista de tuplas: (bbox, texto, confianza)
        """
        # Leer el archivo de imagen como bytes (requerido por Google Vision)
        try:
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
        except Exception as e:
            raise IOError(f"Error al leer el archivo de imagen: {e}")
        
        # Crear objeto Image de Google Vision
        image = vision.Image(content=content)
        
        # Llamar a la API de Google Cloud Vision
        # document_text_detection es mejor para texto denso (cómics)
        try:
            response = self.vision_client.document_text_detection(image=image)
        except Exception as e:
            raise RuntimeError(f"Error en Google Cloud Vision API: {e}")
        
        # Verificar si hay errores en la respuesta
        if response.error.message:
            raise Exception(f'Error en Google Cloud Vision API: {response.error.message}')
        
        # Lista para almacenar todas las detecciones
        detections = []
        
        # ====================================================================
        # Procesar la respuesta de Google Vision
        # ====================================================================
        # Estructura de Google: Páginas -> Bloques -> Párrafos -> Palabras -> Símbolos
        # Nosotros queremos los BLOQUES para obtener frases completas
        
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                block_text = ""
                
                # Reconstruir el texto del bloque uniendo párrafos y palabras
                for paragraph in block.paragraphs:
                    para_text = ""
                    for word in paragraph.words:
                        # Unir todos los símbolos (letras) de cada palabra
                        word_text = ''.join([symbol.text for symbol in word.symbols])
                        para_text += word_text + " "  # Agregar espacio entre palabras
                    block_text += para_text
                
                # Limpiar espacios al inicio y final
                block_text = block_text.strip()
                
                # Saltar bloques vacíos
                if not block_text:
                    continue
                
                # Obtener coordenadas del bloque (Bounding Box)
                vertices = block.bounding_box.vertices
                
                # Convertir formato de Google [[x,y], [x,y]...] a nuestro formato
                bbox = [[v.x, v.y] for v in vertices]
                
                # Obtener confianza del bloque (Google da confianza por palabra, usamos la del bloque)
                confidence = block.confidence if hasattr(block, 'confidence') else 0.9
                
                # ============================================================
                # Filtro de tamaño - MUY PERMISIVO para capturar texto pequeño
                # ============================================================
                x_coords = [v.x for v in vertices]
                y_coords = [v.y for v in vertices]
                w = max(x_coords) - min(x_coords)  # Ancho del bloque
                h = max(y_coords) - min(y_coords)  # Alto del bloque
                
                # Solo filtrar detecciones microscópicas (probablemente ruido)
                # Reducido a 1 píxel para ser lo más permisivo posible
                if w < 1 or h < 1:
                    continue
                
                # Agregar a la lista de detecciones
                detections.append((bbox, block_text, confidence))
        
        return detections
    
    def get_contour_mask(self, image: np.ndarray, bbox: List[List[int]]) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
        """
        MODIFICADO (General): Usa precisión quirúrgica.
        - Padding reducido (10 -> 4) para no invadir vecinos.
        - Dilatación suave (2 -> 1) para tapar solo lo justo.
        """
        x_min, y_min, x_max, y_max = self.get_bbox_rect(bbox)
        height, width = image.shape[:2]
        
        # CAMBIO: Usamos un margen muy pequeño. Solo queremos holgura para curvas, no espacio extra.
        padding = 4 
        roi_x_min = max(0, x_min - padding)
        roi_y_min = max(0, y_min - padding)
        roi_x_max = min(width, x_max + padding)
        roi_y_max = min(height, y_max + padding)
        
        roi = image[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        if roi.size == 0:
            return None, (0, 0, 0, 0)
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu es bueno y general para separar texto del fondo
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, (roi_x_min, roi_y_min, roi_x_max, roi_y_max)
        
        # Lógica para encontrar el contorno que realmente contiene el texto
        orig_cx = (x_min + x_max) // 2 - roi_x_min
        orig_cy = (y_min + y_max) // 2 - roi_y_min
        
        best_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50: continue # Ignorar ruido
            
            # Verificar si el centro del texto está dentro o MUY cerca del contorno
            dist = cv2.pointPolygonTest(contour, (orig_cx, orig_cy), True)
            if dist > -20: 
                if area > max_area:
                    max_area = area
                    best_contour = contour
        
        if best_contour is None:
            # Fallback general: Si no coincide el centro, cogemos el más grande (suele ser el globo)
            best_contour = max(contours, key=cv2.contourArea)
        
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [best_contour], -1, 255, -1)
        
        # CAMBIO: Solo 1 iteración. Tapamos el texto original, pero respetamos los bordes negros.
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask, (roi_x_min, roi_y_min, roi_x_max, roi_y_max)
    
    def get_bubble_color_from_mask(self, image: np.ndarray, mask: np.ndarray, 
                                   roi_coords: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
        """
        Detecta el color dominante del fondo del globo desde el área de la máscara.
        
        En lugar de usar un rectángulo simple, solo analiza los píxeles
        dentro de la máscara (la forma real del globo).
        
        Proceso:
        1. Extraer píxeles solo donde la máscara es blanca (área del globo)
        2. Usar mediana (no promedio) para evitar ruido del texto negro
        3. Retornar color en formato BGR
        
        Args:
            image: Imagen completa (formato BGR)
            mask: Máscara binaria (blanco = área del globo)
            roi_coords: Coordenadas (x1, y1, x2, y2) del ROI
            
        Returns:
            Tupla de color BGR (Blue, Green, Red)
        """
        x1, y1, x2, y2 = roi_coords
        roi = image[y1:y2, x1:x2]
        
        # Verificar que las dimensiones coincidan
        if roi.shape[:2] != mask.shape[:2]:
            return (255, 255, 255)  # Por defecto blanco
        
        # Extraer píxeles solo donde la máscara es blanca (área del globo)
        masked_pixels = roi[mask == 255]
        
        # Si hay muy pocos píxeles, retornar blanco por defecto
        if len(masked_pixels) < 10:
            return (255, 255, 255)
        
        # Usar mediana (no promedio) para evitar ruido del texto negro
        # La mediana es más robusta a valores extremos
        avg_color = np.median(masked_pixels, axis=0)
        
        return (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))
    
    def translate_text(self, text: str) -> str:
        """
        Traduce texto de inglés a español (España) usando OpenAI API.
        
        MANTIENE ESTRICTAMENTE LAS MAYÚSCULAS/MINÚSCULAS ORIGINALES
        para preservar los matices de voz del cómic.
        
        Ejemplos:
        - "HELP!" -> "¡AYUDA!" (mantiene mayúsculas)
        - "Help!" -> "¡Ayuda!" (mantiene minúsculas)
        
        Proceso:
        1. Crear mensaje de sistema con instrucciones específicas
        2. Enviar texto a OpenAI
        3. Recibir traducción
        4. Limpiar espacios
        
        Args:
            text: Texto en inglés a traducir
            
        Returns:
            Texto traducido al español con mayúsculas/minúsculas preservadas
        """
        try:
            # Llamar a la API de OpenAI con instrucciones estrictas de fidelidad
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Modelo económico y rápido
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un traductor profesional de cómics EN->ES (España). "
                            "Reglas CRÍTICAS:\n"
                            "1. Mantén el significado exacto y el tono.\n"
                            "2. NO traduzcas Nombres Propios de personajes (ej: 'Immortus', 'Death's Head', 'Stark', 'Maker'). Déjalos tal cual.\n"
                            "3. Si el texto parece cortado o incompleto, tradúcelo lo mejor posible basándote en el contexto, NO lo dejes en inglés.\n"
                            "4. Mantén MAYÚSCULAS/minúsculas originales.\n"
                            "5. Solo devuelve el texto traducido, nada más."
                        )
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.2,  # Baja temperatura = más determinista
                max_tokens=500  # Límite de tokens
            )
            
            # Extraer y limpiar la traducción
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error de traducción: {e}")
            return text  # En caso de error, retornar texto original
    
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

        # CAMBIO: Márgenes agresivos pero seguros (General para óvalos y rectángulos)
        safe_width = int(box_width * 0.90) 
        safe_height = int(box_height * 0.90)

        # Preparación del bucle
        font_size = 60 # Empezar grande
        min_font_size = 10
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
            # Espaciado mínimo para que no se peguen
            line_spacing = int(font_size * 0.1) 

            # --- CÁLCULO HORIZONTAL ---
            # Usamos 'M' para asegurar que caben las mayúsculas anchas
            avg_char_w = font.getlength("M")
            if avg_char_w == 0: avg_char_w = 1 # Evitar div por cero
            
            # Estimación inicial de caracteres por línea
            chars_per_line = max(1, int(safe_width / avg_char_w))
            
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
            final_lines = textwrap.wrap(text, width=15)
            ascent, descent = final_font.getmetrics()
            final_line_height = ascent + descent
            final_line_spacing = 2

        # --- DIBUJADO ---
        # Calcular altura total final para centrado vertical
        total_h = (final_line_height * len(final_lines)) + (final_line_spacing * (len(final_lines) - 1))
        start_y = y_min + (box_height - total_h) // 2
        
        current_y = start_y
        for line in final_lines:
            w = final_font.getlength(line)
            # Centrado horizontal exacto
            x = x_min + (box_width - w) // 2
            
            draw.text((x, current_y), line, font=final_font, fill=text_color)
            current_y += final_line_height + final_line_spacing

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def process_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Función principal de procesamiento: detecta, traduce y renderiza texto.
        
        Este es el método principal que orquesta todo el proceso:
        
        1. Cargar imagen
        2. Detectar texto usando Google Cloud Vision
        3. Para cada detección:
           a. Filtrar cajas muy grandes (títulos de página completa)
           b. Traducir texto
           c. Obtener máscara del globo (forma orgánica)
           d. Detectar color del fondo del globo
           e. Rellenar el área del globo con el color detectado
           f. Determinar color del texto (negro o blanco según brillo del fondo)
           g. Renderizar texto traducido
        
        Args:
            image_path: Ruta al archivo de imagen
            
        Returns:
            Tupla de (imagen_original, imagen_procesada)
        """
        # Cargar imagen
        original_image = self.load_image(image_path)
        processed_image = original_image.copy()
        
        # ====================================================================
        # PASO 1: Detectar texto usando Google Cloud Vision
        # ====================================================================
        # Google Vision detecta bloques completos (resuelve fragmentación)
        text_results = self.detect_text_google(image_path)
        
        # Si no hay detecciones, retornar imágenes sin cambios
        if not text_results:
            return original_image, processed_image
        
        # ====================================================================
        # PASO 2: Procesar cada detección - MÁXIMA SENSITIVIDAD
        # ====================================================================
        for bbox, text, confidence in text_results:
            # Filtrar solo cajas muy grandes (títulos de página completa)
            # PERMITIR cajas pequeñas - queremos capturar texto pequeño como "Ouch", "Eh?", "NO"
            height, width = original_image.shape[:2]
            x_min, y_min, x_max, y_max = self.get_bbox_rect(bbox)
            box_area = (x_max - x_min) * (y_max - y_min)
            total_area = height * width
            
            # Filtrar solo si es demasiado grande (títulos de página completa)
            # NO filtrar por área mínima - queremos capturar texto pequeño
            if box_area > (total_area * self.max_bubble_area_ratio):
                continue
            
            # NO filtrar por confianza - Google Vision es suficientemente preciso, queremos todo
            # NO filtrar por longitud de texto - queremos traducir "NO", "AY", "Eh?", "Ouch", etc.
            
            # ================================================================
            # Traducir texto (mantiene mayúsculas/minúsculas originales)
            # ================================================================
            translated_text = self.translate_text(text)
            # Si la traducción llega vacía por algún motivo, usar el texto original
            if not translated_text.strip():
                translated_text = text
            
            # ================================================================
            # Obtener máscara del contorno para inpainting orgánico
            # ================================================================
            mask, (rx1, ry1, rx2, ry2) = self.get_contour_mask(original_image, bbox)
            
            target_bbox = bbox
            text_color = (0, 0, 0)  # Por defecto negro
            
            if mask is not None:
                # Obtener color del globo
                fill_color = self.get_bubble_color_from_mask(original_image, mask, (rx1, ry1, rx2, ry2))
                
                # --- CAMBIO: ELIMINAMOS EL FORZADO DE BLANCO ---
                # Borramos el bloque "if brightness > 220: fill_color = (255,255,255)"
                # Confiamos en que get_bubble_color_from_mask nos dé el color real (sea gris, crema o negro).
                
                # Calcular brillo para decidir color de texto
                # Fórmula estándar de luminancia: (0.299*R + 0.587*G + 0.114*B)
                brightness = (fill_color[2] * 0.299 + fill_color[1] * 0.587 + fill_color[0] * 0.114)
                
                # Aplicar color de relleno (Inpainting)
                roi_target = processed_image[ry1:ry2, rx1:rx2]
                roi_target[mask == 255] = fill_color
                
                # Decidir color del texto (Contraste General)
                # Si el fondo es oscuro (<128), texto blanco. Si es claro, texto negro.
                text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
                
                # Actualizar target_bbox para que coincida con los límites de la máscara
                # para mejor centrado
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    mx, my, mw, mh = cv2.boundingRect(largest)
                    
                    # AJUSTE PRO: Añadimos un micro-margen interno (padding negativo)
                    # para asegurarnos de que el texto no toque los bordes blancos irregulares
                    # Es como dejar un marco de seguridad dentro del cuadro.
                    
                    # Corrección de coordenadas relativas a absolutas
                    final_x = rx1 + mx
                    final_y = ry1 + my
                    final_w = mw
                    final_h = mh

                    target_bbox = [
                        [final_x, final_y],
                        [final_x + final_w, final_y],
                        [final_x + final_w, final_y + final_h],
                        [final_x, final_y + final_h]
                    ]
            else:
                # Fallback: relleno simple con rectángulo blanco
                x1, y1, x2, y2 = self.get_bbox_rect(bbox)
                cv2.rectangle(processed_image, (x1, y1), (x2, y2), (255, 255, 255), -1)
            
            # ================================================================
            # Renderizar texto traducido
            # ================================================================
            processed_image = self.render_text_in_mask(
                processed_image, translated_text, target_bbox, self.font_path, text_color
            )
        
        return original_image, processed_image
