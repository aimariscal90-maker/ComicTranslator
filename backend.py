import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Tuple, Optional
import textwrap

class ComicTranslator:
    def __init__(self, font_path: Optional[str] = None):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        self.client = OpenAI(api_key=api_key)
        # IMPORTANTE: paragraph=True ayuda a leer frases cortas mejor
        self.reader = easyocr.Reader(['en'], gpu=False) 
        
        if font_path is None:
            possible_paths = ["comic_font.ttf", "comic_font.ttf.ttf"]
            self.font_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.font_path = path
                    break
            if self.font_path is None:
                print("Warning: Comic font file not found. Using default font.")
        else:
            self.font_path = font_path
        
        # --- PARÁMETROS AGRESIVOS PARA NO PERDER TEXTO ---
        self.max_bubble_area_ratio = 0.40  # Subido a 40%
        self.min_bubble_area_ratio = 0.0005 # Bajado al mínimo (para pillar "Ouch")
        self.edge_margin_ratio = 0.001      # Pegado al borde permitido

    def load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Could not load image from {image_path}")
        return image

    def get_bbox_rect(self, bbox: List[List[int]]) -> Tuple[int, int, int, int]:
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        return int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))

    def is_valid_speech_bubble(self, bbox: List[List[int]], image_shape: Tuple[int, int]) -> bool:
        height, width = image_shape[:2]
        x_min, y_min, x_max, y_max = self.get_bbox_rect(bbox)
        box_area = (x_max - x_min) * (y_max - y_min)
        total_area = width * height
        
        if box_area > (total_area * self.max_bubble_area_ratio): return False
        if box_area < (total_area * self.min_bubble_area_ratio): return False
        return True

    def detect_text(self, image: np.ndarray) -> List[Tuple[List[List[int]], str, float]]:
        # CAMBIO CLAVE: paragraph=False para detectar CADA palabra, luego nosotros agrupamos.
        # Ajustamos canvas_size para pillar texto pequeño
        results = self.reader.readtext(image, paragraph=False, canvas_size=2048, mag_ratio=1.5)
        valid_results = []
        for (bbox, text, confidence) in results:
            if self.is_valid_speech_bubble(bbox, image.shape):
                valid_results.append((bbox, text.strip(), confidence))
        return valid_results

    def cluster_boxes(self, detections):
        if not detections: return []
        boxes = []
        for bbox, text, conf in detections:
            x_min, y_min, x_max, y_max = self.get_bbox_rect(bbox)
            boxes.append({'bbox': bbox, 'text': text, 'conf': conf, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max, 'merged': False, 'width': x_max-x_min, 'height': y_max-y_min})
        
        clusters = []
        for i, box_a in enumerate(boxes):
            if box_a['merged']: continue
            cluster = [box_a]
            box_a['merged'] = True
            
            changed = True
            while changed:
                changed = False
                for box_b in boxes:
                    if box_b['merged']: continue
                    for c_box in cluster:
                        v_gap = max(0, max(box_b['y_min'], c_box['y_min']) - min(box_b['y_max'], c_box['y_max']))
                        avg_h = (box_b['height'] + c_box['height']) / 2
                        h_overlap = max(0, min(box_b['x_max'], c_box['x_max']) - max(box_b['x_min'], c_box['x_min']))
                        
                        # Lógica de agrupación más flexible
                        if v_gap < avg_h * 2.0 and (h_overlap > 0 or abs(box_b['x_min'] - c_box['x_min']) < 100):
                            cluster.append(box_b); box_b['merged'] = True; changed = True; break
            
            x1 = min(b['x_min'] for b in cluster); y1 = min(b['y_min'] for b in cluster)
            x2 = max(b['x_max'] for b in cluster); y2 = max(b['y_max'] for b in cluster)
            cluster.sort(key=lambda b: b['y_min']) # Ordenar de arriba a abajo
            text = ' '.join(b['text'] for b in cluster)
            bbox = [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            avg_conf = sum(b['conf'] for b in cluster) / len(cluster)
            clusters.append((bbox, text, avg_conf))
        return clusters

    def get_contour_mask(self, image: np.ndarray, bbox: List[List[int]]) -> Tuple[Optional[np.ndarray], Tuple[int,int,int,int]]:
        x_min, y_min, x_max, y_max = self.get_bbox_rect(bbox)
        h_img, w_img = image.shape[:2]
        
        pad = 25
        x1 = max(0, x_min - pad); y1 = max(0, y_min - pad)
        x2 = min(w_img, x_max + pad); y2 = min(h_img, y_max + pad)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0: return None, (0,0,0,0)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Verificar si necesitamos invertir (si el fondo es negro y letras blancas)
        # Calculamos la media de los bordes. Si los bordes son negros, el fondo es negro.
        edge_mean = (np.mean(binary[0,:]) + np.mean(binary[-1,:]) + np.mean(binary[:,0]) + np.mean(binary[:,-1])) / 4
        if edge_mean < 127: 
             # Fondo negro detectado en binario, probablemente Otsu lo invirtió bien para detectar letras,
             # pero nosotros queremos el GLOBO.
             # Para detección de globos, asumimos que el globo es más claro que el borde.
             pass

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, (x1,y1,x2,y2)
        
        orig_cx = (x_min + x_max) // 2 - x1
        orig_cy = (y_min + y_max) // 2 - y1
        
        best_cnt = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50: continue # Ruido mínimo
            dist = cv2.pointPolygonTest(cnt, (orig_cx, orig_cy), True)
            if dist > -20: # Permitir margen
                if area > max_area: max_area = area; best_cnt = cnt
        
        if best_cnt is None: best_cnt = max(contours, key=cv2.contourArea)

        mask = np.zeros_like(gray); cv2.drawContours(mask, [best_cnt], -1, 255, -1)
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)
        return mask, (x1, y1, x2, y2)

    def get_bubble_color_from_mask(self, image: np.ndarray, mask: np.ndarray, roi_coords: Tuple) -> Tuple[int, int, int]:
        x1, y1, x2, y2 = roi_coords
        roi = image[y1:y2, x1:x2]
        masked_pixels = roi[mask == 255]
        if len(masked_pixels) < 10: return (255, 255, 255)
        # Usar mediana para evitar ruido del texto negro
        avg_color = np.median(masked_pixels, axis=0)
        return (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))

    def translate_text(self, text):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "Translate to Spanish (Spain). Concise."}, {"role": "user", "content": text}],
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except: return text

    def process_image(self, image_path: str):
        original_image = self.load_image(image_path)
        processed_image = original_image.copy()
        
        text_results = self.detect_text(original_image)
        if not text_results: return original_image, processed_image
        
        clustered_results = self.cluster_boxes(text_results)
        
        for bbox, text, conf in clustered_results:
            # UMBRAL BAJÍSIMO PARA PILLAR TODO
            if conf < 0.1: continue 
            
            translated_text = self.translate_text(text)
            mask, (rx1, ry1, rx2, ry2) = self.get_contour_mask(original_image, bbox)
            
            target_bbox = bbox
            text_color = (0,0,0)
            
            if mask is not None:
                fill_color = self.get_bubble_color_from_mask(original_image, mask, (rx1, ry1, rx2, ry2))
                
                # --- CORRECCIÓN DE COLOR PARA CAJAS GRISES ---
                # Si la diferencia entre canales es baja (es gris), asegúrate de que no sea blanco puro
                # Esto es un hack visual: si es muy claro (>230), lo forzamos a blanco para limpiar.
                # Si es gris medio, lo dejamos.
                brightness = sum(fill_color)/3
                if brightness > 230: fill_color = (255,255,255)
                
                roi_target = processed_image[ry1:ry2, rx1:rx2]
                roi_target[mask == 255] = fill_color
                
                # Texto blanco o negro
                text_color = (0, 0, 0) if brightness > 100 else (255, 255, 255)
                
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    largest = max(cnts, key=cv2.contourArea)
                    mx, my, mw, mh = cv2.boundingRect(largest)
                    target_bbox = [[rx1 + mx, ry1 + my], [rx1 + mx + mw, ry1 + my], [rx1 + mx + mw, ry1 + my + mh], [rx1 + mx, ry1 + my + mh]]
            else:
                x1, y1, x2, y2 = self.get_bbox_rect(bbox)
                cv2.rectangle(processed_image, (x1, y1), (x2, y2), (255,255,255), -1)

            processed_image = self.render_text_in_mask(processed_image, translated_text, target_bbox, self.font_path, text_color)
                
        return original_image, processed_image

    def render_text_in_mask(self, image, text, bbox, font_path, text_color=(0,0,0)):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        x_min, y_min, x_max, y_max = self.get_bbox_rect(bbox)
        box_width, box_height = x_max - x_min, y_max - y_min
        
        # PADDING AUMENTADO (Para que no choque con los bordes)
        safe_width = box_width * 0.85 
        safe_height = box_height * 0.90
        
        font_size = 50
        min_font_size = 8
        final_font = None; final_lines = []
        
        while font_size >= min_font_size:
            try: font = ImageFont.truetype(font_path, font_size)
            except: font = ImageFont.load_default()
            
            avg_char_w = font.getlength("a")
            # Ajuste de anchura conservador
            chars_per_line = max(1, int(safe_width / (avg_char_w * 0.9)))
            lines = textwrap.wrap(text, width=chars_per_line)
            
            h_tot = sum([font.getbbox(l)[3]-font.getbbox(l)[1] for l in lines])
            # Comprobar que cabe en altura y anchura
            max_w_line = max([font.getlength(l) for l in lines]) if lines else 0
            
            if h_tot <= safe_height and max_w_line <= safe_width:
                final_font = font; final_lines = lines; break
            font_size -= 2
            
        if not final_font: final_font = ImageFont.load_default(); final_lines = [text]

        line_spacing = 4
        total_text_height = sum([final_font.getbbox(l)[3]-final_font.getbbox(l)[1] for l in final_lines]) + (len(final_lines)-1) * line_spacing
        y = y_min + (box_height - total_text_height) // 2
        
        for line in final_lines:
            w = final_font.getlength(line)
            x = x_min + (box_width - w) // 2
            draw.text((x, y), line, font=final_font, fill=text_color)
            y += final_font.getbbox(line)[3] - final_font.getbbox(line)[1] + line_spacing
            
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)