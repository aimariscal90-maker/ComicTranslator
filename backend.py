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
    """
    Class-based architecture for translating comic book pages.
    Detects speech bubbles, translates text, and renders translations.
    """
    
    def __init__(self, font_path: Optional[str] = None):
        """
        Initialize the ComicTranslator.
        
        Args:
            font_path: Path to the comic font TTF file (auto-detects if None)
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        self.client = OpenAI(api_key=api_key)
        
        # Initialize EasyOCR reader (English only for now)
        # Set gpu=True if you have a compatible GPU, otherwise False
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        # Auto-detect font path if not provided
        if font_path is None:
            # Try common font file names
            possible_paths = ["comic_font.ttf", "comic_font.ttf.ttf"]
            self.font_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.font_path = path
                    break
            if self.font_path is None:
                # Fallback to default if not found, but warn
                print("Warning: Comic font file not found. Using default font.")
                self.font_path = None 
        else:
            if not os.path.exists(font_path):
                raise FileNotFoundError(f"Font file not found: {font_path}")
            self.font_path = font_path
        
        # Thresholds for bubble detection
        self.max_bubble_area_ratio = 0.20  # 20% of page
        self.edge_margin_ratio = 0.05  # 5% margin from edges
        
        # Clustering parameters
        self.vertical_proximity_factor = 1.5  # 1.5x line height for vertical clustering
        self.horizontal_overlap_threshold = 0.3  # 30% horizontal overlap for alignment
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file path.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image
    
    def get_bbox_rect(self, bbox: List[List[int]]) -> Tuple[int, int, int, int]:
        """
        Get bounding box as (x_min, y_min, x_max, y_max).
        """
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        return int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))
    
    def is_valid_speech_bubble(self, bbox: List[List[int]], image_shape: Tuple[int, int]) -> bool:
        """
        Determine if a text bounding box is a valid speech bubble.
        Filters out titles and large text boxes.
        """
        height, width = image_shape[:2]
        x_min, y_min, x_max, y_max = self.get_bbox_rect(bbox)
        
        box_width = x_max - x_min
        box_height = y_max - y_min
        box_area = box_width * box_height
        total_area = width * height
        
        # Filter 1: Too large (>20% of page)
        if box_area > (total_area * self.max_bubble_area_ratio):
            return False
        
        # Filter 2: At extreme edges (likely titles/SFX)
        edge_margin_x = width * self.edge_margin_ratio
        edge_margin_y = height * self.edge_margin_ratio
        
        # Check if box is too close to any edge
        if (x_min < edge_margin_x or 
            x_max > (width - edge_margin_x) or
            y_min < edge_margin_y or
            y_max > (height - edge_margin_y)):
            return False
        
        return True
    
    def detect_text(self, image: np.ndarray) -> List[Tuple[List[List[int]], str, float]]:
        """
        Detect text in the image using EasyOCR.
        """
        results = self.reader.readtext(image)
        
        # Filter valid speech bubbles
        valid_results = []
        for (bbox, text, confidence) in results:
            if self.is_valid_speech_bubble(bbox, image.shape):
                valid_results.append((bbox, text.strip(), confidence))
        
        return valid_results
    
    def cluster_boxes(self, detections: List[Tuple[List[List[int]], str, float]]) -> List[Tuple[List[List[int]], str, float]]:
        """
        Group vertically close text boxes into a single bubble.
        """
        if not detections:
            return []
        
        # Convert to list of dicts for easier manipulation
        boxes = []
        for bbox, text, confidence in detections:
            x_min, y_min, x_max, y_max = self.get_bbox_rect(bbox)
            boxes.append({
                'bbox': bbox,
                'text': text,
                'confidence': confidence,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'width': x_max - x_min,
                'height': y_max - y_min,
                'merged': False
            })
        
        clusters = []
        
        for i, box_a in enumerate(boxes):
            if box_a['merged']:
                continue
            
            # Start a new cluster with box_a
            cluster_boxes = [box_a]
            box_a['merged'] = True
            
            # Find boxes that should be merged with this cluster
            changed = True
            while changed:
                changed = False
                for box_b in boxes:
                    if box_b['merged']:
                        continue
                    
                    # Check if box_b should be merged with any box in the cluster
                    for cluster_box in cluster_boxes:
                        # Calculate vertical distance
                        vertical_gap = max(0, max(box_b['y_min'], cluster_box['y_min']) - 
                                            min(box_b['y_max'], cluster_box['y_max']))
                        
                        # Calculate horizontal overlap
                        horizontal_overlap = max(0, min(box_b['x_max'], cluster_box['x_max']) - 
                                                 max(box_b['x_min'], cluster_box['x_min']))
                        min_width = min(box_b['width'], cluster_box['width'])
                        overlap_ratio = horizontal_overlap / min_width if min_width > 0 else 0
                        
                        # Check if vertically close (within 1.5x average line height)
                        avg_height = (box_b['height'] + cluster_box['height']) / 2
                        max_vertical_distance = avg_height * self.vertical_proximity_factor
                        
                        # Check if horizontally aligned (30% overlap)
                        if (vertical_gap <= max_vertical_distance and 
                            overlap_ratio >= self.horizontal_overlap_threshold):
                            cluster_boxes.append(box_b)
                            box_b['merged'] = True
                            changed = True
                            break
            
            # Merge the cluster into a single bounding box
            if len(cluster_boxes) > 1:
                x_min = min(box['x_min'] for box in cluster_boxes)
                y_min = min(box['y_min'] for box in cluster_boxes)
                x_max = max(box['x_max'] for box in cluster_boxes)
                y_max = max(box['y_max'] for box in cluster_boxes)
                
                # Create merged bbox (rectangle format)
                merged_bbox = [
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ]
                
                # Merge text with spaces
                merged_text = ' '.join(box['text'] for box in cluster_boxes)
                
                # Average confidence
                avg_confidence = sum(box['confidence'] for box in cluster_boxes) / len(cluster_boxes)
                
                clusters.append((merged_bbox, merged_text, avg_confidence))
            else:
                # Single box, no merging needed
                clusters.append((box_a['bbox'], box_a['text'], box_a['confidence']))
        
        return clusters
    
    def get_bubble_color(self, image: np.ndarray, bbox: List[List[int]]) -> Tuple[int, int, int]:
        """
        Detect the dominant background color of a speech bubble.
        Returns standard Python integers to avoid OpenCV errors.
        """
        x_min, y_min, x_max, y_max = self.get_bbox_rect(bbox)
        
        # Add small padding to avoid edge artifacts
        padding = 3
        height, width = image.shape[:2]
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)
        
        # Extract ROI
        roi = image[y_min:y_max, x_min:x_max]
        
        if roi.size == 0:
            return (255, 255, 255)  # Default to white
        
        # Reshape to list of pixels
        pixels = roi.reshape(-1, 3)
        
        # Filter out dark pixels (text) - threshold for "dark" is brightness < 80
        # Convert BGR to grayscale for brightness calculation
        gray_pixels = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).reshape(-1)
        brightness_threshold = 80
        
        # Keep only bright pixels (background)
        bright_mask = gray_pixels > brightness_threshold
        background_pixels = pixels[bright_mask]
        
        # If no bright pixels found, use all pixels (fallback)
        if len(background_pixels) == 0:
            background_pixels = pixels
        
        # Use simple mean if few pixels
        if len(background_pixels) < 10:
             avg_color = np.mean(background_pixels, axis=0)
             return (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))

        # Use K-Means clustering to find dominant color
        background_pixels = background_pixels.reshape(-1, 3).astype(np.float32)
        
        k = min(3, len(background_pixels))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(background_pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Find the cluster with most pixels (dominant color)
        unique, counts = np.unique(labels, return_counts=True)
        dominant_cluster_idx = unique[np.argmax(counts)]
        dominant_color = centers[dominant_cluster_idx]
        
        # FORCE CAST TO INT
        return (int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2]))
    
    def translate_text(self, text: str) -> str:
        """
        Translate text from English to Spanish (Spain) using OpenAI API.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional comic translator. Translate the English text to Spanish (Spain). Keep it concise. Do not explain."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text on error
    
    def inpaint_text(self, image: np.ndarray, bbox: List[List[int]], 
                    fill_color: Tuple[int, int, int]) -> np.ndarray:
        """
        Draw a rectangle over the original text using the detected bubble color.
        """
        image_copy = image.copy()
        
        x_min, y_min, x_max, y_max = self.get_bbox_rect(bbox)
        
        # Add inflation (expand by 8-10 pixels) to cover artifacts
        inflation = 8
        height, width = image.shape[:2]
        x_min = max(0, x_min - inflation)
        y_min = max(0, y_min - inflation)
        x_max = min(width, x_max + inflation)
        y_max = min(height, y_max + inflation)
        
        # Draw rectangle with detected color (BGR format for OpenCV)
        # Ensure fill_color is tuple of ints
        safe_color = (int(fill_color[0]), int(fill_color[1]), int(fill_color[2]))
        cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), safe_color, -1)
        
        return image_copy

    def render_text(self, image: np.ndarray, text: str, bbox: List[List[int]], 
                   font_path: Optional[str] = None) -> np.ndarray:
        """
        Render translated text on the image using the comic font with dynamic sizing and wrapping.
        """
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        x_min, y_min, x_max, y_max = self.get_bbox_rect(bbox)
        box_width = x_max - x_min
        box_height = y_max - y_min
        
        # Add slight padding inside the box
        padding = 5
        avail_width = max(10, box_width - 2 * padding)
        avail_height = max(10, box_height - 2 * padding)
        
        # Dynamic font sizing loop
        font_size = 40  # Start large
        min_font_size = 10
        final_font = None
        final_lines = []
        
        # If font_path is missing or None, try to use default
        current_font_path = font_path if font_path else self.font_path
        
        while font_size >= min_font_size:
            try:
                if current_font_path:
                    font = ImageFont.truetype(current_font_path, font_size)
                else:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # Estimate chars per line
            # getlength works for single line
            avg_char_w = font.getlength("a")
            chars_per_line = int(avail_width / (avg_char_w * 0.9)) # 0.9 safety factor
            if chars_per_line < 1: chars_per_line = 1
            
            lines = textwrap.wrap(text, width=chars_per_line)
            
            # Calculate total text height
            # getbbox returns (left, top, right, bottom)
            line_heights = []
            max_line_w = 0
            for line in lines:
                bb = font.getbbox(line)
                h = bb[3] - bb[1]
                w = font.getlength(line)
                line_heights.append(h)
                if w > max_line_w: max_line_w = w
            
            total_text_h = sum(line_heights) + (len(lines) - 1) * 4  # 4px spacing
            
            if total_text_h <= avail_height and max_line_w <= avail_width:
                final_font = font
                final_lines = lines
                break
            
            font_size -= 2
            
        # If loop finishes without fit, use smallest
        if final_font is None:
            try:
                if current_font_path:
                    final_font = ImageFont.truetype(current_font_path, min_font_size)
                else:
                    final_font = ImageFont.load_default()
            except:
                final_font = ImageFont.load_default()
            final_lines = textwrap.wrap(text, width=max(10, int(avail_width/6)))

        # Draw lines centered
        total_h = sum([final_font.getbbox(l)[3] - final_font.getbbox(l)[1] for l in final_lines]) + (len(final_lines)-1)*4
        current_y = y_min + (box_height - total_h) // 2
        
        for line in final_lines:
            line_w = final_font.getlength(line)
            line_x = x_min + (box_width - line_w) // 2
            
            # Use black text by default
            draw.text((line_x, current_y), line, font=final_font, fill=(0, 0, 0))
            
            bb = final_font.getbbox(line)
            line_h = bb[3] - bb[1]
            current_y += line_h + 4
        
        # Convert back to BGR
        image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return image_bgr
    
    def process_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main processing function.
        """
        # Load image
        original_image = self.load_image(image_path)
        processed_image = original_image.copy()
        
        # Step 1: Detect text
        text_results = self.detect_text(original_image)
        
        if not text_results:
            return original_image, processed_image
        
        # Step 2: Cluster boxes
        clustered_results = self.cluster_boxes(text_results)
        
        if not clustered_results:
            return original_image, processed_image
        
        # Step 3: Process each cluster
        for bbox, text, confidence in clustered_results:
            if confidence < 0.4: # Lowered slightly
                continue
            
            # Translate
            translated_text = self.translate_text(text)
            
            # Detect color
            bubble_color = self.get_bubble_color(original_image, bbox)
            
            # Inpaint
            processed_image = self.inpaint_text(processed_image, bbox, bubble_color)
            
            # Render
            processed_image = self.render_text(processed_image, translated_text, bbox, self.font_path)
        
        return original_image, processed_image