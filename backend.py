import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Tuple, Optional


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
                raise FileNotFoundError("Comic font file not found. Please ensure comic_font.ttf exists in the root directory.")
        else:
            if not os.path.exists(font_path):
                raise FileNotFoundError(f"Font file not found: {font_path}")
            self.font_path = font_path
        
        # Thresholds for bubble detection
        self.max_bubble_area_ratio = 0.20  # 20% of page
        self.edge_margin_ratio = 0.05  # 5% margin from edges
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array (BGR format for OpenCV)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image
    
    def is_valid_speech_bubble(self, bbox: List[List[int]], image_shape: Tuple[int, int]) -> bool:
        """
        Determine if a text bounding box is a valid speech bubble.
        Filters out titles and large text boxes.
        
        Args:
            bbox: Bounding box coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            image_shape: (height, width) of the image
            
        Returns:
            True if valid speech bubble, False otherwise
        """
        height, width = image_shape[:2]
        
        # Calculate bounding box dimensions
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
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
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of tuples: (bbox, text, confidence)
        """
        results = self.reader.readtext(image)
        
        # Filter valid speech bubbles
        valid_results = []
        for (bbox, text, confidence) in results:
            if self.is_valid_speech_bubble(bbox, image.shape):
                valid_results.append((bbox, text.strip(), confidence))
        
        return valid_results
    
    def translate_text(self, text: str) -> str:
        """
        Translate text from English to Spanish (Spain) using OpenAI API.
        
        Args:
            text: English text to translate
            
        Returns:
            Translated Spanish text
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator. Translate the given English text to Spanish (Spain). Preserve the tone and style. Only return the translation, no explanations."
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
    
    def calculate_font_size(self, text: str, bbox: List[List[int]], 
                           font_path: str, max_size: int = 100) -> Tuple[ImageFont.FreeTypeFont, int]:
        """
        Calculate the optimal font size that fits within the bounding box.
        Uses iterative approach to find the largest font that fits.
        
        Args:
            text: Text to render
            bbox: Bounding box coordinates
            font_path: Path to font file
            max_size: Maximum font size to try
            
        Returns:
            Tuple of (font object, font size)
        """
        # Calculate bounding box dimensions
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        box_width = max(x_coords) - min(x_coords)
        box_height = max(y_coords) - min(y_coords)
        
        # Try font sizes from max_size down to 8
        for font_size in range(max_size, 7, -2):
            try:
                font = ImageFont.truetype(font_path, font_size)
                
                # Create a temporary image to measure text
                temp_img = Image.new('RGB', (box_width, box_height), 'white')
                draw = ImageDraw.Draw(temp_img)
                
                # Get text bounding box
                bbox_text = draw.textbbox((0, 0), text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                
                # Check if text fits (with some padding)
                padding = 10
                if text_width <= (box_width - padding) and text_height <= (box_height - padding):
                    return font, font_size
            except Exception as e:
                continue
        
        # Fallback to smallest size
        try:
            return ImageFont.truetype(font_path, 8), 8
        except:
            return ImageFont.load_default(), 12
    
    def inpaint_text(self, image: np.ndarray, bbox: List[List[int]]) -> np.ndarray:
        """
        Draw a white rectangle over the original text (inpainting).
        
        Args:
            image: Image as numpy array
            bbox: Bounding box coordinates
            
        Returns:
            Image with white rectangle drawn over text
        """
        image_copy = image.copy()
        
        # Get bounding box coordinates
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)
        
        # Draw white rectangle
        cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)
        
        return image_copy
    
    def render_text(self, image: np.ndarray, text: str, bbox: List[List[int]], 
                   font_path: str) -> np.ndarray:
        """
        Render translated text on the image using the comic font.
        
        Args:
            image: Image as numpy array (BGR)
            text: Text to render
            bbox: Bounding box coordinates
            font_path: Path to font file
            
        Returns:
            Image with text rendered
        """
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Calculate font size
        font, font_size = self.calculate_font_size(text, bbox, font_path)
        
        # Get bounding box center
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        box_width = x_max - x_min
        box_height = y_max - y_min
        
        # Get text bounding box to center it
        bbox_text = draw.textbbox((0, 0), text, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # Calculate position to center text in bounding box
        x_pos = x_min + (box_width - text_width) // 2
        y_pos = y_min + (box_height - text_height) // 2
        
        # Draw text (black color)
        draw.text((x_pos, y_pos), text, font=font, fill=(0, 0, 0))
        
        # Convert back to BGR for OpenCV
        image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return image_bgr
    
    def process_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main processing function: detect, translate, and render text.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (original_image, translated_image)
        """
        # Load image
        original_image = self.load_image(image_path)
        processed_image = original_image.copy()
        
        # Detect text
        text_results = self.detect_text(original_image)
        
        if not text_results:
            return original_image, processed_image
        
        # Process each text detection
        for bbox, text, confidence in text_results:
            # Skip if confidence is too low
            if confidence < 0.5:
                continue
            
            # Translate text
            translated_text = self.translate_text(text)
            
            # Inpaint original text
            processed_image = self.inpaint_text(processed_image, bbox)
            
            # Render translated text
            processed_image = self.render_text(processed_image, translated_text, bbox, self.font_path)
        
        return original_image, processed_image

