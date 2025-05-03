import cv2
import json
import pytesseract
import numpy as np
import urllib.request
import threading
import time
import os
import re
from flask import Flask, Response, jsonify, stream_with_context, request
from flask_cors import CORS
import warnings
import mediapipe as mp
import nltk
from nltk.corpus import stopwords
from symspellpy import SymSpell
from symspellpy.symspellpy import Verbosity
from gtts import gTTS
from io import BytesIO
import base64
from deep_translator import GoogleTranslator
from PIL import Image
from spellchecker import SpellChecker
import google.generativeai as genai
import pyttsx3

app = Flask(__name__)
CORS(app)


# URL for the mobile camera feed (adjust as per your setup)
#url = "http://192.168.31.124:8080/video"
#url = "http://100.66.85.238:8080/video"
url="http://100.76.187.91:8080/video"
# ESP32-CAM URL
ESP32_CAM_URL = "http://192.168.43.23/1024x768.mjpeg"

def frames(): 
    global ROI_X, ROI_Y, ocr_running
    try:
        stream = urllib.request.urlopen(ESP32_CAM_URL, timeout=10)
    except Exception as e:
        print(f"Error connecting to ESP32-CAM: {e}")
        return

    bytes_data = b''
    while True:
        try:
            bytes_data += stream.read(1024)
            a, b = bytes_data.find(b'\xff\xd8'), bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                frame = cv2.flip(frame, 1)
                
                # Convert frame to JPEG
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                
                # Yield the frame as part of a multipart response
                frame_data = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        except Exception as e:
            print(f"Error while streaming: {e}")
            break

# Route to stream video
@app.route('/frames')
def frames():
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame') 


def is_mjpeg_streaming(url: str, timeout: int = 10, debug: bool = True) -> bool:
    """
    Checks if MJPEG stream is active by reading a valid JPEG frame.

    Parameters:
        url (str): MJPEG stream URL (e.g. from ESP32-CAM).
        timeout (int): Timeout in seconds for the connection.
        debug (bool): If True, print debug information.

    Returns:
        bool: True if streaming is active, False otherwise.
    """
    try:
        if debug:
            print(f"[INFO] Attempting to connect to {url}")
        stream = urllib.request.urlopen(url, timeout=timeout)
        bytes_data = b''

        for i in range(50):  # Try reading 50 chunks
            bytes_data += stream.read(2048)

            start = bytes_data.find(b'\xff\xd8')  # JPEG start marker
            end = bytes_data.find(b'\xff\xd9')    # JPEG end marker

            if debug:
                print(f"[DEBUG] Iteration {i}: SOI at {start}, EOI at {end}, buffer size = {len(bytes_data)}")

            if start != -1 and end != -1 and end > start:
                jpg = bytes_data[start:end+2]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    if debug:
                        print("[INFO] Successfully decoded a frame. Stream is active.")
                    return True
                else:
                    if debug:
                        print("[WARNING] Frame could not be decoded. Saving buffer for inspection.")
                    with open("frame_debug.jpg", "wb") as f:
                        f.write(jpg)
                    return False

        if debug:
            print("[WARNING] No valid frame received from MJPEG stream.")
        return False

    except Exception as e:
        print(f"[ERROR] Could not connect to stream: {e}")
        return False


# Flask route to check the ESP32-CAM stream
@app.route("/ESPcamera", methods=["GET"])
def ESPcamera():
    try:
        # Use the is_mjpeg_streaming function to check if the stream is active
        if is_mjpeg_streaming(ESP32_CAM_URL):
            return jsonify({"status": "connected"}), 200
        else:
            return jsonify({"status": "not connected"}), 200

    except Exception as e:
        return jsonify({"status": "not connected", "error": str(e)}), 500


# Load SymSpell Dictionary
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "C:/Users//ADMIN//Downloads//1//END//backend//frequency_dictionary_en_82_765.txt"
if os.path.exists(dictionary_path):
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
else:
    raise FileNotFoundError(f"Dictionary file not found: {dictionary_path}")

# Set Tesseract Path
TESSERACT_PATH = r"C://Users//ADMIN//Downloads//1//END//backend//OCR//tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    raise FileNotFoundError(f"Tesseract not found at {TESSERACT_PATH}. Check the path.")


mp_hands = mp.solutions.hands
# ROI Settings
ROI_W, ROI_H = 100, 50
ROI_X, ROI_Y = 0, 0
latest_word = ""
ocr_running = False

# Download NLTK stopwords
nltk.download("stopwords", download_dir="NLPLIST")
nltk.data.path.append("NLPLIST")
nltk_stop_words = set(stopwords.words("english"))

def remove_stopwords(word):
    return word if word not in nltk_stop_words else ""

# Word Stability Tracking
word_count = {}
word_threshold = 3
lock = threading.Lock()

def correct_word(word,sym_spell):
    suggestions = sym_spell.lookup(word, verbosity=2, max_edit_distance=2)
    return suggestions[0].term if suggestions else word

def process_ocr(roi):
    global ocr_running, latest_word, word_count
    if roi.size > 0:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(roi_gray, output_type=pytesseract.Output.DICT)
        with lock:
            for text in data['text']:
                cleaned_word = re.sub(r'\W+', '', text.lower())
                if cleaned_word:
                    word_count[cleaned_word] = word_count.get(cleaned_word, 0) + 1
                    if word_count[cleaned_word] >= word_threshold:
                        meaningful_word = remove_stopwords(cleaned_word)
                        if meaningful_word:
                            latest_word = correct_word(meaningful_word,sym_spell)
                            word_count.clear()
                        break
    ocr_running = False

def generate_frames():
    global ROI_X, ROI_Y, ocr_running
    try:
        stream = urllib.request.urlopen(ESP32_CAM_URL, timeout=10)
    except Exception as e:
        print(f"Error connecting to ESP32-CAM: {e}")
        return

    bytes_data = b''
    with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            try:
                bytes_data += stream.read(1024)
                a, b = bytes_data.find(b'\xff\xd8'), bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    frame = cv2.flip(frame, 1)
                    image = cv2.resize(frame, (640, 480))
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image_rgb)
                    roi_visible = False
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            h, w, _ = image.shape
                            ROI_X = int(index_finger_tip.x * w) - ROI_W // 2
                            ROI_Y = int(index_finger_tip.y * h) - ROI_H - 20
                            ROI_X, ROI_Y = max(0, min(ROI_X, w - ROI_W)), max(0, min(ROI_Y, h - ROI_H))
                            roi_visible = True
                            cv2.circle(image, (int(index_finger_tip.x * w), int(index_finger_tip.y * h)), 10, (0, 255, 0), -1)
                    if roi_visible:
                        cv2.rectangle(image, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (0, 255, 0), 2)
                        roi = image[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
                        if not ocr_running:
                            ocr_running = True
                            threading.Thread(target=process_ocr, args=(roi,), daemon=True).start()
                    _, buffer = cv2.imencode('.jpg', image)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error processing video stream: {e}")
                break

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/word_stream')
def word_stream():
    def event_stream():
        global latest_word
        last_sent_word = None
        while True:
            time.sleep(1)
            with lock:
                if latest_word and latest_word != last_sent_word:
                    last_sent_word = latest_word
                    yield f"data: {json.dumps({'word': latest_word})}\n\n"
    return Response(event_stream(), content_type="text/event-stream")

# Global variables
best_frame = None
best_sharpness = 0
best_text_confidence = 0

stream_active = False

# Function to extract text and confidence score
def extract_text_with_confidence(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    valid_confidences = [int(conf) for conf in d['conf'] if str(conf).isdigit()]
    confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0
    text = pytesseract.image_to_string(gray)
    return text, confidence

# Function to calculate sharpness
def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

# Function to process video stream
def process_video(camera_url):
    global best_frame, best_sharpness, best_text_confidence, stream_active
    stream_active = True
    extracted_text = ""
    try:
        stream = urllib.request.urlopen(camera_url)
        bytes_data = b''
        start_time = time.time()

        while stream_active and time.time() - start_time < 5:
            try:
                bytes_data += stream.read(1024)
                a, b = bytes_data.find(b'\xff\xd8'), bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        continue

                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    frame = cv2.flip(frame, 1)
                    frame = cv2.resize(frame, (640, 480))

                    # Ensure OpenCV event handling continues
                    cv2.waitKey(1)

                    # Check for best frame
                    sharpness = calculate_sharpness(frame)
                    text, confidence = extract_text_with_confidence(Image.fromarray(frame))

                    if sharpness > best_sharpness and confidence > best_text_confidence:
                        best_sharpness = sharpness
                        best_text_confidence = confidence
                        best_frame = frame
                        extracted_text = text
                    
                    # Stop the loop if 'q' is pressed (even though no window is shown)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stream_active = False
                        break
            except Exception as e:
                print(f"Error processing frame: {e}")
    except Exception as e:
        print(f"Error processing camera stream: {e}")
    finally:
        stream_active = False
    return extracted_text

# Flask API endpoint
@app.route('/scan', methods=['POST'])
def scan():
    data = request.json
    command = data.get('command', '').strip().lower()
    
    if command == 'start':
        extracted_text = process_video(ESP32_CAM_URL)
        return jsonify({"extracted_text": extracted_text})
    elif command == 'stop':
        global stream_active
        stream_active = False
        return jsonify({"message": "Scanning stopped"})
    else:
        return jsonify({"error": "Invalid command"}), 400


# Supported languages with their language codes
LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ta": "Tamil",
    "zh-cn": "Chinese (Simplified)",
    "hi": "Hindi",
    "ar": "Arabic",
    "pt": "Portuguese",
    "bn": "Bengali",
    "ru": "Russian",
    "ja": "Japanese",
    "pa": "Punjabi",
}

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="SymbolDatabase.GetPrototype() is deprecated",
)

# Create a directory to save debug images
debug_images_dir = "C://Users//ADMIN//Downloads//1//END//debug_images"
os.makedirs(debug_images_dir, exist_ok=True)


def text_to_speech(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        print(f"Error during text to speech: {e}")
        return None


# Function to draw bounding boxes around detected text regions, append detected words to a list, and count occurrences
def draw_bounding_boxes(image, data, detected_words, word_counts):
    n_boxes = len(data["level"])
    for i in range(n_boxes):
        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        # Draw a rectangle around the detected text
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add text annotation above the rectangle
        text = data["text"][i]
        cv2.putText(
            image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

        # Clean the detected word for counting
        cleaned_word = re.sub(r"\W+", "", text.strip()).lower()

        # Count occurrences of cleaned word within ROI
        if cleaned_word in word_counts:
            word_counts[cleaned_word] += 1
        else:
            word_counts[cleaned_word] = 1

        # Append the cleaned word to detected words
        if cleaned_word:
            detected_words.append(cleaned_word)


# Function to filter out non-empty strings and clean up detected words
def clean_detected_words(detected_words):
    cleaned_words = []
    for word in detected_words:
        # Remove extra spaces and unwanted characters using regex
        cleaned_word = re.sub(r"\W+", "", word.strip())
        if cleaned_word:
            cleaned_words.append(
                cleaned_word.lower()
            )  # Convert to lowercase for consistency
    return cleaned_words


# Function to correct a word using SymSpell
def correct_word(word, sym_spell):
    suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term
    return None  # Return None if no correction found


# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Load a frequency dictionary
dictionary_path = (
    r"C:/Users//ADMIN//Downloads//1//END//backend//frequency_dictionary_en_82_765.txt"  # Path to your dictionary file
)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


@app.route("/checkcamera", methods=["GET"])
def check_camera():
    try:
        # Attempt to read a frame from the camera
        cap = cv2.VideoCapture(url)
        ret, frame = cap.read()

        if not ret:
            return jsonify({"status": "not connected"}), 200

        # If the image is successfully decoded, the camera is connected
        return jsonify({"status": "connected"}), 200

    except Exception as e:
        return jsonify({"status": "not connected", "error": str(e)}), 500

    finally:
        if "cap" in locals() and cap.isOpened():
            cap.release()


@app.route("/speech", methods=["POST"])
def translate_text():
    data = request.get_json()
    
    if app.debug:
        print("Raw Request Data:", request.data)
        print("Parsed JSON:", data)

    text_input = data.get("Text") or data.get("text")
    source_lang = data.get("SourceLanguage") or data.get("sourceLanguage")
    target_lang = data.get("TargetLanguage") or data.get("targetLanguage")

    if not text_input or not source_lang or not target_lang:
        return jsonify({"error": "Missing required fields"}), 400

    if source_lang not in LANGUAGES or target_lang not in LANGUAGES:
        return jsonify({"error": "Unsupported language"}), 400

    try:
        translated_text = GoogleTranslator(
            source=source_lang, target=target_lang
        ).translate(text_input)

        source_audio_fp = text_to_speech(text_input, source_lang)
        target_audio_fp = text_to_speech(translated_text, target_lang)

        return jsonify({
            "text": text_input,
            "translated_text": translated_text,
            "source_audio": (
                base64.b64encode(source_audio_fp.read()).decode("utf-8")
                if source_audio_fp else None
            ),
            "target_audio": (
                base64.b64encode(target_audio_fp.read()).decode("utf-8")
                if target_audio_fp else None
            ),
        })

    except Exception as e:
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500

@app.route("/extract", methods=["GET"])
def extract_text():
    detected_words = []  # List to store detected words
    word_counts = {}  # Dictionary to store word counts
    corrected_word_count = 0

    roi_w, roi_h = (
        150,
        95,
    )  # Adjusted ROI dimensions based on measurement (width reduced)
    roi_x, roi_y = 0, 0  # Initial position of ROI

    try:
        cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            return jsonify({"error": "Failed to open video stream"}), 500

        with mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            while cap.isOpened():
                ret, image = cap.read()

                if not ret:
                    return (
                        jsonify({"error": "Failed to read frame from video stream"}),
                        500,
                    )

                # Process the image and detect hands
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                # Initialize ROI visibility flag
                roi_visible = False

                # Highlight the index finger tip and move the ROI
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get the index finger tip coordinates
                        index_finger_tip = hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_TIP
                        ]
                        h, w, _ = image.shape
                        index_finger_tip_x = int(index_finger_tip.x * w)
                        index_finger_tip_y = int(index_finger_tip.y * h)

                        # Update ROI position above the index finger tip
                        roi_x = index_finger_tip_x - roi_w // 2
                        roi_y = (
                            index_finger_tip_y - roi_h - 20
                        )  # Adjust the offset as needed
                        roi_visible = True

                        # Ensure ROI is within image bounds
                        roi_x = max(0, min(roi_x, w - roi_w))
                        roi_y = max(0, min(roi_y, h - roi_h))

                        # Draw a circle at the index finger tip
                        cv2.circle(
                            image,
                            (index_finger_tip_x, index_finger_tip_y),
                            10,
                            (0, 255, 0),
                            -1,
                        )

                if roi_visible:
                    # Draw the adjusted size ROI rectangle if the index finger is detected
                    cv2.rectangle(
                        image,
                        (roi_x, roi_y),
                        (roi_x + roi_w, roi_y + roi_h),
                        (0, 255, 0),
                        2,
                    )

                    # Convert ROI to grayscale and perform OCR using Tesseract
                    roi = image[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
                    if roi.size > 0:  # Check if ROI is valid
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        data = pytesseract.image_to_data(
                            roi_gray, output_type=pytesseract.Output.DICT
                        )

                        # Draw bounding boxes around detected text in ROI and append words to the list
                        draw_bounding_boxes(image, data, detected_words, word_counts)

                # Clean up detected words and process them
                cleaned_words = clean_detected_words(detected_words)

                # Process detected words and correct them
                response_data = []
                for word in cleaned_words:
                    corrected_word = correct_word(word, sym_spell)
                    if corrected_word and corrected_word.lower() not in nltk_stop_words:
                        corrected_word = corrected_word.lower()
                        count = word_counts.get(corrected_word, 0)
                        if count >= 3:
                            response_data.append(
                                {"word": corrected_word, "count": count}
                            )
                            corrected_word_count += 1
                            detected_words.remove(
                                word
                            )  # Remove the printed word from detected_words

                # Clear detected words after processing
                detected_words.clear()

                # If corrected words are found, return the response and stop processing
                if corrected_word_count > 0:
                    debug_image_path = os.path.join(debug_images_dir, "debug_image.jpg")
                    cv2.imwrite(debug_image_path, image)
                    return jsonify(
                        {
                            "corrected_words": response_data,
                            "corrected_word_count": corrected_word_count,
                            "debug_image": debug_image_path,
                        }
                    )

        # If no words are detected and corrected, return an appropriate message
        return jsonify({"message": "No words detected and corrected"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if "cap" in locals() and cap.isOpened():
            cap.release()



# Global variables
scanning = False
exit_program = False
best_frames = []
highest_conf_text_full = ""
highest_conf_score_full = 0
spell = SpellChecker()

# Function to extract text and confidence
def extract_text_with_confidence(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    text = " ".join(data['text']).strip()
    confidences = [conf for conf in data['conf'] if isinstance(conf, (int, float))]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    return text, avg_confidence

# Function to clean and correct text
def clean_text(extracted_text):
    lines = extracted_text.split('\n')
    corrected_text = []

    def correct_line(line):
        line = re.sub(r'[^a-zA-Z0-9.,\'\" -]', '', line)
        line = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', line)
        words = line.split()
        return ' '.join([spell.correction(word) or word for word in words])

    threads = []
    for line in lines:
        thread = threading.Thread(target=lambda l: corrected_text.append(correct_line(l)), args=(line,))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    return '\n'.join(corrected_text).strip()

# Function to process scanned pages
def process_scanned_pages():
    global highest_conf_text_full, highest_conf_score_full
    
    for frame in best_frames:
        full_text, full_confidence = extract_text_with_confidence(Image.fromarray(frame))
        full_text = clean_text(full_text)
        if full_confidence > highest_conf_score_full:
            highest_conf_score_full = full_confidence
            highest_conf_text_full = full_text

# Function to capture best frames from webcam
def process_page_from_webcam(url):
    global scanning, exit_program, best_frames
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        return "Error: Unable to access the camera. Check the URL."
    
    scanning = True
    start_time = time.time()
    sharpness_threshold = 100
    
    try:
        while time.time() - start_time < 6:
            ret, frame = cap.read()
            if not ret:
                return "Error: Unable to read frame from the camera."
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            
            if sharpness > sharpness_threshold:
                best_frames.append(frame)
                sharpness_threshold = sharpness  # Adjust threshold dynamically
    
    finally:
        cap.release()
        if best_frames:
            process_scanned_pages()
        else:
            return "No valid frame captured."
        exit_program = True
    
    return highest_conf_text_full if highest_conf_text_full else "No valid text extracted."

# Flask API endpoint
@app.route('/Full_scan', methods=['POST'])
def Full_scan():
    data = request.json
    command = data.get('command', '').strip().lower()
    
    if command == 'start':
        extracted_text = process_page_from_webcam(url)
        return jsonify({"extracted_text": extracted_text})
    else:
        return jsonify({"error": "Invalid command"}), 400


genai.configure(api_key="pls Provide API Key")

class ContextChatbot:
    def __init__(self, context=""):
        """Initialize chatbot with optional context and TTS."""
        self.model = genai.GenerativeModel(model_name="gemini-2.0-pro-exp-02-05")
        self.context = context
        self.last_response = context
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 150)
        self.mode = "text"

    def update_context(self, new_context):
        """Update the chatbot context."""
        self.context = new_context
        self.last_response = new_context

    def speak(self, text):
        """Convert text to speech."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def get_response(self, user_input):
        """Generate a structured response based on user input."""
        prompt = f"Context:\n{self.context}\n\nUser's request:\n{user_input}\n\nProvide a well-structured response in paragraph form:"
        response = self.model.generate_content(prompt)
        
        if response and hasattr(response, "text"):
            return response.text.replace("*", "").strip()
        return "I'm sorry, I couldn't generate a response."

# Instantiate chatbot with empty default context
chatbot = ContextChatbot()

@app.route("/set_context", methods=["POST"])
def set_context():
    """API endpoint to set or update the story context."""
    data = request.get_json()
    new_context = data.get("context", "").strip()
    
    if not new_context:
        return jsonify({"url": "/set_context", "request": data, "response": "No context provided."})
    
    chatbot.update_context(new_context)
    return jsonify({"url": "/set_context", "request": data, "response": "Context updated successfully."})

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests."""
    data = request.get_json()
    user_input = data.get("message", "").strip().lower()
    
    if not user_input:
        return jsonify({"url": "/chat", "request": data, "response": "Please provide a valid input."})
    
    if user_input in ("exit", "stop"):
        return jsonify({"url": "/chat", "request": data, "response": "Conversation ended. Goodbye!"})
    
    chatbot_reply = chatbot.get_response(user_input)
    chatbot.last_response = chatbot_reply
    
    return jsonify({"url": "/chat", "request": data, "response": chatbot_reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
