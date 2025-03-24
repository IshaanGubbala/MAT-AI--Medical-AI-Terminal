import os
import numpy as np
import sounddevice as sd
import torch
import threading
import queue
import datetime
from collections import deque
import time
import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F

# Third party imports
import whisper  # OpenAI's official, free, open-source Whisper package
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QTextEdit, QPushButton, QLabel, QFrame, QHBoxLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont
from duckduckgo_search import DDGS  # For web search

# Helper function: update_console clears the terminal and prints a message (disappearing text effect)
def update_console(message):
    os.system('cls' if os.name=='nt' else 'clear')
    print(message)

# Print CUDA diagnostics on startup
print("\n=== CUDA Diagnostics ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("WARNING: CUDA is not available. Install PyTorch with CUDA support:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
print("=====================\n")

if torch.cuda.is_available():
    torch.cuda.init()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

#############################
# Signal and UI Classes     #
#############################

class SignalHandler(QObject):
    transcript_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    medical_signal = pyqtSignal(str)

class TranscriptionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.signal_handler = SignalHandler()
        self.medical_info = []
        self.conversation_history = deque(maxlen=10)
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Medical Speech Assistant')
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f2f5; }
            QLabel { color: #1a1a1a; font-weight: 500; }
            QTextEdit {
                border: none; border-radius: 10px; padding: 15px;
                background-color: white; color: #333333;
                selection-background-color: #007AFF; selection-color: white;
            }
            QPushButton {
                border: none; border-radius: 20px; padding: 12px 25px;
                font-weight: bold; font-size: 14px;
            }
        """)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        
        # Header Section
        header_frame = QFrame()
        header_frame.setStyleSheet("QFrame { background-color: white; border-radius: 15px; padding: 10px; }")
        header_layout = QHBoxLayout(header_frame)
        self.status_label = QLabel('Initializing...')
        self.status_label.setFont(QFont('Segoe UI', 12))
        self.status_label.setStyleSheet("color: #007AFF; padding: 10px;")
        header_layout.addWidget(self.status_label)
        main_layout.addWidget(header_frame)
        
        # Transcription Section
        transcript_frame = QFrame()
        transcript_frame.setStyleSheet("QFrame { background-color: white; border-radius: 15px; padding: 15px; }")
        transcript_layout = QVBoxLayout(transcript_frame)
        transcript_label = QLabel('Live Transcription')
        transcript_label.setFont(QFont('Segoe UI', 14, QFont.Weight.Bold))
        transcript_layout.addWidget(transcript_label)
        self.transcript_display = QTextEdit()
        self.transcript_display.setReadOnly(True)
        self.transcript_display.setFont(QFont('Segoe UI', 12))
        self.transcript_display.setStyleSheet("background-color: #f8f9fa; min-height: 150px;")
        transcript_layout.addWidget(self.transcript_display)
        main_layout.addWidget(transcript_frame)
        
        # Medical Analysis Section
        medical_frame = QFrame()
        medical_frame.setStyleSheet("QFrame { background-color: white; border-radius: 15px; padding: 15px; }")
        medical_layout = QVBoxLayout(medical_frame)
        medical_label = QLabel('Medical Analysis')
        medical_label.setFont(QFont('Segoe UI', 14, QFont.Weight.Bold))
        medical_layout.addWidget(medical_label)
        self.medical_display = QTextEdit()
        self.medical_display.setReadOnly(True)
        self.medical_display.setFont(QFont('Segoe UI', 12))
        self.medical_display.setStyleSheet("background-color: #f8f9fa; min-height: 200px;")
        medical_layout.addWidget(self.medical_display)
        main_layout.addWidget(medical_frame)
        
        # Control Button
        button_container = QFrame()
        button_container.setStyleSheet("background: transparent;")
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        self.control_button = QPushButton('Stop Recording')
        self.control_button.setFixedWidth(200)
        self.control_button.setStyleSheet("""
            QPushButton { background-color: #dc3545; color: white; }
            QPushButton:hover { background-color: #c82333; }
        """)
        self.control_button.clicked.connect(self.toggle_recording)
        button_layout.addWidget(self.control_button, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(button_container)
        
        self.signal_handler.transcript_signal.connect(self.update_transcript)
        self.signal_handler.status_signal.connect(self.update_status)
        self.signal_handler.medical_signal.connect(self.update_medical_info)
        self.recording = True
        
    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            self.control_button.setText('Stop Recording')
            self.control_button.setStyleSheet("""
                QPushButton { background-color: #dc3545; color: white; border: none;
                border-radius: 20px; padding: 12px 25px; font-weight: bold; font-size: 14px; }
                QPushButton:hover { background-color: #c82333; }
            """)
        else:
            self.control_button.setText('Start Recording')
            self.control_button.setStyleSheet("""
                QPushButton { background-color: #28a745; color: white; border: none;
                border-radius: 20px; padding: 12px 25px; font-weight: bold; font-size: 14px; }
                QPushButton:hover { background-color: #218838; }
            """)
    
    def update_transcript(self, text):
        cursor = self.transcript_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertHtml(f'<p style="margin: 0; padding: 5px 0;">{text}</p>')
        self.transcript_display.setTextCursor(cursor)
        self.transcript_display.ensureCursorVisible()
    
    def update_status(self, status):
        self.status_label.setText(status)

    def update_medical_info(self, text):
        cursor = self.medical_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        formatted_text = text.replace('\n', '<br>')
        formatted_text = (
            '<div style="background-color: #f8f9fa; padding: 10px; border-left: 4px solid #007AFF; margin: 10px 0;">'
            f'{formatted_text}</div>'
        )
        cursor.insertHtml(formatted_text)
        self.medical_display.setTextCursor(cursor)
        self.medical_display.ensureCursorVisible()

##############################
# Medical Analysis Functions #
##############################

class MedicalAnalyzer:
    def __init__(self):
        self.load_mayo_data()
        
    def load_mayo_data(self):
        try:
            self.mayo_data = pd.read_csv('mayo_diseases.csv')
            print(f"Loaded {len(self.mayo_data)} Mayo Clinic conditions")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.mayo_data['combined_text'] = self.mayo_data.apply(
                lambda x: f"{x['Disease Name']} {x['Overview']} {x['Symptoms']}",
                axis=1
            )
            self.mayo_embeddings = self.embedder.encode(
                self.mayo_data['combined_text'].tolist(), convert_to_tensor=True
            )
        except Exception as e:
            print(f"Error loading Mayo data: {e}")
            self.mayo_data = None

    def find_relevant_mayo_info(self, full_text: str, top_k: int = 3) -> list:
        if self.mayo_data is None:
            return []
        query_embedding = self.embedder.encode(full_text, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, self.mayo_embeddings)[0]
        top_results = torch.topk(cos_scores, k=len(self.mayo_data))
        disease_dict = {}
        for score, idx in zip(top_results[0], top_results[1]):
            i = idx.item()
            condition = self.mayo_data.iloc[i]
            disease_name = condition['Disease Name']
            score_val = float(score)
            if disease_name in disease_dict:
                if score_val > disease_dict[disease_name]['relevance_score']:
                    disease_dict[disease_name] = {
                        'Disease Name': disease_name,
                        'Overview': condition['Overview'],
                        'Symptoms': condition['Symptoms'],
                        'URLs': condition.get('URLs', condition.get('urls', '')),
                        'relevance_score': score_val
                    }
            else:
                disease_dict[disease_name] = {
                    'Disease Name': disease_name,
                    'Overview': condition['Overview'],
                    'Symptoms': condition['Symptoms'],
                    'URLs': condition.get('URLs', condition.get('urls', '')),
                    'relevance_score': score_val
                }
        sorted_conditions = sorted(disease_dict.values(), key=lambda x: x['relevance_score'], reverse=True)
        top_conditions = sorted_conditions[:top_k]
        total_score = sum(item['relevance_score'] for item in top_conditions)
        for item in top_conditions:
            item['relevance_score'] = (item['relevance_score'] / total_score * 100) if total_score > 0 else 0
        return top_conditions

def search_medications(query, max_results=3):
    results = []
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
    return results if results is not None else []

###################################
# Processing Functions (Threads)  #
###################################

def process_transcription(audio_queue, text_queue, whisper_model, signal_handler):
    while True:
        try:
            audio_segment = audio_queue.get()
            if audio_segment is None:
                text_queue.put(None)
                break
            result = whisper_model.transcribe(audio_segment, language="en")
            transcript = result.get("text", "")
            if transcript.strip():
                print("[Monitor] Transcribed:", transcript)
                signal_handler.transcript_signal.emit(f"You: {transcript}")
                text_queue.put(transcript)
        except Exception as e:
            print(f"Error during transcription loop: {e}")

def process_medical_analysis(text_queue, medical_model, medical_tokenizer, device, signal_handler, window):
    analyzer = MedicalAnalyzer()
    while True:
        try:
            transcript = text_queue.get()
            if transcript is None:
                break
            window.conversation_history.append(transcript)
            full_context = " ".join(list(window.conversation_history))
            
            # --- Step 1: Mayo Clinic Analysis ---
            t0 = time.time()
            mayo_info = analyzer.find_relevant_mayo_info(full_context)
            t1 = time.time()
            mayo_time = t1 - t0
            update_console(f"[Monitor] Mayo data processing completed in {mayo_time:.2f} seconds.")
            
            mayo_section = "Potential Diseases (Mayo Clinic Data):\n"
            for idx, condition in enumerate(mayo_info, 1):
                if condition['relevance_score'] > 0.3:
                    mayo_section += (f"Condition {idx}: Disease: {condition['Disease Name']}, "
                                      f"Overview: {condition['Overview']}, "
                                      f"Symptoms: {condition['Symptoms']}, "
                                      f"Confidence: {condition['relevance_score']:.2f}%\n")
            
            # --- Step 2: Web Search for Medications ---
            t2 = time.time()
            med_results = search_medications(full_context)
            t3 = time.time()
            web_time = t3 - t2
            update_console(f"[Monitor] Web search for medications completed in {web_time:.2f} seconds.")
            
            med_section = "Medication Recommendations (Web Search Results):\n"
            for idx, result in enumerate(med_results, 1):
                med_section += (f"Result {idx}: Title: {result.get('title', 'N/A')}, "
                                f"URL: {result.get('href', 'N/A')}, "
                                f"Snippet: {result.get('body', 'N/A')}\n")
            
            # --- Step 3: Build Structured Prompt Template ---
            template = (
                "You are a medical professional. Please provide a structured analysis using the template below:\n\n"
                "Template:\n"
                "1. Potential Diseases: List the potential diseases (from Mayo Clinic data) with their confidence percentages (ensure they add up to 100%) and a brief overview.\n"
                "2. Reasoning: Explain concisely the reasoning behind the disease selection without repetition.\n"
                "3. Possible Treatments: Based on the web search medication recommendations provided, list potential treatments or medications.\n\n"
                "Conversation Context:\n" + full_context + "\n\n"
                "Mayo Clinic Reference Data:\n" + mayo_section + "\n\n"
                "Web Search Medication Data:\n" + med_section + "\n\n"
                "Do not repeat any information. Provide a concise, well-structured final analysis."
            )
            
            t4 = time.time()
            with torch.no_grad():
                inputs = medical_tokenizer(
                    template,
                    return_tensors="pt",
                    max_length=768,
                    truncation=True,
                    padding=True
                ).to(device)
                outputs = medical_model.generate(
                    **inputs,
                    max_length=192,
                    min_length=100,
                    num_beams=5,
                    early_stopping=True,
                    repetition_penalty=1.5
                )
                final_analysis = medical_tokenizer.decode(outputs[0], skip_special_tokens=True)
            t5 = time.time()
            gen_time = t5 - t4
            update_console(f"[Monitor] FLAN generation completed in {gen_time:.2f} seconds.")
            
            formatted_analysis = f'''
                <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; 
                     box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="color: #2c3e50; font-size: 14px; margin-bottom: 15px;">
                        <strong>Structured Analysis</strong> ({datetime.datetime.now().strftime("%H:%M:%S")})
                    </div>
                    <div style="color: #34495e; font-size: 15px; line-height: 1.6;">
                        {final_analysis}
                    </div>
                </div>
            '''
            signal_handler.medical_signal.emit(formatted_analysis)
        except Exception as e:
            print(f"Error in medical analysis loop: {e}")
            torch.cuda.empty_cache()

#########################
# Main Application Code #
#########################

def main():
    app = QApplication([])
    window = TranscriptionUI()
    
    sample_rate = 16000
    chunk_duration = 0.45
    silence_threshold = 0.0005
    min_silence_duration = 0.75
    
    audio_queue = queue.Queue()
    text_queue = queue.Queue()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("GPU not available, using CPU")
    
    signal_handler = window.signal_handler
    signal_handler.status_signal.emit("Loading models...")
    
    # Load Whisper model (using the large variant)
    try:
        whisper_model = whisper.load_model("large")
        print("Whisper model loaded successfully")
    except Exception as e:
        print(f"Error loading Whisper: {e}")
        return

    # Load the medical analysis model using google/flan-t5-xl with 8-bit quantization for speed optimization.
    try:
        print("Loading medical analysis model...")
        model_name = "google/flan-t5-xl"
        medical_tokenizer = AutoTokenizer.from_pretrained(model_name)
        medical_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            load_in_8bit=True,  # Use 8-bit quantization to reduce memory and speed up inference
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        if torch.cuda.is_available():
            # If using 8-bit quantization, you may not need to call .half() explicitly.
            pass
        medical_model.eval()
        torch.cuda.empty_cache()
        print("Medical analysis model loaded successfully")
    except Exception as e:
        print(f"Error loading medical model: {e}")
        medical_model = None
        medical_tokenizer = None
    
    transcription_thread = threading.Thread(
        target=process_transcription,
        args=(audio_queue, text_queue, whisper_model, signal_handler),
        daemon=True
    )
    transcription_thread.start()
    
    if medical_model is not None:
        medical_thread = threading.Thread(
            target=process_medical_analysis,
            args=(text_queue, medical_model, medical_tokenizer, device, signal_handler, window),
            daemon=True
        )
        medical_thread.start()
    
    signal_handler.status_signal.emit("Listening...")
    
    speech_buffer = []
    is_speaking = False
    silence_counter = 0
    max_silence_frames = int(min_silence_duration / chunk_duration)
    
    def audio_callback(indata, frames, time_info, status):
        if not window.recording:
            return
        nonlocal is_speaking, speech_buffer, silence_counter
        energy = np.sqrt(np.mean(np.square(indata)))
        if energy > silence_threshold:
            silence_counter = 0
            if not is_speaking:
                is_speaking = True
            speech_buffer.extend(indata.flatten())
        else:
            if is_speaking:
                silence_counter += 1
                speech_buffer.extend(indata.flatten())
                if silence_counter >= max_silence_frames:
                    if len(speech_buffer) > sample_rate * 0.5:
                        audio_queue.put(np.array(speech_buffer))
                    is_speaking = False
                    speech_buffer = []
                    silence_counter = 0
    
    window.show()
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
        blocksize=int(chunk_duration * sample_rate),
        callback=audio_callback
    ):
        app.exec()
    
    audio_queue.put(None)
    transcription_thread.join()
    if medical_model is not None:
        medical_thread.join()

def create_reference_files():
    medical_data = {
        "conditions.json": {
            "migraine": {
                "description": "Severe, recurring headache with additional symptoms",
                "common_symptoms": ["throbbing pain", "nausea", "light sensitivity"],
                "common_treatments": ["sumatriptan", "preventive medications"],
                "when_to_see_doctor": "If migraines are severe or frequent"
            }
        },
        "medications.json": {
            "acetaminophen": {
                "uses": ["pain relief", "fever reduction"],
                "dosage": "325-650mg every 4-6 hours",
                "warnings": ["Don't exceed 4000mg per day", "Avoid alcohol"],
                "interactions": ["liver medications"]
            }
        },
        "symptoms.json": {
            "chest_pain": {
                "description": "Pain or discomfort in the chest area",
                "associated_conditions": ["heart attack", "angina", "anxiety"],
                "severity_indicators": ["spreading to arm/jaw", "difficulty breathing"],
                "immediate_action": "Seek emergency care if severe or with other symptoms"
            }
        }
    }
    data_path = Path("medical_data")
    data_path.mkdir(exist_ok=True)
    for filename, data in medical_data.items():
        with open(data_path / filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
