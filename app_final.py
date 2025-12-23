"""
🤟 SIGN LANGUAGE DETECTOR PRO
Real-Time ASL Detection with AI Translation + Mouse Control
"""

import streamlit as st
import cv2
import numpy as np
import pickle
import time
from datetime import datetime
from collections import deque
from mediapipe import Image as MPImage, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Check optional imports
try:
    from translator import Translator
    import translation_config as trans_config
    TRANSLATION_AVAILABLE = True
except:
    TRANSLATION_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except:
    TTS_AVAILABLE = False

try:
    import autopy
    MOUSE_CONTROL_AVAILABLE = True
except:
    MOUSE_CONTROL_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Sign Language Detector Pro",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 SIGN LANGUAGE DETECTOR PRO")
st.write("Real-Time ASL Detection with AI Translation + Mouse Control")

# Camera control at the top
st.markdown("---")
camera_col1, camera_col2 = st.columns([3, 1])

with camera_col1:
    st.info("💡 **Quick Start:** 1️⃣ Adjust settings in sidebar → 2️⃣ Click START CAMERA → 3️⃣ Show ASL signs")
    st.info("✋ **Important:** Use your LEFT HAND for accurate sign detection")

with camera_col2:
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    
    camera_button = st.button("📹 START CAMERA" if not st.session_state.camera_running else "⏹️ STOP CAMERA", 
                 type="primary" if not st.session_state.camera_running else "secondary",
                 use_container_width=True,
                 key="camera_toggle_btn")
    
    if camera_button:
        st.session_state.camera_running = not st.session_state.camera_running
        # Release camera when stopping
        if not st.session_state.camera_running and 'cap' in st.session_state:
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
        st.rerun()

st.markdown("---")

# Initialize session state
if 'initialized' not in st.session_state:
    with st.spinner("🔄 Loading models... Please wait..."):
        st.session_state.initialized = False
        st.session_state.sentence = ""
        st.session_state.translated_sentence = ""
        st.session_state.current_language = "Spanish"
        st.session_state.mode = "Detection"
        
        # Detection metrics
        st.session_state.total_detections = 0
        st.session_state.successful_detections = 0
        
        # Timing
        st.session_state.frame_times = deque(maxlen=30)
        st.session_state.detection_times = deque(maxlen=30)
        
        # Detection state
        st.session_state.last_detected_char = None
        st.session_state.char_stable_since = None
        st.session_state.detection_cooldown = 1.5
        st.session_state.last_detection_time = 0
        st.session_state.auto_add = True
        st.session_state.confidence_history = deque(maxlen=100)
        
        # Current detection
        st.session_state.current_detected_letter = None
        st.session_state.current_confidence = 0.0
        
        # Camera object
        st.session_state.cap = None
        
        # Mouse control state
        st.session_state.prev_x = 0
        st.session_state.prev_y = 0
        st.session_state.click_cooldown = 0
        st.session_state.is_dragging = False
        
        try:
            # Load model
            model_dict = pickle.load(open('./model.p', 'rb'))
            st.session_state.model = model_dict['model']
            st.session_state.labels_dict = {i: chr(65 + i) for i in range(26)}
            
            # Initialize MediaPipe
            base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.5
            )
            st.session_state.detector = vision.HandLandmarker.create_from_options(options)
            
            # Initialize translator
            if TRANSLATION_AVAILABLE:
                st.session_state.translator = Translator()
                st.session_state.available_languages = trans_config.list_available_languages()
            else:
                st.session_state.available_languages = ['Spanish']
            
            # Initialize TTS
            if TTS_AVAILABLE:
                st.session_state.tts_engine = pyttsx3.init()
                st.session_state.tts_engine.setProperty('rate', 150)
            
            st.session_state.initialized = True
            st.success("✅ All systems loaded!")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("Check that model.p and hand_landmarker.task exist")
            st.stop()

# Helper functions
def detect_and_predict(frame):
    """Detect hand and predict character"""
    start_time = time.time()
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb_frame)
    detection_result = st.session_state.detector.detect(mp_image)
    
    detection_time = (time.time() - start_time) * 1000
    st.session_state.detection_times.append(detection_time)
    
    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        
        # Extract features
        data_aux = []
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        min_x, min_y = min(x_coords), min(y_coords)
        
        for lm in hand_landmarks:
            data_aux.append(lm.x - min_x)
            data_aux.append(lm.y - min_y)
        
        # Predict
        try:
            prediction = st.session_state.model.predict([np.asarray(data_aux)])
            probabilities = st.session_state.model.predict_proba([np.asarray(data_aux)])[0]
            confidence = max(probabilities)
            predicted_char = st.session_state.labels_dict[int(prediction[0])]
            
            st.session_state.confidence_history.append(confidence)
            return predicted_char, confidence, hand_landmarks
        except:
            return None, 0.0, hand_landmarks
    
    return None, 0.0, None

def draw_hand_with_skeleton(frame, hand_landmarks):
    """Draw hand skeleton, rectangle, and landmarks"""
    if hand_landmarks is None:
        return frame
    
    H, W = frame.shape[:2]
    
    # Get landmark points
    points = [(int(lm.x * W), int(lm.y * H)) for lm in hand_landmarks]
    
    # Draw GREEN RECTANGLE around hand
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, x_max = max(0, min(x_coords) - 20), min(W, max(x_coords) + 20)
    y_min, y_max = max(0, min(y_coords) - 20), min(H, max(y_coords) + 20)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)
    
    # Draw GREEN skeleton lines (25 connections)
    connections = [
        (0,1), (1,2), (2,3), (3,4),  # Thumb
        (0,5), (5,6), (6,7), (7,8),  # Index
        (0,9), (9,10), (10,11), (11,12),  # Middle
        (0,13), (13,14), (14,15), (15,16),  # Ring
        (0,17), (17,18), (18,19), (19,20),  # Pinky
        (5,9), (9,13), (13,17)  # Palm connections
    ]
    
    for start_idx, end_idx in connections:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 5)
    
    # Draw LARGE landmark circles
    for point in points:
        cv2.circle(frame, point, 10, (0, 255, 255), -1)  # Yellow outer
        cv2.circle(frame, point, 5, (255, 0, 0), -1)  # Blue center
    
    return frame

def control_mouse(hand_landmarks, frame_width, frame_height):
    """Control mouse using hand gestures - intuitive palm-based controls"""
    if not MOUSE_CONTROL_AVAILABLE or hand_landmarks is None:
        return "Mouse control not available"
    
    # Get screen size
    screen_width, screen_height = autopy.screen.size()
    
    # Get key landmarks
    index_tip = hand_landmarks[8]      # Index finger tip
    middle_tip = hand_landmarks[12]    # Middle finger tip
    ring_tip = hand_landmarks[16]      # Ring finger tip
    pinky_tip = hand_landmarks[20]     # Pinky finger tip
    thumb_tip = hand_landmarks[4]      # Thumb tip
    
    # Get finger bases (knuckles)
    index_base = hand_landmarks[5]
    middle_base = hand_landmarks[9]
    ring_base = hand_landmarks[13]
    pinky_base = hand_landmarks[17]
    
    # Palm center (approximate)
    palm_x = int(hand_landmarks[0].x * frame_width)
    palm_y = int(hand_landmarks[0].y * frame_height)
    
    # Index finger position (for cursor movement)
    index_x = int(index_tip.x * frame_width)
    index_y = int(index_tip.y * frame_height)
    
    # Check if fingers are extended (finger tip above its base)
    def is_finger_extended(tip, base):
        return tip.y < base.y - 0.05  # Tip is above base by threshold
    
    index_extended = is_finger_extended(index_tip, index_base)
    middle_extended = is_finger_extended(middle_tip, middle_base)
    ring_extended = is_finger_extended(ring_tip, ring_base)
    pinky_extended = is_finger_extended(pinky_tip, pinky_base)
    
    fingers_up_count = sum([index_extended, middle_extended, ring_extended, pinky_extended])
    
    # Smooth mouse movement
    smoothing = 5
    curr_x = st.session_state.prev_x + (index_x - st.session_state.prev_x) / smoothing
    curr_y = st.session_state.prev_y + (index_y - st.session_state.prev_y) / smoothing
    
    # Map to screen coordinates
    mouse_x = np.interp(curr_x, (100, frame_width - 100), (0, screen_width))
    mouse_y = np.interp(curr_y, (100, frame_height - 100), (0, screen_height))
    
    # Calculate distances for click gestures
    thumb_x = int(thumb_tip.x * frame_width)
    thumb_y = int(thumb_tip.y * frame_height)
    middle_x = int(middle_tip.x * frame_width)
    middle_y = int(middle_tip.y * frame_height)
    
    thumb_index_dist = np.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
    thumb_middle_dist = np.sqrt((middle_x - thumb_x)**2 + (middle_y - thumb_y)**2)
    
    status = f"Move: ({int(mouse_x)}, {int(mouse_y)})"
    
    # OPEN PALM DRAG: All 4 fingers extended = drag mode
    if fingers_up_count >= 4:
        # Move mouse to palm center position for dragging
        palm_smooth_x = st.session_state.prev_x + (palm_x - st.session_state.prev_x) / smoothing
        palm_smooth_y = st.session_state.prev_y + (palm_y - st.session_state.prev_y) / smoothing
        
        drag_mouse_x = np.interp(palm_smooth_x, (100, frame_width - 100), (0, screen_width))
        drag_mouse_y = np.interp(palm_smooth_y, (100, frame_height - 100), (0, screen_height))
        
        try:
            autopy.mouse.move(drag_mouse_x, drag_mouse_y)
        except:
            pass
        
        if not st.session_state.get('is_dragging', False):
            # Start drag
            try:
                autopy.mouse.toggle(down=True)
                st.session_state.is_dragging = True
                status = "🖐️ DRAG START!"
            except:
                pass
        else:
            status = "🖐️ DRAGGING..."
    
    # FIST SCROLL: All fingers closed (0 fingers up) = scroll mode
    elif fingers_up_count == 0:
        # Use palm Y movement for scrolling
        if 'last_scroll_y' not in st.session_state:
            st.session_state.last_scroll_y = palm_y
        
        scroll_delta = st.session_state.last_scroll_y - palm_y
        
        if abs(scroll_delta) > 15:  # Threshold to avoid jitter
            # Autopy scroll: positive = scroll down, negative = scroll up
            # We invert it for natural scrolling (move hand up = scroll up)
            scroll_clicks = -int(scroll_delta / 30)  # Scale and invert
            
            try:
                # Use smooth scrolling
                for _ in range(abs(scroll_clicks)):
                    autopy.mouse.scroll(0, 1 if scroll_clicks > 0 else -1)
                    time.sleep(0.01)  # Small delay for smooth scroll
                
                if scroll_clicks > 0:
                    status = f"✊ SCROLL DOWN"
                else:
                    status = f"✊ SCROLL UP"
            except Exception as e:
                status = f"✊ Scroll error: {str(e)}"
            
            st.session_state.last_scroll_y = palm_y
        else:
            status = "✊ FIST - Ready to scroll"
    
    else:
        # Normal mode - move mouse with index finger
        try:
            autopy.mouse.move(mouse_x, mouse_y)
        except:
            pass
        
        # End drag if it was active
        if st.session_state.get('is_dragging', False):
            try:
                autopy.mouse.toggle(down=False)
                st.session_state.is_dragging = False
                st.session_state.click_cooldown = 10
                status = "DRAG END!"
            except:
                pass
        
        # LEFT CLICK: Thumb and index pinch (only when not in drag/scroll mode)
        if thumb_index_dist < 40:
            if st.session_state.click_cooldown == 0:
                try:
                    autopy.mouse.click()
                    st.session_state.click_cooldown = 10
                    status = "👌 LEFT CLICK!"
                except:
                    pass
        
        # RIGHT CLICK: Thumb and middle pinch
        elif thumb_middle_dist < 40:
            if st.session_state.click_cooldown == 0:
                try:
                    autopy.mouse.click(autopy.mouse.Button.RIGHT)
                    st.session_state.click_cooldown = 10
                    status = "👌 RIGHT CLICK!"
                except:
                    pass
    
    st.session_state.prev_x = curr_x
    st.session_state.prev_y = curr_y
    
    # Cooldown management
    if st.session_state.click_cooldown > 0:
        st.session_state.click_cooldown -= 1
    
    return status

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Mode selection with automatic camera restart on change
    new_mode = st.radio("Mode", ["Detection", "Mouse Control"])
    
    # Detect mode change and restart camera
    if 'mode' not in st.session_state:
        st.session_state.mode = new_mode
    elif st.session_state.mode != new_mode:
        st.session_state.mode = new_mode
        # Restart camera to apply mode change
        if st.session_state.camera_running:
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.success(f"✅ Switched to {new_mode} mode!")
            time.sleep(0.5)
            st.rerun()
    
    if st.session_state.mode == "Detection":
        st.subheader("Detection")
        st.session_state.auto_add = st.checkbox("Auto-add letters", value=True)
        st.session_state.detection_cooldown = st.slider("Hold time (s)", 0.5, 3.0, 1.5, 0.5)
        
        st.divider()
        
        if TRANSLATION_AVAILABLE:
            st.subheader("Translation")
            available_langs = st.session_state.get('available_languages', ['Spanish'])
            st.session_state.current_language = st.selectbox("Language", available_langs)
    
    else:  # Mouse Control mode
        st.subheader("Mouse Control")
        if MOUSE_CONTROL_AVAILABLE:
            st.success("✅ Mouse control enabled")
            st.info("🖱️ **Intuitive Gestures:**")
            st.markdown("""
            - **Move**: ☝️ Point with index finger
            - **Left Click**: 👌 Pinch thumb + index
            - **Right Click**: 🤏 Pinch thumb + middle
            - **Drag**: 🖐️ Open palm (all fingers up)
            - **Scroll**: ✊ Close fist, move up/down
            """)
            st.caption("💡 Palm-based drag & scroll don't interfere with clicks!")
        else:
            st.warning("⚠️ Mouse control requires `autopy` library")
            st.code("pip install autopy", language="bash")

# Main layout
camera_col, info_col = st.columns([2.5, 1])

with camera_col:
    st.subheader("📹 Live Camera")
    video_placeholder = st.empty()

with info_col:
    st.subheader("🔤 Current Letter")
    letter_display = st.empty()
    
    st.subheader("📊 Stats")
    stats_display = st.empty()

# Sentence and Translation (only for Detection mode)
if st.session_state.mode == "Detection":
    sent_col, trans_col = st.columns(2)
    
    with sent_col:
        st.subheader("📝 Detected Sentence")
        sentence_display = st.empty()
    
    with trans_col:
        st.subheader("🌍 Live Translation")
        translation_display = st.empty()
    
    # Controls
    st.subheader("🎮 Controls")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔤 Space", key="space_btn"):
            st.session_state.sentence += " "
            # Auto-translate
            if TRANSLATION_AVAILABLE and st.session_state.sentence.strip():
                try:
                    st.session_state.translated_sentence = st.session_state.translator.translate(
                        st.session_state.sentence.strip(),
                        st.session_state.current_language
                    )
                except:
                    pass
    
    with col2:
        if st.button("⌫ Backspace", key="backspace_btn"):
            st.session_state.sentence = st.session_state.sentence[:-1]
            # Auto-translate or clear
            if TRANSLATION_AVAILABLE and st.session_state.sentence.strip():
                try:
                    st.session_state.translated_sentence = st.session_state.translator.translate(
                        st.session_state.sentence.strip(),
                        st.session_state.current_language
                    )
                except:
                    pass
            elif not st.session_state.sentence.strip():
                st.session_state.translated_sentence = ""
    
    with col3:
        if st.button("🗑️ Clear", key="clear_btn"):
            st.session_state.sentence = ""
            st.session_state.translated_sentence = ""
    
    with col4:
        if st.button("🔊 Speak", key="speak_btn"):
            if TTS_AVAILABLE and st.session_state.sentence.strip():
                st.session_state.tts_engine.say(st.session_state.sentence)
                st.session_state.tts_engine.runAndWait()

# Camera logic with persistent state - WHILE LOOP (NO st.rerun in loop!)
if st.session_state.initialized:
    if st.session_state.camera_running:
        # Initialize camera once and store in session state
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        cap = st.session_state.cap
        
        if cap.isOpened():
            # WHILE LOOP - runs continuously without st.rerun()
            while st.session_state.camera_running:
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read frame from camera")
                    break
                
                frame = cv2.flip(frame, 1)
                H, W = frame.shape[:2]
                
                # Handle mode-specific processing
                if st.session_state.mode == "Detection":
                    # Detect and predict ONLY in Detection mode
                    char, confidence, landmarks = detect_and_predict(frame)
                    
                    # Draw hand with skeleton and rectangle
                    frame = draw_hand_with_skeleton(frame, landmarks)
                    if char and confidence > 0.5:
                        st.session_state.successful_detections += 1
                        st.session_state.current_detected_letter = char
                        st.session_state.current_confidence = confidence
                        current_time = time.time()
                        
                        # Display on frame
                        cv2.putText(frame, f"Letter: {char}", (20, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        cv2.putText(frame, f"Confidence: {confidence:.0%}", (20, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        
                        if st.session_state.auto_add:
                            if char == st.session_state.last_detected_char:
                                if st.session_state.char_stable_since is None:
                                    st.session_state.char_stable_since = current_time
                                
                                time_held = current_time - st.session_state.char_stable_since
                                
                                # Progress bar
                                progress = min(time_held / st.session_state.detection_cooldown, 1.0)
                                bar_w = int(400 * progress)
                                cv2.rectangle(frame, (W//2 - 200, H - 70), (W//2 - 200 + bar_w, H - 50), (0, 255, 0), -1)
                                cv2.rectangle(frame, (W//2 - 200, H - 70), (W//2 + 200, H - 50), (255, 255, 255), 3)
                                
                                if time_held >= st.session_state.detection_cooldown:
                                    if current_time - st.session_state.last_detection_time > st.session_state.detection_cooldown:
                                        st.session_state.sentence += char
                                        st.session_state.total_detections += 1
                                        st.session_state.last_detection_time = current_time
                                        st.session_state.char_stable_since = None
                                        
                                        # AUTO-TRANSLATE IMMEDIATELY
                                        if TRANSLATION_AVAILABLE and st.session_state.sentence.strip():
                                            try:
                                                st.session_state.translated_sentence = st.session_state.translator.translate(
                                                    st.session_state.sentence.strip(),
                                                    st.session_state.current_language
                                                )
                                            except:
                                                pass
                            else:
                                st.session_state.last_detected_char = char
                                st.session_state.char_stable_since = None
                    else:
                        st.session_state.char_stable_since = None
                        st.session_state.current_detected_letter = None
                        st.session_state.current_confidence = 0.0
                        cv2.putText(frame, "Show hand sign", (20, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 165, 0), 2)
                
                elif st.session_state.mode == "Mouse Control":
                    # Mouse control mode - detect hand for mouse control
                    # We need to detect hand landmarks but NOT predict letters
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb_frame)
                    detection_result = st.session_state.detector.detect(mp_image)
                    
                    if detection_result.hand_landmarks:
                        hand_landmarks = detection_result.hand_landmarks[0]
                        # Draw hand (but no letter detection)
                        frame = draw_hand_with_skeleton(frame, hand_landmarks)
                        # Control mouse
                        status = control_mouse(hand_landmarks, W, H)
                        cv2.putText(frame, status, (20, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    else:
                        cv2.putText(frame, "Show hand to control mouse", (20, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 165, 0), 2)
                    
                    cv2.putText(frame, "Mouse Control Mode", (20, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                
                # FPS
                frame_time = (time.time() - frame_start) * 1000
                st.session_state.frame_times.append(frame_time)
                fps = 1000.0 / np.mean(st.session_state.frame_times) if st.session_state.frame_times else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (W - 150, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_container_width=True)
                
                # Update displays
                if st.session_state.mode == "Detection":
                    # Update current letter display
                    with letter_display:
                        if st.session_state.get('current_detected_letter'):
                            letter = st.session_state.current_detected_letter
                            conf = st.session_state.current_confidence
                            st.markdown(f"""
                            <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px;'>
                                <h1 style='font-size: 100px; margin: 0; color: white;'>{letter}</h1>
                                <p style='font-size: 24px; margin: 10px 0 0 0; color: white;'>{conf:.0%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("👋 Show a sign")
                    
                    # Update stats
                    with stats_display:
                        st.metric("Detections", st.session_state.total_detections)
                        st.metric("FPS", f"{fps:.1f}")
                    
                    # Update sentence
                    with sentence_display:
                        sentence_text = st.session_state.sentence.strip() if st.session_state.sentence else "Start signing..."
                        st.markdown(f"""
                        <div style='font-size: 24px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; min-height: 80px;'>
                            {sentence_text}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Update translation
                    with translation_display:
                        if st.session_state.translated_sentence:
                            st.markdown(f"""
                            <div style='font-size: 22px; padding: 20px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; border-radius: 10px; min-height: 80px;'>
                                {st.session_state.translated_sentence}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info(f"Translation will appear here ({st.session_state.current_language})")
                else:
                    # Mouse Control mode stats
                    with letter_display:
                        st.info("🖱️ Mouse Control Active")
                    with stats_display:
                        st.metric("FPS", f"{fps:.1f}")
                        st.metric("Mode", "Mouse Control")
                
                # Small delay - NO st.rerun()!
                time.sleep(0.03)
        else:
            st.error("❌ Cannot access camera")
    else:
        # Camera stopped - release it
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.info("📷 Camera is off. Click START CAMERA button at the top to begin.")
else:
    st.warning("⚠️ System initializing... Please wait")

st.divider()
