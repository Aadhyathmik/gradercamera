import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
from PIL import Image, ImageEnhance


# Configure Tesseract path if necessary (only needed if Tesseract is not in PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path for Windows

st.title("Assignment Scanner and Text Processor")

# Step 1: Capture an Image or Upload
st.header("Capture an Image or Upload Assignment")
img_option = st.selectbox("Select input method", ("Use Camera", "Upload Image"))

if img_option == "Use Camera":
    # Use Streamlit's camera input for mobile compatibility
    img = st.camera_input("Take a photo of the assignment")
else:
    # Use file uploader for uploaded images
    img = st.file_uploader("Upload an image of the assignment", type=["jpg", "jpeg", "png"])

# Step 2: Process the Image
if img is not None:
    image = Image.open(img)
    st.image(image, caption="Original Image", use_column_width=True)

    # Enhance contrast and sharpness
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Adjust the factor as needed for clarity

    sharpness_enhancer = ImageEnhance.Sharpness(image)
    image = sharpness_enhancer.enhance(2)  # Sharpen the image further

    st.image(image, caption="Enhanced Image", use_column_width=True)

    # Convert to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Grayscale and apply adaptive thresholding
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Optional: Slight blur to reduce noise
    blur = cv2.medianBlur(thresh, 3)
    st.image(blur, caption="Preprocessed Image for OCR", use_column_width=True)

    # OCR extraction
    extracted_text = pytesseract.image_to_string(blur, config="--psm 6")
    st.text_area("Extracted Text", extracted_text, height=300)

    # Step 4: Process the Text (customize this based on your needs)
    # Example: Simple word count
    word_count = len(extracted_text.split())
    st.write("Word Count:", word_count)

    # Step 5: Additional Processing (customize this for assignment-specific needs)
    st.write("Additional processing based on extracted text...")
    # Placeholder for more processing steps
    # Example: keyword analysis, grading, or specific parsing
