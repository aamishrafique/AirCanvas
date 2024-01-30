# Air Canvas: Real-Time Drawing and Text Recognition

### Overview

This Python code implements a real-time hand gesture-based paint application using computer vision techniques. The application allows users to draw on a virtual canvas by tracking their hand movements through a webcam. It incorporates features such as color selection, canvas clearing, and TrOCR to interpret the drawn content.

### Dependencies

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

### Usage

1. Run the AirCanvas.py script.
2. The webcam will open, and you'll see a window with a drawing canvas.
3. Use hand gestures to draw on the canvas, select colors, and clear the drawing.
4. An TrOCR window will display the interpreted text from the drawn content.

### Controls

- To draw, move your hand on the canvas.
- Color selection and clearing the canvas is based on hand positions.
- TrOCR results are displayed in a window.

### Acknowledgments

- This project utilizes the MediaPipe hands module for hand tracking.
- OCR functionality is implemented using pytesseract.
-
