# Air Canvas: Real-Time Drawing and Text Recognition

## Overview

Air Canvas is a Python application that leverages computer vision techniques to create a real-time, hand gesture-based paint application. This application allows users to draw on a virtual canvas by tracking their hand movements through a webcam. Key features include color selection, canvas clearing, and integration with TrOCR for interpreting the drawn content.

## Model Comparison

| Model             | CER    |
| ----------------- | ------ |
| Microsoft’s TrOCR | 0.0167 |
| Custom Fine-Tuned | 0.0014 |

## Dependencies

To install the required dependencies, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the AirCanvas.py script.
2. The webcam will open, displaying a window with a drawing canvas.
3. Utilize hand gestures to draw on the canvas, select colors, and clear the drawing.
4. A TrOCR window will showcase the interpreted text derived from the drawn content.

## Controls

- Drawing: Move your hand over the canvas.
- Color Selection and Clearing: Hand positions determine color selection and canvas clearing.
- TrOCR Results: Displayed in a separate window.

## Acknowledgments

- This project makes use of the MediaPipe hands module for efficient hand tracking.
- Optical Character Recognition (OCR) functionality is implemented using pytesseract.
