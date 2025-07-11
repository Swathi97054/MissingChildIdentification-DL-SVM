# Missing Child Identification System

This project is a desktop application for identifying missing children using Deep Learning (VGG16) and Multi-Class SVM, built with Tkinter for the user interface.

## Features
- Public portal for uploading suspected child reports
- Official portal for searching, managing reports, and user management
- Facial recognition using VGG16 feature extraction
- Multi-class SVM for matching faces
- Calendar date picker for date fields
- CSV-based data storage

## Requirements
- Python 3.7+
- TensorFlow
- scikit-learn
- pandas
- Pillow
- tkcalendar

## Installation
1. Clone the repository or download the source files.
2. Install dependencies:
   ```bash
   pip install tensorflow scikit-learn pandas pillow tkcalendar
   ```
3. Run the application:
   ```bash
   python vgg16.py
   ```

## Usage
- Public users can upload suspected child details and images.
- Officials can log in, search for matches, view, add, or delete reports, and manage users.

## File Structure
- `vgg16.py` : Main application file
- `missing_children_reports.csv` : Reports data
- `users.csv` : User credentials
- `report_images/` : Uploaded images
- `svm_model.pkl`, `features.pkl` : Model and features data

## Notes
- Default admin login: `admin` / `admin123`
- All data is stored locally in CSV and image files.

## License
MIT License
