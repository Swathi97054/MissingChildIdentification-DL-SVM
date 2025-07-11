# Missing Child Identification System Using Deep Learning and Multi-Class SVM

## Overview
This project is a desktop application for identifying missing children using facial recognition technology. It combines deep learning (VGG16 for feature extraction) and a multi-class SVM classifier to match uploaded images with a database of missing child reports. The system provides separate interfaces for public users and officials, with user authentication and role-based access.

## Features

### General
- **Graphical User Interface (GUI):** Built with Tkinter for an interactive and user-friendly experience.
- **Status Bar:** Real-time status updates for user actions and system processes.
- **Persistent Storage:** Uses CSV and pickle files to store reports, user data, and extracted features.

### Public Portal
- **Upload Suspected Child:** Public users can upload images and details of suspected missing children found in public places.
- **Form Validation:** Ensures all required fields are filled before submission.
- **Image Upload and Preview:** Allows users to select and preview images before submitting a report.

### Official Portal
- **User Authentication:** Officials must log in to access the dashboard. Default admin credentials are provided.
- **Role-Based Access:** Admins can manage users; regular officials can only view and search reports.
- **Dashboard Tabs:**
  - **Reports Tab:** View, refresh, and delete missing child reports. Detailed view for each report, including image and information.
  - **Search Tab:** Upload an image to search for matches in the database using facial recognition. Displays match results with similarity scores and details.
  - **Users Tab (Admin Only):** Add or delete users, view all registered users.

### Facial Recognition & Machine Learning
- **Deep Learning Feature Extraction:** Uses VGG16 (pre-trained on ImageNet) to extract facial features from images.
- **Multi-Class SVM Classifier:** Trains an SVM model to classify and match faces based on extracted features.
- **Cosine Similarity Matching:** Compares uploaded images with database entries to find potential matches.
- **Automatic Model Update:** Retrains the SVM model when new reports are added or deleted.

### Data Management
- **Report Management:** Add, view, and delete missing child reports. Each report includes name, phone, location, date, and image.
- **User Management (Admin):** Add and remove users with different roles (admin/user).
- **Data Persistence:** All data is saved to CSV and pickle files for persistence across sessions.

### Additional Features
- **Date Selection:** Uses a calendar widget for easy date input.
- **Image Handling:** Images are resized and stored in a dedicated directory.
- **Error Handling:** User-friendly error messages and warnings for invalid actions.
- **Print Report (Placeholder):** Simulated print functionality for report details.

## Getting Started
1. **Install Requirements:**
   - Python 3.x
   - Required packages: `tkinter`, `tkcalendar`, `Pillow`, `scikit-learn`, `tensorflow`, `pandas`, `numpy`
2. **Run the Application:**
   - Execute the script: `python MissingChildIdentification-DL-Multiclass_SVM.py`
3. **Default Admin Login:**
   - Username: `admin`
   - Password: `admin123`

## File Structure
- `MissingChildIdentification-DL-Multiclass_SVM.py` — Main application script
- `missing_children_reports.csv` — Stores missing child reports
- `users.csv` — Stores user credentials and roles
- `svm_model.pkl` — Trained SVM model
- `features.pkl` — Extracted image features
- `report_images/` — Directory for uploaded images

## Notes
- This application is for demonstration and educational purposes. For real-world deployment, additional security, privacy, and performance considerations are required.
- The print functionality is a placeholder and should be implemented for actual printing needs.

## License
This project is open source and available under the MIT License.
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
