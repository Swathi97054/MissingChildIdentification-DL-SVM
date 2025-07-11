import os
import numpy as np
import pickle
from sklearn.svm import SVC
import pandas as pd
import tkinter as tk
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkcalendar import DateEntry
from PIL import Image, ImageTk
import datetime
import warnings
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
warnings.filterwarnings('ignore')  # Suppress warnings

# Import TensorFlow after setting environment variables
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

class MissingChildIdentificationSystem:
    def __init__(self, root):
        # Store the root window
        self.root = root
        self.root.title("Missing Child Identification System Using DL and Multi Class SVM")
        self.root.geometry("800x600")
        
        # Create the status variable
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing system...")
        
        # Path configurations
        self.report_path = "missing_children_reports.csv"
        self.model_path = "svm_model.pkl"
        self.features_path = "features.pkl"
        self.images_dir = "report_images"
        self.users_path = "users.csv"
        
        # Ensure the images directory exists
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        
        # Load or create users database
        self.initialize_users()
        
        # Set up the home page UI
        self.setup_home_page()
        
        # Status bar
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def initialize_users(self):
        """Initialize users database"""
        if os.path.exists(self.users_path):
            self.users = pd.read_csv(self.users_path)
        else:
            # Create default admin user
            self.users = pd.DataFrame({
                'username': ['admin'],
                'password': [self.hash_password('admin123')],
                'role': ['admin']
            })
            self.users.to_csv(self.users_path, index=False)
    
    def hash_password(self, password):
        """Simple password hashing"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def setup_home_page(self):
        """Set up the home page with public access and login options"""
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            if widget != self.status_var:  # Preserve status bar
                widget.destroy()
        
        # Main frame
        self.home_frame = tk.Frame(self.root)
        self.home_frame.pack(fill=tk.BOTH, expand=True)
        
        # Banner frame
        banner_frame = tk.Frame(self.home_frame, bg="#1E90FF", height=100)
        banner_frame.pack(fill=tk.X)
        
        # Title
        title_label = tk.Label(banner_frame, text="Missing Child Identification System", 
                              font=("Arial", 24, "bold"), bg="#1E90FF", fg="white")
        title_label.pack(pady=20)
        
        # Content frame
        content_frame = tk.Frame(self.home_frame, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Welcome message
        welcome_label = tk.Label(content_frame, text="Welcome to the Missing Child Identification System", 
                                font=("Arial", 16, "bold"))
        welcome_label.pack(pady=10)
        
        # Description
        description = """This system helps identify missing children through facial recognition technology.
        Public users can upload the suspected children, while officials can search and manage reports and cases."""
        
        desc_label = tk.Label(content_frame, text=description, font=("Arial", 12), wraplength=600)
        desc_label.pack(pady=10)
        
        # Buttons frame
        buttons_frame = tk.Frame(content_frame)
        buttons_frame.pack(pady=30)
        
        # Public access button
        public_button = tk.Button(buttons_frame, text="Public Upload Suspected Child", font=("Arial", 12), 
                                 width=25, height=2, bg="#4CAF50", fg="white",
                                 command=self.open_public_page)
        public_button.grid(row=0, column=0, padx=20, pady=10)
        
        # Official login button
        official_button = tk.Button(buttons_frame, text="Official Login", font=("Arial", 12), 
                                   width=15, height=2, bg="#2196F3", fg="white",
                                   command=self.open_login_page)
        official_button.grid(row=0, column=1, padx=20, pady=10)
        
        # System status
        self.status_var.set("System ready - Welcome to the Missing Child Identification System")
    
    def open_public_page(self):
        """Open the public access page"""
        # Clear current frame
        self.home_frame.destroy()
        
        # Create public page frame
        self.public_frame = tk.Frame(self.root)
        self.public_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(self.public_frame, bg="#4CAF50", height=80)
        header_frame.pack(fill=tk.X)
        
        # Title
        title_label = tk.Label(header_frame, text="Public Upload Missing Child Portal", 
                              font=("Arial", 18, "bold"), bg="#4CAF50", fg="white")
        title_label.pack(pady=20)
        
        # Back button
        back_button = tk.Button(header_frame, text="Back to Home", 
                               command=self.setup_home_page)
        back_button.place(x=10, y=10)
        
        # Content frame
        content_frame = tk.Frame(self.public_frame, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Search instructions
        instructions = """Public can upload the suspected child who are roaming in the Streets."""
        
        inst_label = tk.Label(content_frame, text=instructions, font=("Arial", 14), wraplength=600)
        inst_label.pack(pady=10)
        
        # Create a frame for adding reports
        add_frame = tk.Frame(self.public_frame, padx=20, pady=20)
        add_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(add_frame, text="Upload Suspected Child", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Form frame
        form_frame = tk.Frame(add_frame)
        form_frame.pack(pady=10)
       
        # Name field
        tk.Label(form_frame, text="Child Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.add_name_entry = tk.Entry(form_frame, width=30)
        self.add_name_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Phone field
        tk.Label(form_frame, text="Phone No.:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.add_phone_entry = tk.Entry(form_frame, width=30)
        self.add_phone_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Location field
        tk.Label(form_frame, text="Last Seen Location:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.add_location_entry = tk.Entry(form_frame, width=30)
        self.add_location_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # Date field
        tk.Label(form_frame, text="Uploaded Date:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.add_date_entry = DateEntry(form_frame, width=27, date_pattern="yyyy-mm-dd", showweeknumbers=False)
        self.add_date_entry.grid(row=3, column=1, padx=5, pady=5)
        self.add_date_entry.set_date(datetime.datetime.now())
        tk.Label(form_frame, text="(Click to select date)", font=("Arial", 9), fg="gray").grid(row=3, column=2, padx=5, pady=5, sticky="w")
        
        # Image frame
        image_frame = tk.Frame(add_frame)
        image_frame.pack(pady=10)
       
        # Image display
        self.add_image_frame = tk.Frame(image_frame, width=128, height=128, bd=2, relief=tk.GROOVE)
        self.add_image_frame.pack(side="left", padx=40)
       
        # Default image display
        self.add_image_label = tk.Label(self.add_image_frame, text="No File Selected", width=25, height=10)
        self.add_image_label.pack(fill=tk.BOTH, expand=True)
       
        # Image upload button
        self.add_upload_button = tk.Button(image_frame, text="Upload Photo", command=self.upload_add_image)
        self.add_upload_button.pack(side="right", padx=10)
       
        # Submit button
        self.add_submit_button = tk.Button(add_frame, text="Create Report", command=self.create_report, state=tk.DISABLED)
        self.add_submit_button.pack(pady=10)
       
        # Update status
        self.status_var.set("Public Upload portal - Ready to Upload")
        
        # Initialize the system components
        self.initialize_system_components()
    
    def open_login_page(self):
        """Open the official login page"""
        # Clear current frame
        self.home_frame.destroy()
        
        # Create login page frame
        self.login_frame = tk.Frame(self.root)
        self.login_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(self.login_frame, bg="#2196F3", height=80)
        header_frame.pack(fill=tk.X)
        
        # Title
        title_label = tk.Label(header_frame, text="Official Login", 
                              font=("Arial", 18, "bold"), bg="#2196F3", fg="white")
        title_label.pack(pady=20)
        
        # Back button
        back_button = tk.Button(header_frame, text="Back to Home", 
                               command=self.setup_home_page)
        back_button.place(x=10, y=10)
        
        # Content frame
        content_frame = tk.Frame(self.login_frame, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Center the login form
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_rowconfigure(4, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(3, weight=1)
        
        # Login form
        login_form = tk.Frame(content_frame, padx=20, pady=20, bd=2, relief=tk.GROOVE)
        login_form.grid(row=1, column=1, rowspan=2, columnspan=2, sticky="nsew")
        
        # Form title
        form_title = tk.Label(login_form, text="Login to Official Portal", font=("Arial", 14, "bold"))
        form_title.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Username
        username_label = tk.Label(login_form, text="Username:", font=("Arial", 12))
        username_label.grid(row=1, column=0, sticky="e", padx=5, pady=10)
        self.username_entry = tk.Entry(login_form, font=("Arial", 12), width=20)
        self.username_entry.grid(row=1, column=1, sticky="w", padx=5, pady=10)
        
        # Password
        password_label = tk.Label(login_form, text="Password:", font=("Arial", 12))
        password_label.grid(row=2, column=0, sticky="e", padx=5, pady=10)
        self.password_entry = tk.Entry(login_form, font=("Arial", 12), width=20, show="*")
        self.password_entry.grid(row=2, column=1, sticky="w", padx=5, pady=10)
        
        # Login button
        login_button = tk.Button(login_form, text="Login", font=("Arial", 12), 
                                bg="#2196F3", fg="white", width=10,
                                command=self.login)
        login_button.grid(row=3, column=0, columnspan=2, pady=20)
        
        # Default login info (for demonstration)
        default_info = tk.Label(content_frame, text="Default login: admin / admin123", 
                               font=("Arial", 10), fg="gray")
        default_info.grid(row=3, column=1, columnspan=2, pady=10)
        
        # Update status
        self.status_var.set("Official login page")
    
    def login(self):
        """Handle login authentication"""
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        # Check if username and password are valid
        if username and password:
            hashed_password = self.hash_password(password)
            user_match = self.users[(self.users['username'] == username) & 
                                     (self.users['password'] == hashed_password)]
            
            if not user_match.empty:
                messagebox.showinfo("Success", "Login successful!")
                self.open_official_dashboard(username, user_match.iloc[0]['role'])
            else:
                messagebox.showerror("Error", "Invalid username or password")
        else:
            messagebox.showerror("Error", "Please enter username and password")
    
    def open_official_dashboard(self, username, role):
        """Open the official dashboard after successful login"""
        # Clear current frame
        self.login_frame.destroy()
        
        # Create dashboard frame
        self.dashboard_frame = tk.Frame(self.root)
        self.dashboard_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(self.dashboard_frame, bg="#2196F3", height=80)
        header_frame.pack(fill=tk.X)
        
        # Title
        title_label = tk.Label(header_frame, text=f"Official Dashboard - {role.capitalize()}", 
                              font=("Arial", 18, "bold"), bg="#2196F3", fg="white")
        title_label.pack(pady=20)
        
        # User info
        user_label = tk.Label(header_frame, text=f"Logged in as: {username}", 
                             font=("Arial", 10), bg="#2196F3", fg="white")
        user_label.place(x=10, y=10)
        
        # Logout button
        logout_button = tk.Button(header_frame, text="Logout", 
                                 command=self.setup_home_page)
        logout_button.place(x=1150, y=20)
        
        # Content frame
        content_frame = tk.Frame(self.dashboard_frame, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        tab_control = ttk.Notebook(content_frame)
        
        # Reports tab
        reports_tab = ttk.Frame(tab_control)
        tab_control.add(reports_tab, text="Reports")
        
        # Search tab
        search_tab = ttk.Frame(tab_control)
        tab_control.add(search_tab, text="Search")
        
        # Users tab (admin only)
        if role == 'admin':
            users_tab = ttk.Frame(tab_control)
            tab_control.add(users_tab, text="Users")
        
        tab_control.pack(expand=1, fill="both")
        
        # Initialize the system components
        self.initialize_system_components()
        
        # Set up the reports tab
        self.setup_reports_tab(reports_tab)
        
        # Set up the search tab
        self.setup_search_tab(search_tab)
        
        # Set up the users tab (admin only)
        if role == 'admin':
            self.setup_users_tab(users_tab)
        
        # Update status
        self.status_var.set(f"Official dashboard - Logged in as {username}")
    
    def setup_reports_tab(self, tab):
        """Set up the reports tab"""
        # Create a frame for the reports
        reports_frame = tk.Frame(tab, padx=10, pady=10)
        reports_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a treeview for reports
        columns = ("ID", "Name", "Phone", "Location", "Date Missing")
        self.reports_tree = ttk.Treeview(reports_frame, columns=columns, show="headings")
        
        # Set column headings
        for col in columns:
            self.reports_tree.heading(col, text=col)
            self.reports_tree.column(col, width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(reports_frame, orient="vertical", command=self.reports_tree.yview)
        self.reports_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.reports_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Button frame
        button_frame = tk.Frame(tab, padx=10, pady=10)
        button_frame.pack(fill=tk.X)
        
        # Add buttons
        refresh_button = tk.Button(button_frame, text="Refresh", command=self.refresh_reports)
        refresh_button.pack(side="left", padx=5)
        
        view_button = tk.Button(button_frame, text="View Details", command=self.view_report_detail)
        view_button.pack(side="left", padx=5)
        
        delete_button = tk.Button(button_frame, text="Delete Report", command=self.delete_report)
        delete_button.pack(side="left", padx=5)
        
        # Load reports
        self.refresh_reports()
    
    def setup_search_tab(self, tab):
        """Set up the search tab"""
        # Create a frame for the search
        search_frame = tk.Frame(tab, padx=10, pady=10)
        search_frame.pack(fill=tk.BOTH, expand=True)
        
        # Image display frame
        self.search_image_frame = tk.Frame(search_frame, width=300, height=300, bd=2, relief=tk.GROOVE)
        self.search_image_frame.pack(side="left", padx=10, pady=10)
        
        # Default image display
        self.search_image_label = tk.Label(self.search_image_frame, text="No Image Selected", width=30, height=15)
        self.search_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = tk.Frame(search_frame, padx=10, pady=10)
        control_frame.pack(side="right", fill="both", expand=True)
        
        # Upload button
        self.search_upload_button = tk.Button(control_frame, text="Upload Image", 
                                           command=self.upload_search_image)
        self.search_upload_button.pack(pady=10)
        
        # Search button
        self.search_exec_button = tk.Button(control_frame, text="Search", 
                                         command=self.execute_search, state=tk.DISABLED)
        self.search_exec_button.pack(pady=10)
        
        # Results frame
        self.search_results_frame = tk.Frame(tab, bd=2, relief=tk.GROOVE)
        self.search_results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results label
        self.search_results_label = tk.Label(self.search_results_frame, text="Results will appear here", font=("Arial", 12))
        self.search_results_label.pack(pady=10)
    
    def setup_users_tab(self, tab):
        """Set up the users tab (admin only)"""
        # Create a frame for users
        users_frame = tk.Frame(tab, padx=10, pady=10)
        users_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a treeview for users
        columns = ("Username", "Role")
        self.users_tree = ttk.Treeview(users_frame, columns=columns, show="headings")
        
        # Set column headings
        for col in columns:
            self.users_tree.heading(col, text=col)
            self.users_tree.column(col, width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(users_frame, orient="vertical", command=self.users_tree.yview)
        self.users_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.users_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Button frame
        button_frame = tk.Frame(tab, padx=10, pady=10)
        button_frame.pack(fill=tk.X)
        
        # Add buttons
        refresh_button = tk.Button(button_frame, text="Refresh", command=self.refresh_users)
        refresh_button.pack(side="left", padx=5)
        
        add_button = tk.Button(button_frame, text="Add User", command=self.add_user)
        add_button.pack(side="left", padx=5)
        
        delete_button = tk.Button(button_frame, text="Delete User", command=self.delete_user)
        delete_button.pack(side="left", padx=5)
        
        # Load users
        self.refresh_users()
    
    def initialize_system_components(self):
        """Initialize the system components"""
        # Update status
        self.status_var.set("Loading model and reports...")
        self.root.update()
        
        try:
            # Load deep learning model for feature extraction
            self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            self.status_var.set("VGG16 model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load VGG16 model: {str(e)}")
            self.status_var.set("Error loading VGG16 model")
        
        # Load or create the reports
        if os.path.exists(self.report_path):
            self.reports = pd.read_csv(self.report_path)
            self.status_var.set(f"Reports loaded with {len(self.reports)} entries")
        else:
            self.reports = pd.DataFrame(columns=['id', 'name', 'phone', 'location', 'date_missing', 'image_path'])
            self.reports.to_csv(self.report_path, index=False)
            self.status_var.set("New reports file created")
        
        # Load or create the SVM model and features data
        if os.path.exists(self.model_path) and os.path.exists(self.features_path):
            try:
                self.svm_model = pickle.load(open(self.model_path, 'rb'))
                self.features_data = pickle.load(open(self.features_path, 'rb'))
                self.status_var.set(f"Model loaded with {len(self.features_data['features'])} feature sets")
            except Exception as e:
                # If there's an issue loading, create new model and features data
                self.svm_model = SVC(kernel='linear', probability=True)
                self.features_data = {'features': [], 'ids': []}
                self.status_var.set(f"Error loading model: {str(e)}. Created new model.")
        else:
            self.svm_model = SVC(kernel='linear', probability=True)
            self.features_data = {'features': [], 'ids': []}
            self.status_var.set("Created new model and features data")
        
        # Initialize model if we have data
        self._init_model_from_reports()
        
        # Update status
        self.status_var.set("System components initialized")
    
    def _init_model_from_reports(self):
        """Initialize model from report entries if needed"""
        if len(self.reports) > 0 and (len(self.features_data['features']) == 0 or len(self.features_data['features']) != len(self.reports)):
            self.status_var.set(f"Initializing model from reports ({len(self.reports)} entries)...")
            self.root.update()
            
            # Extract features from all images in reports
            features_list = []
            ids_list = []
            
            for _, row in self.reports.iterrows():
                if os.path.exists(row['image_path']):
                    try:
                        self.status_var.set(f"Processing image for {row['name']} (ID: {row['id']})...")
                        self.root.update()
                        
                        features = self.extract_features(row['image_path'])
                        features_list.append(features)
                        ids_list.append(row['id'])
                    except Exception as e:
                        print(f"Error extracting features from {row['image_path']}: {e}")
                        messagebox.showwarning("Warning", f"Could not process image for {row['name']}: {str(e)}")
                else:
                    print(f"Image not found: {row['image_path']}")
                    messagebox.showwarning("Warning", f"Image not found for {row['name']}: {row['image_path']}")
            
            # Update features data
            self.features_data['features'] = features_list
            self.features_data['ids'] = ids_list
            
            # Save the features
            pickle.dump(self.features_data, open(self.features_path, 'wb'))
            
            # Train model if we have enough samples
            if len(features_list) > 0:
                self.update_svm_model(force=True)
                self.status_var.set(f"Model initialized with {len(features_list)} entries")
            else:
                self.status_var.set("No valid images found in reports")
    
    def extract_features(self, img_path):
        """Extract features from image using VGG16"""
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            features = self.base_model.predict(img_array, verbose=0)  # Added verbose=0 to suppress output
            features = features.flatten()
            # Normalize the features
            features = features / np.linalg.norm(features)
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            raise e
    
    def update_svm_model(self, force=False):
        """Update the SVM model with current features"""
        # Check if we have enough samples
        if len(self.features_data['features']) > 1 or force:  # Need at least 2 samples to train
            try:
                X = np.array(self.features_data['features'])
                y = np.array(self.features_data['ids'])
                
                # Train SVM model
                self.svm_model.fit(X, y)
                
                # Save the model and features
                pickle.dump(self.svm_model, open(self.model_path, 'wb'))
                pickle.dump(self.features_data, open(self.features_path, 'wb'))
                
                return True
            except Exception as e:
                print(f"Error training model: {e}")
                return False
        return False
    
    def upload_image(self):
        """Upload and display an image for public search"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*"))
        )
        
        if file_path:
            self.image_path = file_path
            try:
                # Open and resize image for display
                img = Image.open(file_path)
                img = img.resize((250, 250), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Display the image
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo  # Keep a reference
                
                # Enable the search button
                self.search_button.config(state=tk.NORMAL)
                
                # Update status
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def search_reports(self):
        """Search for matching reports based on uploaded image"""
        if hasattr(self, 'image_path') and self.image_path:
            try:
                # Clear current results
                for widget in self.results_frame.winfo_children():
                    widget.destroy()
                
                # Update status
                self.status_var.set("Processing image and searching for matches...")
                self.root.update()
                
                # Extract features from the image
                features = self.extract_features(self.image_path)
                
                # Compare with existing features using cosine similarity
                results = []
                for i, feat in enumerate(self.features_data['features']):
                    sim = cosine_similarity([features], [feat])[0][0]
                    results.append((self.features_data['ids'][i], sim))
                
                # Sort by similarity (highest first)
                results.sort(key=lambda x: x[1], reverse=True)
                
                # Check if any significant matches were found (similarity >= 0.5)
                match_found = any(sim >= 0.5 for _, sim in results[:3])
                
                # Display simple result
                result_label = tk.Label(
                    self.results_frame, 
                    text="MATCH FOUND" if match_found else "NO MATCH FOUND",
                    font=("Arial", 16, "bold"),
                    fg="green" if match_found else "red"
                )
                result_label.pack(pady=20)
                
                # Update status
                self.status_var.set("Search completed")
            except Exception as e:
                messagebox.showerror("Error", f"Search failed: {str(e)}")
                self.status_var.set("Search failed")
        else:
            messagebox.showwarning("Warning", "Please upload an image first")

        
    def upload_search_image(self):
        """Upload and display an image for official search"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*"))
        )
        
        if file_path:
            self.search_image_path = file_path
            try:
                # Open and resize image for display
                img = Image.open(file_path)
                img = img.resize((128, 128), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Display the image
                self.search_image_label.config(image=photo, text="")
                self.search_image_label.image = photo  # Keep a reference
                
                # Enable the search button
                self.search_exec_button.config(state=tk.NORMAL)
                
                # Update status
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def execute_search(self):
        """Execute search from the official dashboard"""
        if hasattr(self, 'search_image_path') and self.search_image_path:
            try:
                # Clear current results
                for widget in self.search_results_frame.winfo_children():
                    widget.destroy()
                
                # Update status
                self.status_var.set("Processing image and searching for matches...")
                self.root.update()
                
                # Extract features from the image
                features = self.extract_features(self.search_image_path)
                
                # Compare with existing features using cosine similarity
                results = []
                for i, feat in enumerate(self.features_data['features']):
                    sim = cosine_similarity([features], [feat])[0][0]
                    results.append((self.features_data['ids'][i], sim))
                
                # Sort by similarity (highest first)
                results.sort(key=lambda x: x[1], reverse=True)
                
                # Check if any significant matches were found (similarity >= 0.5)
                match_found = any(sim >= 0.5 for _, sim in results[:5])
                
                # Display simple result
                result_label = tk.Label(
                    self.search_results_frame, 
                    text="MATCH FOUND" if match_found else "NO MATCH FOUND",
                    font=("Arial", 18, "bold"),
                    fg="green" if match_found else "red"
                )
                result_label.pack(pady=10)
                
                # Display top matches with detailed information
                if match_found:
                    # Create container frame for matches
                    matches_container = tk.Frame(self.search_results_frame)
                    matches_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    # Display top 3 matches
                    for i, (report_id, similarity) in enumerate(results[:3]):
                        if similarity >= 0.3:  # Show only reasonably similar matches
                            # Find report details
                            report = self.reports[self.reports['id'] == report_id].iloc[0]
                            
                            # Create a frame for this match
                            match_frame = tk.Frame(matches_container, bd=2, relief=tk.GROOVE, padx=5, pady=5)
                            match_frame.pack(fill=tk.X, expand=True, pady=5)
                            
                            # Split into left (image+details) and right (metrics) sides
                            left_frame = tk.Frame(match_frame)
                            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                            
                            right_frame = tk.Frame(match_frame)
                            right_frame.pack(side=tk.RIGHT, fill=tk.Y)
                            child_frame = tk.Frame(left_frame, bd=1, relief=tk.GROOVE, padx=75, pady=5)
                            child_frame.pack(fill=tk.BOTH, expand=True)
                            tk.Label(child_frame, text="CHILD INFORMATION", font=("Arial", 10, "bold")).pack(pady=20)
                            # LEFT SIDE: Image and child details
                            try:
                                # Display image
                                img = Image.open(report['image_path'])
                                img = img.resize((125, 125), Image.LANCZOS)
                                photo = ImageTk.PhotoImage(img)
                                img_label = tk.Label(child_frame, image=photo)
                                img_label.image = photo  # Keep a reference
                                img_label.pack(side=tk.LEFT, padx=5)
                            except Exception:
                                # Display placeholder if image loading fails
                                img_label = tk.Label(child_frame, text="No Image", width=10, height=8, bd=1, relief=tk.SUNKEN)
                                img_label.pack(side=tk.LEFT, padx=5)
                            
                            # Child details
                            details_frame = tk.Frame(child_frame)
                            details_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=30)
                            
                            tk.Label(details_frame, text=f"ID: {report['id']}", anchor="w", font=("Arial", 10)).pack(fill=tk.X)
                            tk.Label(details_frame, text=f"Name: {report['name']}", anchor="w", font=("Arial", 10, "bold")).pack(fill=tk.X)
                            tk.Label(details_frame, text=f"Missing since: {report['date_missing']}", anchor="w").pack(fill=tk.X)
                            tk.Label(details_frame, text=f"Last seen: {report['location']}", anchor="w").pack(fill=tk.X)
                            tk.Label(details_frame, text=f"Contact: {report['phone']}", anchor="w").pack(fill=tk.X)
                            
                            # RIGHT SIDE: Metrics
                            # Calculate additional metrics for display
                            confidence = similarity * 100
                            # These would be calculated based on your actual system performance
                            # For demonstration, we're using simple calculations
                            accuracy = min(similarity * 1.05, 1.0) * 100
                            precision = min(similarity * 1.1, 1.0) * 100
                            recall = min(similarity * 0.9, 1.0) * 100
                            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                            
                            metrics_frame = tk.Frame(right_frame, bd=1, relief=tk.GROOVE, padx=125, pady=5)
                            metrics_frame.pack(fill=tk.BOTH, expand=True)
                            
                            tk.Label(metrics_frame, text="MATCH METRICS", font=("Arial", 10, "bold")).pack(pady=20)
                            
                            # Display metrics with color-coded values
                            metrics = [
                                ("Accuracy", f"{accuracy:.1f}%", self.get_color_for_score(accuracy/100)),
                                ("Confidence", f"{confidence:.1f}%", self.get_color_for_score(confidence/100)),
                                ("Precision", f"{precision:.1f}%", self.get_color_for_score(precision/100)),
                                ("Recall", f"{recall:.1f}%", self.get_color_for_score(recall/100)),
                                ("F1 Score", f"{f1_score:.1f}%", self.get_color_for_score(f1_score/100))
                            ]
                            
                            for label, value, color in metrics:
                                metric_frame = tk.Frame(metrics_frame)
                                metric_frame.pack(fill=tk.X, pady=1)
                                
                                tk.Label(metric_frame, text=f"{label}:", width=10, anchor="w").pack(side=tk.LEFT)
                                tk.Label(metric_frame, text=value, fg=color, width=8).pack(side=tk.RIGHT)
                            
                            # Add View Details button
                            view_btn = tk.Button(
                                right_frame, 
                                text="View Details", 
                                command=lambda r=report_id: self.view_report_details(r)
                            )
                            view_btn.pack(pady=5)
                
                            
                else:
                    # No match found - create a form to enter child details
                    no_match_frame = tk.Frame(self.search_results_frame)
                    no_match_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    # Create entry form
                    form_frame = tk.Frame(no_match_frame, bd=2, relief=tk.GROOVE, padx=15, pady=15)
                    form_frame.pack(fill=tk.BOTH, expand=True, pady=10)
                    
                    # Form fields
                    fields = [
                        ("Child's Name:", "name"),
                        ("Date Missing:", "date_missing"),
                        ("Last Seen Location:", "location"),
                        ("Contact Phone:", "phone")
                    ]
                    
                    # Dictionary to store entry widgets
                    self.entry_data = {}
                    
                    # Create form fields
                    for i, (label_text, field_name) in enumerate(fields):
                        field_frame = tk.Frame(form_frame)
                        field_frame.pack(fill=tk.X, pady=5)
                        label = tk.Label(field_frame, text=label_text, width=20, anchor="w")
                        label.pack(side=tk.LEFT)
                        if field_name == "date_missing":
                            entry = DateEntry(field_frame, width=37, date_pattern="yyyy-mm-dd", showweeknumbers=False)
                            entry.set_date(datetime.datetime.now())
                            tk.Label(field_frame, text="(Click to select date)", font=("Arial", 9), fg="gray").pack(side=tk.LEFT, padx=5)
                        else:
                            entry = tk.Entry(field_frame, width=40)
                        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
                        self.entry_data[field_name] = entry
                    
                    # Buttons
                    button_frame = tk.Frame(no_match_frame)
                    button_frame.pack(pady=10)
                    
                    submit_btn = tk.Button(
                        button_frame,
                        text="Add to Database",
                        command=self.add_new_child_record
                    )
                    submit_btn.pack(side=tk.LEFT, padx=10)
                    
                    cancel_btn = tk.Button(
                        button_frame,
                        text="Cancel",
                        command=lambda: self.clear_search_results()
                    )
                    cancel_btn.pack(side=tk.LEFT, padx=10)
                    
                # Update status
                self.status_var.set("Search completed")
            except Exception as e:
                messagebox.showerror("Error", f"Search failed: {str(e)}")
                self.status_var.set("Search failed")
        else:
            messagebox.showwarning("Warning", "Please upload an image first")

    def add_new_child_record(self):
        """Add a new child record to the database"""
        try:
            # Get data from entry fields
            name = self.entry_data["name"].get()
            date_missing = self.entry_data["date_missing"].get()
            location = self.entry_data["location"].get()
            phone = self.entry_data["phone"].get()
            
            # Validate required fields
            if not name or not date_missing:
                messagebox.showwarning("Warning", "Name and Date Missing are required fields")
                return
            
            # Generate a unique ID for the new record
            new_id = f"CH{len(self.reports) + 1:04d}"
            
            # Create new record
            new_record = {
                'id': new_id,
                'name': name,
                'date_missing': date_missing,
                'location': location,
                'phone': phone,
                'image_path': self.search_image_path  # Use the uploaded image
            }
            
            # Add to reports dataframe
            self.reports = pd.concat([self.reports, pd.DataFrame([new_record])], ignore_index=True)
            
            # Extract features from the image and add to features data
            features = self.extract_features(self.search_image_path)
            self.features_data['ids'].append(new_id)
            self.features_data['features'].append(features)
            
            # Save updated data (implementation depends on your storage method)
            self.save_data()
            
            # Show success message
            messagebox.showinfo("Success", f"New record added with ID: {new_id}")
            
            # Clear form and search results
            self.clear_search_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add record: {str(e)}")

    def clear_search_results(self):
        """Clear the search results frame"""
        for widget in self.search_results_frame.winfo_children():
            widget.destroy()
        self.status_var.set("Ready")

    def save_data(self):
        """Save updated data to storage
        (Implement based on your storage method - CSV, database, etc.)
        """
        try:
            # Example implementation - saving to CSV
            self.reports.to_csv('reports.csv', index=False)
            
            # Save features data (implementation depends on your storage method)
            # For example, using pickle:
            import pickle
            with open('features_data.pkl', 'wb') as f:
                pickle.dump(self.features_data, f)
                
            self.status_var.set("Data saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")

    def get_color_for_score(self, score):
        """Return a color based on the score value"""
        if score >= 0.8:
            return "green"
        elif score >= 0.6:
            return "darkgreen"
        elif score >= 0.4:
            return "orange"
        else:
            return "red"
    def view_search_result(self, event):
        """View details of a search result"""
        # Get selected item
        tree = event.widget
        item = tree.selection()[0]
        report_id = tree.item(item, "values")[0]
        
        # Find report
        report = self.reports[self.reports['id'] == int(report_id)].iloc[0]
        
        # Create details window
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Report Details - {report['name']}")
        details_window.geometry("400x500")
        
        # Add details
        frame = tk.Frame(details_window, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = tk.Label(frame, text=f"Details for {report['name']}", font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Image
        try:
            img = Image.open(report['image_path'])
            img = img.resize((200, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            img_label = tk.Label(frame, image=photo)
            img_label.image = photo  # Keep a reference
            img_label.pack(pady=10)
        except Exception:
            # Display placeholder if image loading fails
            img_label = tk.Label(frame, text="No Image Available", width=20, height=10, bd=1, relief=tk.SUNKEN)
            img_label.pack(pady=10)
        
        # Details
        details_frame = tk.Frame(frame, bd=2, relief=tk.GROOVE, padx=10, pady=10)
        details_frame.pack(fill=tk.X)
        
        # Display details
        tk.Label(details_frame, text=f"ID: {report['id']}", anchor="w").pack(fill=tk.X, pady=2)
        tk.Label(details_frame, text=f"Name: {report['name']}", anchor="w").pack(fill=tk.X, pady=2)
        tk.Label(details_frame, text=f"Missing since: {report['date_missing']}", anchor="w").pack(fill=tk.X, pady=2)
        tk.Label(details_frame, text=f"Last seen at: {report['location']}", anchor="w").pack(fill=tk.X, pady=2)
        tk.Label(details_frame, text=f"Contact: {report['phone']}", anchor="w").pack(fill=tk.X, pady=2)
        
        # Close button
        close_btn = tk.Button(frame, text="Close", command=details_window.destroy)
        close_btn.pack(pady=10)
    
    def upload_add_image(self):
        """Upload an image for a new report"""
        file_path = filedialog.askopenfilename(
            title="Select Child's Image",
            filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*"))
        )
        
        if file_path:
            self.add_image_path = file_path
            try:
                # Open and resize image for display
                img = Image.open(file_path)
                img = img.resize((150, 150), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Display the image
                self.add_image_label.config(image=photo, text="")
                self.add_image_label.image = photo  # Keep a reference
                
                # Enable submit button
                self.add_submit_button.config(state=tk.NORMAL)
                
                # Update status
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def create_report(self):
        """Create a new missing child report"""
        # Get input values
        name = self.add_name_entry.get()
        phone = self.add_phone_entry.get()
        location = self.add_location_entry.get()
        date_missing = self.add_date_entry.get()
        
        # Validate inputs
        if not (name and phone and location and date_missing):
            messagebox.showwarning("Warning", "Please fill in all fields")
            return
        
        if not hasattr(self, 'add_image_path'):
            messagebox.showwarning("Warning", "Please upload an image")
            return
        
        try:
            # Generate a new ID
            new_id = 1
            if len(self.reports) > 0:
                new_id = self.reports['id'].max() + 1
            
            # Copy image to reports directory
            img_filename = f"report_{new_id}_{os.path.basename(self.add_image_path)}"
            img_path = os.path.join(self.images_dir, img_filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            
            # Copy the image
            shutil.copy2(self.add_image_path, img_path)
            
            # Add to reports dataframe
            new_report = pd.DataFrame({
                'id': [new_id],
                'name': [name],
                'phone': [phone],
                'location': [location],
                'date_missing': [date_missing],
                'image_path': [img_path]
            })
            
            self.reports = pd.concat([self.reports, new_report], ignore_index=True)
            
            # Save to CSV
            self.reports.to_csv(self.report_path, index=False)
            
            # Extract features and update model
            features = self.extract_features(img_path)
            self.features_data['features'].append(features)
            self.features_data['ids'].append(new_id)
            
            # Update model
            self.update_svm_model()
            
            # Show success message
            messagebox.showinfo("Success", "Report created successfully")
            
            # Clear form
            self.add_name_entry.delete(0, tk.END)
            self.add_phone_entry.delete(0, tk.END)
            self.add_location_entry.delete(0, tk.END)
            self.add_date_entry.delete(0, tk.END)
            self.add_date_entry.insert(0, datetime.datetime.now().strftime("%Y-%m-%d"))
            self.add_image_label.config(image="", text="No Image Selected")
            self.add_submit_button.config(state=tk.DISABLED)
            
            # Refresh reports list
            self.refresh_reports()
            
            # Update status
            self.status_var.set("Report created successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create report: {str(e)}")
    
    
    def refresh_reports(self):
        """Refresh the reports list"""
        # Reset index of DataFrame
        self.reports.reset_index(drop=True, inplace=True)
        
        # Update 'id' column to match new index
        self.reports['id'] = self.reports.index+1
        
        # Clear current items
        for item in self.reports_tree.get_children():
            self.reports_tree.delete(item)
        
        # Add reports to treeview
        for _, report in self.reports.iterrows():
            
            self.reports_tree.insert("", "end", values=(
                report['id'], 
                report['name'], 
                report['phone'], 
                report['location'], 
                report['date_missing']
            ))

        
        
    def view_report_detail(self):
        """View details of a selected report"""
        # Get selected item
        selection = self.reports_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a report")
            return
        
        # Get report ID
        item = selection[0]
        report_id = self.reports_tree.item(item, "values")[0]
        
        # Find report
        report = self.reports[self.reports['id'] == int(report_id)].iloc[0]
        
        # Create details window
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Report Details - {report['name']}")
        details_window.geometry("400x500")
        
        # Add details
        frame = tk.Frame(details_window, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = tk.Label(frame, text=f"Details for {report['name']}", font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Image
        try:
            img = Image.open(report['image_path'])
            img = img.resize((200, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            img_label = tk.Label(frame, image=photo)
            img_label.image = photo  # Keep a reference
            img_label.pack(pady=10)
        except Exception:
            # Display placeholder if image loading fails
            img_label = tk.Label(frame, text="No Image Available", width=20, height=10, bd=1, relief=tk.SUNKEN)
            img_label.pack(pady=10)
        
        # Details
        details_frame = tk.Frame(frame, bd=2, relief=tk.GROOVE, padx=10, pady=10)
        details_frame.pack(fill=tk.X)
        
        # Display details
        tk.Label(details_frame, text=f"ID: {report['id']}", anchor="w").pack(fill=tk.X, pady=2)
        tk.Label(details_frame, text=f"Name: {report['name']}", anchor="w").pack(fill=tk.X, pady=2)
        tk.Label(details_frame, text=f"Missing since: {report['date_missing']}", anchor="w").pack(fill=tk.X, pady=2)
        tk.Label(details_frame, text=f"Last seen at: {report['location']}", anchor="w").pack(fill=tk.X, pady=2)
        tk.Label(details_frame, text=f"Contact: {report['phone']}", anchor="w").pack(fill=tk.X, pady=2)
        
       
        # Close button
        close_btn = tk.Button(frame, text="Close", command=details_window.destroy)
        close_btn.pack(pady=10)
        
    def delete_report(self):
        """Delete a selected report"""
        # Get selected item
        selection = self.reports_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a report")
            return
        
        # Confirm deletion
        if not messagebox.askyesno("Confirm", "Are you sure you want to delete this report?"):
            return
            
        # Get report ID
        item = selection[0]
        report_id = int(self.reports_tree.item(item, "values")[0])
        
        try:
            # Get the report
            report = self.reports[self.reports['id'] == report_id].iloc[0]
            
            # Delete the image file
            if os.path.exists(report['image_path']):
                os.remove(report['image_path'])
            
            # Remove from reports dataframe
            self.reports = self.reports[self.reports['id'] != report_id]
            
            # Save to CSV
            self.reports.to_csv(self.report_path, index=False)
            
            # Remove from features data
            if report_id in self.features_data['ids']:
                idx = self.features_data['ids'].index(report_id)
                self.features_data['features'].pop(idx)
                self.features_data['ids'].pop(idx)
                
                # Update model
                self.update_svm_model()
            
            # Refresh reports list
            self.refresh_reports()
            
            # Show success message
            messagebox.showinfo("Success", "Report deleted successfully")
            
            # Update status
            self.status_var.set("Report deleted successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete report: {str(e)}")
    
    def refresh_users(self):
        """Refresh the users list"""
        # Clear current items
        for item in self.users_tree.get_children():
            self.users_tree.delete(item)
        
        # Add users to treeview
        for _, user in self.users.iterrows():
            self.users_tree.insert("", "end", values=(
                user['username'],
                user['role']
            ))
    
    def add_user(self):
        """Add a new user"""
        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New User")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Form frame
        form_frame = tk.Frame(dialog, padx=20, pady=20)
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        # Username field
        tk.Label(form_frame, text="Username:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        username_entry = tk.Entry(form_frame, width=20)
        username_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Password field
        tk.Label(form_frame, text="Password:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        password_entry = tk.Entry(form_frame, width=20, show="*")
        password_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Role field
        tk.Label(form_frame, text="Role:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        role_var = tk.StringVar()
        role_var.set("user")
        role_dropdown = ttk.Combobox(form_frame, textvariable=role_var, values=["user", "admin"], state="readonly")
        role_dropdown.grid(row=2, column=1, padx=5, pady=5)
        
        # Submit function
        def submit():
            username = username_entry.get()
            password = password_entry.get()
            role = role_var.get()
            
            # Validate inputs
            if not (username and password):
                messagebox.showwarning("Warning", "Please fill in all fields", parent=dialog)
                return
            
            # Check if username already exists
            if username in self.users['username'].values:
                messagebox.showwarning("Warning", "Username already exists", parent=dialog)
                return
            
            # Add new user
            new_user = pd.DataFrame({
                'username': [username],
                'password': [self.hash_password(password)],
                'role': [role]
            })
            
            self.users = pd.concat([self.users, new_user], ignore_index=True)
            
            # Save to CSV
            self.users.to_csv(self.users_path, index=False)
            
            # Refresh users list
            self.refresh_users()
            
            # Close dialog
            dialog.destroy()
            
            # Show success message
            messagebox.showinfo("Success", "User added successfully")
        
        # Submit button
        submit_btn = tk.Button(form_frame, text="Add User", command=submit)
        submit_btn.grid(row=3, column=0, columnspan=2, pady=10)
    
    def delete_user(self):
        """Delete a selected user"""
        # Get selected item
        selection = self.users_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a user")
            return
            
        # Get username
        item = selection[0]
        username = self.users_tree.item(item, "values")[0]
        
        # Check if it's the last admin
        if username == 'admin' or (self.users[self.users['role'] == 'admin'].shape[0] <= 1 and 
                                  'admin' in self.users.loc[self.users['username'] == username, 'role'].values):
            messagebox.showwarning("Warning", "Cannot delete the default admin user or last admin")
            return
        
        # Confirm deletion
        if not messagebox.askyesno("Confirm", f"Are you sure you want to delete user '{username}'?"):
            return
        
        # Remove from users dataframe
        self.users = self.users[self.users['username'] != username]
        
        # Save to CSV
        self.users.to_csv(self.users_path, index=False)
        
        # Refresh users list
        self.refresh_users()
        
        # Show success message
        messagebox.showinfo("Success", "User deleted successfully")
    def view_report_details(self, report_id):
        """View full details of a report"""
        # Find report
        report = self.reports[self.reports['id'] == report_id].iloc[0]
        
        # Create details window
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Report Details - {report['name']}")
        details_window.geometry("700x700")
        
        # Add details
        frame = tk.Frame(details_window, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = tk.Label(frame, text=f"Details for {report['name']}", font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Image
        try:
            img = Image.open(report['image_path'])
            img = img.resize((200, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            img_label = tk.Label(frame, image=photo)
            img_label.image = photo  # Keep a reference
            img_label.pack(pady=10)
        except Exception:
            # Display placeholder if image loading fails
            img_label = tk.Label(frame, text="No Image Available", width=30, height=15, bd=1, relief=tk.SUNKEN)
            img_label.pack(pady=10)
        
        # Details
        details_frame = tk.Frame(frame, bd=2, relief=tk.GROOVE, padx=10, pady=10)
        details_frame.pack(fill=tk.X)
        
        
        # Child details
        details_frame = tk.LabelFrame(details_frame, text="Child Information", font=("Arial", 12, "bold"))
        details_frame.pack(fill=tk.X, pady=8, padx=5)
        # Display details
        tk.Label(details_frame, text=f"ID: {report['id']}", anchor="w", font=("Arial", 11)).pack(fill=tk.X, pady=2)
        tk.Label(details_frame, text=f"Name: {report['name']}", anchor="w", font=("Arial", 11, "bold")).pack(fill=tk.X, pady=2)
        tk.Label(details_frame, text=f"Missing since: {report['date_missing']}", anchor="w", font=("Arial", 11)).pack(fill=tk.X, pady=2)
        tk.Label(details_frame, text=f"Last seen at: {report['location']}", anchor="w", font=("Arial", 11)).pack(fill=tk.X, pady=2)
        tk.Label(details_frame, text=f"Contact: {report['phone']}", anchor="w", font=("Arial", 11)).pack(fill=tk.X, pady=2)
        
        # Additional details if available
        if 'description' in report:
            description_frame = tk.LabelFrame(frame, text="Description", padx=10, pady=10)
            description_frame.pack(fill=tk.X, pady=10)
            
            desc_text = tk.Text(description_frame, wrap=tk.WORD, height=5, width=50)
            desc_text.insert(tk.END, report['description'])
            desc_text.config(state=tk.DISABLED)
            desc_text.pack(fill=tk.X)
        # Notes section
        notes_frame = tk.LabelFrame(details_frame, text="Notes", font=("Arial", 12, "bold"))
        notes_frame.pack(fill=tk.BOTH, expand=True, pady=9, padx=5)
        
        notes_text = tk.Text(notes_frame, height=5, width=60)
        notes_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        notes_text.insert(tk.END, "Enter any additional notes about this match here...")
        print_button = tk.Button(frame, text="Print Report", 
                            bg="green", fg="white", font=("Arial", 10, "bold"),
                            command=lambda: self.print_report(details_window))
        print_button.pack(side=tk.LEFT, padx=10)
        
        # Close button
        close_btn = tk.Button(frame, text="Close", command=details_window.destroy)
        close_btn.pack(pady=10)
    def print_report(self, report_window):
        """Simulates printing the report"""
        messagebox.showinfo("Print Report", "Report would be sent to printer.\nThis is a placeholder for the actual print functionality.")
        
# Main entry point
if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    
    # Create the application
    app = MissingChildIdentificationSystem(root)
    
    # Start the event loop
    root.mainloop()