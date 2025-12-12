#!/usr/bin/env python3
"""
Image Denoising Desktop Application

A GUI application for automatic noise detection and image denoising using
trained Random Forest classifier and optimal filtering algorithms.

Usage:
    python denoise_app.py
"""

import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import numpy as np
import threading
import time

# Add paths for imports
sys.path.append(str(Path(__file__).parent / 'noise_detecting'))
sys.path.append(str(Path(__file__).parent / 'denoise'))

from noise_detecting.detect_noise import detect_noise
from denoise.denoise_image import apply_denoise_filter


class DenoiseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Denoising Application - ML-Based Noise Detection")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')
        
        # State variables
        self.noisy_image_path = None
        self.noisy_image = None
        self.denoised_image = None
        self.detected_noise = None
        self.processing = False
        
        # Create UI
        self.create_ui()
        
        # Bind window resize to refresh images
        self.root.bind('<Configure>', self.on_window_resize)
        self.last_resize_time = 0
        
    def create_ui(self):
        """Create the user interface"""
        
        # Title bar
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill=tk.X, side=tk.TOP)
        
        title_label = tk.Label(
            title_frame,
            text="🔬 Image Denoising Application",
            font=('Arial', 20, 'bold'),
            bg='#2c3e50',
            fg='white',
            pady=10
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="ML-Based Automatic Noise Detection & Optimal Filtering",
            font=('Arial', 10),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        subtitle_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls and Info
        left_panel = tk.Frame(main_frame, bg='white', relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5), pady=0)
        left_panel.config(width=350)
        left_panel.pack_propagate(False)
        
        # Control buttons
        control_frame = tk.LabelFrame(
            left_panel,
            text="Controls",
            font=('Arial', 12, 'bold'),
            bg='white',
            padx=15,
            pady=15
        )
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.load_btn = tk.Button(
            control_frame,
            text="📁 Load Noisy Image",
            command=self.load_image,
            font=('Arial', 11, 'bold'),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=2,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.load_btn.pack(fill=tk.X, pady=5)
        
        self.denoise_btn = tk.Button(
            control_frame,
            text="🔧 Detect & Denoise",
            command=self.denoise_image,
            font=('Arial', 11, 'bold'),
            bg='#2ecc71',
            fg='white',
            activebackground='#27ae60',
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=2,
            padx=20,
            pady=10,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.denoise_btn.pack(fill=tk.X, pady=5)
        
        self.save_btn = tk.Button(
            control_frame,
            text="💾 Save Denoised Image",
            command=self.save_image,
            font=('Arial', 11, 'bold'),
            bg='#9b59b6',
            fg='white',
            activebackground='#8e44ad',
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=2,
            padx=20,
            pady=10,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.save_btn.pack(fill=tk.X, pady=5)
        
        # Progress bar
        self.progress_frame = tk.Frame(control_frame, bg='white')
        self.progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_label = tk.Label(
            self.progress_frame,
            text="",
            font=('Arial', 9),
            bg='white',
            fg='#7f8c8d'
        )
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode='indeterminate',
            length=300
        )
        
        # Detection info
        info_frame = tk.LabelFrame(
            left_panel,
            text="Detection Information",
            font=('Arial', 12, 'bold'),
            bg='white',
            padx=15,
            pady=15
        )
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Noise type display
        noise_label_frame = tk.Frame(info_frame, bg='white')
        noise_label_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            noise_label_frame,
            text="Detected Noise:",
            font=('Arial', 10, 'bold'),
            bg='white'
        ).pack(anchor=tk.W)
        
        self.noise_type_label = tk.Label(
            noise_label_frame,
            text="—",
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#e74c3c'
        )
        self.noise_type_label.pack(anchor=tk.W, pady=5)
        
        # Filter applied
        filter_label_frame = tk.Frame(info_frame, bg='white')
        filter_label_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            filter_label_frame,
            text="Applied Filter:",
            font=('Arial', 10, 'bold'),
            bg='white'
        ).pack(anchor=tk.W)
        
        self.filter_type_label = tk.Label(
            filter_label_frame,
            text="—",
            font=('Arial', 11),
            bg='white',
            fg='#2c3e50'
        )
        self.filter_type_label.pack(anchor=tk.W, pady=5)
        
        # Image info
        tk.Label(
            info_frame,
            text="Image Information:",
            font=('Arial', 10, 'bold'),
            bg='white'
        ).pack(anchor=tk.W, pady=(15, 5))
        
        self.info_text = tk.Text(
            info_frame,
            height=8,
            font=('Courier', 9),
            bg='#ecf0f1',
            relief=tk.FLAT,
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Noise type descriptions
        desc_frame = tk.LabelFrame(
            left_panel,
            text="Noise Types",
            font=('Arial', 10, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        desc_frame.pack(fill=tk.X, padx=10, pady=10)
        
        noise_descriptions = {
            'Gaussian': 'Normal distribution noise',
            'Salt & Pepper': 'Random white/black pixels',
            'Speckle': 'Multiplicative noise',
            'Uniform': 'Random uniform noise'
        }
        
        for noise, desc in noise_descriptions.items():
            row = tk.Frame(desc_frame, bg='white')
            row.pack(fill=tk.X, pady=2)
            tk.Label(
                row,
                text=f"• {noise}:",
                font=('Arial', 8, 'bold'),
                bg='white',
                width=15,
                anchor=tk.W
            ).pack(side=tk.LEFT)
            tk.Label(
                row,
                text=desc,
                font=('Arial', 8),
                bg='white',
                fg='#7f8c8d',
                anchor=tk.W
            ).pack(side=tk.LEFT)
        
        # Right panel - Image display
        right_panel = tk.Frame(main_frame, bg='#f0f0f0')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Image frames
        image_container = tk.Frame(right_panel, bg='#f0f0f0')
        image_container.pack(fill=tk.BOTH, expand=True)
        
        # Noisy image
        noisy_frame = tk.LabelFrame(
            image_container,
            text="Noisy Image",
            font=('Arial', 12, 'bold'),
            bg='white',
            relief=tk.RAISED,
            borderwidth=2
        )
        noisy_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.noisy_canvas = tk.Canvas(
            noisy_frame,
            bg='#ecf0f1',
            highlightthickness=0
        )
        self.noisy_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Denoised image
        denoised_frame = tk.LabelFrame(
            image_container,
            text="Denoised Image",
            font=('Arial', 12, 'bold'),
            bg='white',
            relief=tk.RAISED,
            borderwidth=2
        )
        denoised_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.denoised_canvas = tk.Canvas(
            denoised_frame,
            bg='#ecf0f1',
            highlightthickness=0
        )
        self.denoised_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready. Load an image to begin.",
            font=('Arial', 9),
            bg='#34495e',
            fg='white',
            anchor=tk.W,
            padx=10
        )
        self.status_label.pack(fill=tk.X)
        
    def load_image(self):
        """Load a noisy image"""
        file_path = filedialog.askopenfilename(
            title="Select Noisy Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.noisy_image_path = Path(file_path)
            # Load image in original format (keep color if it's color)
            self.noisy_image = Image.open(file_path)
            
            # Convert to RGB if it's a palette or other mode
            if self.noisy_image.mode not in ['RGB', 'L']:
                self.noisy_image = self.noisy_image.convert('RGB')
            
            # Reset state
            self.denoised_image = None
            self.detected_noise = None
            
            # Display noisy image
            self.display_image(self.noisy_image, self.noisy_canvas)
            
            # Clear denoised canvas
            self.denoised_canvas.delete("all")
            self.denoised_canvas.create_text(
                400, 300,
                text="No denoised image yet\nClick 'Detect & Denoise'",
                font=('Arial', 12),
                fill='#95a5a6',
                justify=tk.CENTER
            )
            
            # Update UI
            self.denoise_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.DISABLED)
            self.noise_type_label.config(text="—")
            self.filter_type_label.config(text="—")
            
            # Update info
            self.update_image_info()
            
            self.status_label.config(text=f"Loaded: {self.noisy_image_path.name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            
    def display_image(self, pil_image, canvas):
        """Display PIL image on canvas"""
        canvas.delete("all")
        
        # Safety check
        if pil_image is None:
            canvas.create_text(
                400, 300,
                text="Error: No image to display",
                font=('Arial', 12),
                fill='#e74c3c',
                justify=tk.CENTER
            )
            return
        
        # Force canvas update to get correct size
        canvas.update_idletasks()
        canvas_width = max(canvas.winfo_width(), 400)
        canvas_height = max(canvas.winfo_height(), 400)
        
        # Resize image to fit canvas while maintaining aspect ratio
        img_width, img_height = pil_image.size
        scale = min((canvas_width - 20) / img_width, (canvas_height - 20) / img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Ensure minimum size
        if new_width < 50 or new_height < 50:
            new_width = min(400, img_width)
            new_height = min(400, img_height)
        
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(resized_image)
        
        # Center on canvas
        x = canvas_width // 2
        y = canvas_height // 2
        
        canvas.create_image(x, y, image=photo, anchor=tk.CENTER)
        
        # Keep reference to prevent garbage collection
        if not hasattr(canvas, 'images'):
            canvas.images = []
        canvas.images.append(photo)
        
    def denoise_image(self):
        """Detect noise and denoise the image"""
        if self.noisy_image_path is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        # Start processing in background thread
        thread = threading.Thread(target=self._denoise_worker, daemon=True)
        thread.start()
        
    def _denoise_worker(self):
        """Background worker for denoising"""
        try:
            # Update UI - start
            self.root.after(0, self._processing_start)
            
            # Check if image is color or grayscale
            is_color = self.noisy_image.mode == 'RGB'
            
            # Step 1: Detect noise type
            self.root.after(0, lambda: self.progress_label.config(text="Detecting noise type..."))
            time.sleep(0.3)
            
            # Use ML-based detection (trained Random Forest model)
            noise_type = detect_noise(str(self.noisy_image_path), use_ml=True)
            self.detected_noise = noise_type
            
            self.root.after(0, lambda: self.noise_type_label.config(text=noise_type.upper()))
            
            # Step 2: Apply denoising filter
            self.root.after(0, lambda: self.progress_label.config(text="Applying optimal filter..."))
            time.sleep(0.3)
            
            if is_color:
                # Process each color channel separately
                self.root.after(0, lambda: self.progress_label.config(text="Denoising RGB channels..."))
                denoised_image = self._denoise_color_image(str(self.noisy_image_path), noise_type)
            else:
                # Process grayscale image
                denoised_array = apply_denoise_filter(str(self.noisy_image_path), noise_type, verbose=False)
                
                if denoised_array is None:
                    raise Exception("Denoising failed")
                
                # Convert to PIL Image
                denoised_uint8 = (np.clip(denoised_array, 0, 1) * 255).astype(np.uint8)
                denoised_image = Image.fromarray(denoised_uint8)
            
            if denoised_image is None:
                raise Exception("Denoising produced no output")
            
            self.denoised_image = denoised_image
            
            # Update UI - complete
            self.root.after(0, self._processing_complete)
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: self._processing_error(msg))
            
    def _processing_start(self):
        """Update UI when processing starts"""
        self.processing = True
        self.load_btn.config(state=tk.DISABLED)
        self.denoise_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.progress_bar.pack(fill=tk.X, pady=5)
        self.progress_bar.start(10)
        self.status_label.config(text="Processing...")
        
    def _processing_complete(self):
        """Update UI when processing completes"""
        self.processing = False
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.progress_label.config(text="")
        
        # Check if denoising was successful
        if self.denoised_image is None:
            self._processing_error("Denoising returned no image")
            return
        
        # Display denoised image
        self.display_image(self.denoised_image, self.denoised_canvas)
        
        # Update filter info
        filter_map = {
            'gaussian': 'Non-Local Means / Bilateral',
            'salt_pepper': 'Adaptive Median Filter',
            'speckle': 'NLM in Log Domain + Detail Restoration',
            'uniform': 'Bilateral + Multi-Stage Enhancement',
            'jpeg_artifact': 'Bilateral Filter',
            'clean': 'No filtering applied'
        }
        
        filter_name = filter_map.get(self.detected_noise, 'Unknown')
        self.filter_type_label.config(text=filter_name)
        
        # Update info
        self.update_image_info()
        
        # Enable buttons
        self.load_btn.config(state=tk.NORMAL)
        self.denoise_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        
        self.status_label.config(text=f"Denoising complete! Detected: {self.detected_noise}")
        
    def _denoise_color_image(self, img_path, noise_type):
        """Denoise a color image by processing each channel separately"""
        import tempfile
        import os
        
        # Load color image
        color_img = Image.open(img_path).convert('RGB')
        r, g, b = color_img.split()
        
        denoised_channels = []
        
        for i, channel in enumerate([r, g, b]):
            # Save channel as temporary grayscale image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                channel.save(tmp_path)
            
            try:
                # Denoise the channel
                denoised_array = apply_denoise_filter(tmp_path, noise_type, verbose=False)
                
                if denoised_array is None:
                    raise Exception(f"Failed to denoise channel {i}")
                
                # Convert to uint8
                denoised_uint8 = (np.clip(denoised_array, 0, 1) * 255).astype(np.uint8)
                denoised_channels.append(Image.fromarray(denoised_uint8))
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
        
        # Merge channels back into color image
        denoised_color = Image.merge('RGB', denoised_channels)
        return denoised_color
    
    def _processing_error(self, error_msg):
        """Update UI when processing fails"""
        self.processing = False
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.progress_label.config(text="")
        
        self.load_btn.config(state=tk.NORMAL)
        self.denoise_btn.config(state=tk.NORMAL)
        
        self.status_label.config(text="Error occurred during processing")
        messagebox.showerror("Processing Error", f"Denoising failed:\n{error_msg}")
        
    def update_image_info(self):
        """Update image information display"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        if self.noisy_image is not None:
            width, height = self.noisy_image.size
            mode = self.noisy_image.mode
            color_type = "Color (RGB)" if mode == 'RGB' else "Grayscale"
            
            info = f"Filename: {self.noisy_image_path.name}\n"
            info += f"Type: {color_type}\n"
            info += f"Dimensions: {width} × {height}\n"
            info += f"Total Pixels: {width * height:,}\n"
            
            if self.denoised_image is not None:
                # Calculate improvement metrics
                noisy_arr = np.array(self.noisy_image).astype(np.float64) / 255.0
                denoised_arr = np.array(self.denoised_image).astype(np.float64) / 255.0
                
                # Noise removed (difference)
                noise_removed = np.abs(noisy_arr - denoised_arr)
                avg_noise = np.mean(noise_removed)
                
                info += f"\n--- Denoising Results ---\n"
                info += f"Avg. Noise Removed: {avg_noise:.4f}\n"
                info += f"Noise %: {avg_noise*100:.2f}%\n"
                
                if mode == 'RGB':
                    info += f"\nNote: Denoised per RGB channel"
        else:
            info = "No image loaded"
            
        self.info_text.insert(1.0, info)
        self.info_text.config(state=tk.DISABLED)
        
    def save_image(self):
        """Save the denoised image"""
        if self.denoised_image is None:
            messagebox.showwarning("Warning", "No denoised image to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Denoised Image",
            defaultextension=".png",
            initialfile=f"denoised_{self.noisy_image_path.stem}.png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.denoised_image.save(file_path)
            self.status_label.config(text=f"Saved: {Path(file_path).name}")
            messagebox.showinfo("Success", f"Image saved successfully to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")


    def on_window_resize(self, event):
        """Redraw images when window is resized"""
        import time
        current_time = time.time()
        
        # Debounce resize events (only redraw every 0.2 seconds)
        if current_time - self.last_resize_time < 0.2:
            return
        
        self.last_resize_time = current_time
        
        # Redraw images if they exist
        if self.noisy_image is not None:
            self.root.after(100, lambda: self.display_image(self.noisy_image, self.noisy_canvas))
        
        if self.denoised_image is not None:
            self.root.after(100, lambda: self.display_image(self.denoised_image, self.denoised_canvas))


def main():
    root = tk.Tk()
    app = DenoiseApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
