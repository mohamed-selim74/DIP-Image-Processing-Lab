import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ModernImageApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DIP Interactive Lab")
        self.geometry("1450x950")

        self.original_image_cv = None
        self.current_image_cv = None
        self.temp_preview_img = None 
        self.history = []
        self.history_index = -1

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.setup_ui()

    def setup_ui(self):
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Image Filters", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.pack(pady=(20, 10))

        self.btn_browse = ctk.CTkButton(self.sidebar_frame, text="📁 Browse Image", command=self.browse_image)
        self.btn_browse.pack(pady=5, padx=20, fill="x")

        self.btn_reset = ctk.CTkButton(self.sidebar_frame, text="🔄 Reset", fg_color="#D32F2F", hover_color="#B71C1C", command=self.reset_image)
        self.btn_reset.pack(pady=5, padx=20, fill="x")

     
        self.undo_redo_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.undo_redo_frame.pack(pady=10, padx=20, fill="x")
        self.undo_redo_frame.grid_columnconfigure(0, weight=1)
        self.undo_redo_frame.grid_columnconfigure(1, weight=1)
        
        self.btn_undo = ctk.CTkButton(self.undo_redo_frame, text="⏪ Undo", command=self.undo, width=100)
        self.btn_undo.grid(row=0, column=0, padx=(0, 5))
        
        self.btn_redo = ctk.CTkButton(self.undo_redo_frame, text="Redo ⏩", command=self.redo, width=100)
        self.btn_redo.grid(row=0, column=1, padx=(5, 0))

       
        self.btn_auto = ctk.CTkButton(self.sidebar_frame, text="✨ Auto Enhance ✨", fg_color="#FF5722", hover_color="#E64A19", font=ctk.CTkFont(size=16, weight="bold"), command=self.apply_auto_enhance)
        self.btn_auto.pack(pady=(15, 10), padx=20, fill="x")

        self.add_filter_group("1. Low-Pass", "#4CAF50", [
            ("Mean Blur", self.apply_mean), 
            ("Median Blur", self.apply_median), 
            ("Gaussian Blur", self.apply_gaussian)
        ])
        
        self.add_filter_group("2. High-Pass", "#2196F3", [
            ("Laplacian", self.apply_laplacian), 
            ("Unsharp Masking", self.apply_unsharp),
            ("Sobel Edge", self.apply_sobel), 
            ("Canny Edge", self.apply_canny)
        ])
        
        self.add_filter_group("3. Morphology", "#FF9800", [
            ("Erosion", self.apply_erosion), 
            ("Dilation", self.apply_dilation), 
            ("Opening", self.apply_opening), 
            ("Closing", self.apply_closing)
        ])
        
        self.add_filter_group("4. Point Operations", "#9C27B0", [
            ("Image Negative", self.apply_negative), 
            ("Log Transform", self.apply_log),
            ("Gamma Correction", self.apply_gamma)
        ])
        
        self.add_filter_group("5. Histogram", "#E91E63", [
            ("Equalization", self.apply_hist_equalization)
        ])

        self.add_filter_group("6. Arithmetic Operations", "#00BCD4", [
            ("Add Value", lambda: self.arithmetic_op_academic("add")),
            ("Subtract Value", lambda: self.arithmetic_op_academic("sub")),
            ("Multiply Value", lambda: self.arithmetic_op_academic("mul")),
            ("Divide Value", lambda: self.arithmetic_op_academic("div"))
        ])
        
        self.add_filter_group("7. Extra Tools", "#607D8B", [
            ("Threshold", self.apply_simple_threshold) # أضفنا self هنا
        ])

        self.main_view = ctk.CTkFrame(self, fg_color="transparent")
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_view.grid_rowconfigure(1, weight=3) 
        self.main_view.grid_rowconfigure(2, weight=2) 
        self.main_view.grid_columnconfigure(0, weight=1)
        self.main_view.grid_columnconfigure(1, weight=1)

        self.param_frame = ctk.CTkFrame(self.main_view, fg_color="#1e1e1e", corner_radius=10)
        self.param_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 15))

        self.btn_apply = ctk.CTkButton(self.param_frame, text="✅ Apply", fg_color="#4CAF50", hover_color="#388E3C", font=ctk.CTkFont(weight="bold", size=15), width=100, height=40, command=self.apply_manual_adjustments)
        self.btn_apply.pack(side="right", padx=20, pady=10)

        self.arithmetic_frame = ctk.CTkFrame(self.param_frame, fg_color="transparent")
        self.arithmetic_frame.pack(side="right", padx=10, pady=10)

        self.arithmetic_label = ctk.CTkLabel(self.arithmetic_frame, text="Arithmetic Value:", font=ctk.CTkFont(weight="bold"))
        self.arithmetic_label.pack(side="left", padx=(0, 5))
        
        self.arithmetic_entry = ctk.CTkEntry(self.arithmetic_frame, width=80, placeholder_text="Value...")
        self.arithmetic_entry.pack(side="left")

        self.sliders = {}
        self.add_interactive_slider(self.param_frame, "Gamma", 0.1, 5.0, 1.0, "#9C27B0")
        self.add_interactive_slider(self.param_frame, "Saturation", 0.1, 3.0, 1.0, "#2196F3")
        self.add_interactive_slider(self.param_frame, "Kernel Size", 3, 15, 5, "#E91E63", is_int=True)

        self.orig_label = ctk.CTkLabel(self.main_view, text="Original Image", font=ctk.CTkFont(size=14, weight="bold"))
        self.orig_label.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        
        self.proc_label = ctk.CTkLabel(self.main_view, text="Processed Image", font=ctk.CTkFont(size=14, weight="bold"))
        self.proc_label.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="nsew")

        self.hist_frame = ctk.CTkFrame(self.main_view, fg_color="#1e1e1e", corner_radius=10)
        self.hist_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        self.fig, (self.ax_orig, self.ax_proc) = plt.subplots(1, 2, figsize=(10, 3), dpi=100)
        self.fig.patch.set_facecolor('#1e1e1e')
        for ax in [self.ax_orig, self.ax_proc]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.hist_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def add_interactive_slider(self, parent, title, f, t, d, color, is_int=False):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(side="left", expand=True, fill="x", padx=10, pady=10)
        lbl = ctk.CTkLabel(frame, text=f"{title}: {d}", font=ctk.CTkFont(weight="bold", size=12), text_color=color)
        lbl.pack()
        slider = ctk.CTkSlider(frame, from_=f, to=t, command=lambda v: self.on_slider_move(title, v, lbl, is_int))
        slider.set(d)
        slider.pack(pady=5, fill="x")
        self.sliders[title] = slider

    def on_slider_move(self, title, val, lbl, is_int):
        v = int(val) if is_int else round(float(val), 2)
        if is_int and v % 2 == 0: v += 1
        lbl.configure(text=f"{title}: {v}")
        self.live_preview()

    def live_preview(self):
        if self.current_image_cv is None: return
        gamma = self.sliders["Gamma"].get()
        sat = self.sliders["Saturation"].get()
        
        img = self.current_image_cv.copy()
        if gamma != 1.0:
            table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img = cv2.LUT(img, table)
        if sat != 1.0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
            h, s, v = cv2.split(hsv)
            s = np.clip(s * sat, 0, 255)
            img = cv2.cvtColor(cv2.merge([h, s, v]).astype("uint8"), cv2.COLOR_HSV2BGR)
        
        self.temp_preview_img = img
        self.update_display(preview=True)

    def apply_manual_adjustments(self):
        if self.current_image_cv is None: return
        gamma = self.sliders["Gamma"].get()
        sat = self.sliders["Saturation"].get()
        img = self.current_image_cv.copy()
        if gamma != 1.0:
            table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img = cv2.LUT(img, table)
        if sat != 1.0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
            h, s, v = cv2.split(hsv)
            s = np.clip(s * sat, 0, 255)
            img = cv2.cvtColor(cv2.merge([h, s, v]).astype("uint8"), cv2.COLOR_HSV2BGR)
        self.save_state(img)

    def add_filter_group(self, title, color, filters):
        lbl = ctk.CTkLabel(self.sidebar_frame, text=title, font=ctk.CTkFont(size=15, weight="bold"), text_color=color)
        lbl.pack(anchor="w", padx=20, pady=(15, 5))
        for name, cmd in filters:
            btn = ctk.CTkButton(self.sidebar_frame, text=name, command=cmd, fg_color="#333333", hover_color="#555555")
            btn.pack(pady=3, padx=20, fill="x")

    def save_state(self, new_img):
        self.history = self.history[:self.history_index + 1]
        self.history.append(new_img.copy())
        self.history_index += 1
        if len(self.history) > 15: self.history.pop(0); self.history_index -= 1
        self.current_image_cv = self.history[self.history_index].copy()
        self.sliders["Gamma"].set(1.0); self.sliders["Saturation"].set(1.0)
        self.temp_preview_img = None
        self.update_display()

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.current_image_cv = self.history[self.history_index].copy()
            self.update_display()

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_image_cv = self.history[self.history_index].copy()
            self.update_display()

    def browse_image(self):
        p = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.bmp *.tif *.jpeg")])
        if p:
            img_array = np.fromfile(p, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                self.original_image_cv = img.copy(); self.history = []; self.history_index = -1; self.save_state(img)

    def reset_image(self):
        if self.original_image_cv is not None: self.save_state(self.original_image_cv.copy())

    def update_display(self, preview=False):
        disp = self.temp_preview_img if preview else self.current_image_cv
        self.show_img(self.original_image_cv, self.orig_label)
        self.show_img(disp, self.proc_label)
        if not preview: self.update_histogram(disp)

    def show_img(self, img_cv, label):
        if img_cv is None: return
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(450, 350))
        label.configure(image=ctk_img, text="")

    def update_histogram(self, img_cv):
        if img_cv is None: return
        
        self.ax_orig.clear()
        self.ax_proc.clear()
        
        gray_orig = cv2.cvtColor(self.original_image_cv, cv2.COLOR_BGR2GRAY)
        hist_orig = cv2.calcHist([gray_orig], [0], None, [256], [0, 256])
        
        gray_proc = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        hist_proc = cv2.calcHist([gray_proc], [0], None, [256], [0, 256])
        
        self.ax_orig.plot(hist_orig, color='gray')
        self.ax_orig.set_title("Original Histogram", color='white', fontsize=10)
        self.ax_orig.set_xlim([0, 256])
        
        self.ax_proc.plot(hist_proc, color='#2196F3')
        self.ax_proc.set_title("Processed Histogram", color='white', fontsize=10)
        self.ax_proc.set_xlim([0, 256])
        
        self.canvas.draw()

    def apply_auto_enhance(self):
        if self.current_image_cv is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        img = self.current_image_cv.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        sat_val = self.sliders["Saturation"].get()

        noise_ratio = np.sum(cv2.absdiff(gray, cv2.medianBlur(gray, 3)) > 40) / gray.size
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        std_contrast = np.std(gray)
        mean_brightness = np.mean(gray)

        issues = []
        
        if noise_ratio > 0.03: 
            img = cv2.medianBlur(img, 3)
            issues.append("Noise")
            
        if laplacian_var < 100:
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
            issues.append("Blurriness")
            
        if mean_brightness > 200:
            img = cv2.convertScaleAbs(img, alpha=1.0, beta=-40)
            issues.append("Overexposure")
            
        if std_contrast < 30 or mean_brightness < 80:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            
            if sat_val != 1.0:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                h, s, v = cv2.split(hsv)
                s = np.clip(s * sat_val, 0, 255)
                img = cv2.cvtColor(cv2.merge((h, s, v)).astype(np.uint8), cv2.COLOR_HSV2BGR)
            issues.append("Low Contrast / Dark")
            
        if issues:
            messagebox.showinfo("Auto Enhance Result", f"✨ Detected & Fixed:\n\n{' + '.join(issues)}")
        else:
            messagebox.showinfo("Auto Enhance", "No major issues detected. Image looks good already!")
            
        self.save_state(img)

    def apply_mean(self):
        k = int(self.sliders["Kernel Size"].get());
        res = cv2.blur(self.current_image_cv,(k, k));
        self.save_state(res);

    def apply_gaussian(self):
        k = int(self.sliders["Kernel Size"].get());
        k = k if k%2!=0 else k+1
        res = cv2.GaussianBlur(self.current_image_cv, (k, k), 0); 
        self.save_state(res);

    def apply_median(self):
        k = int(self.sliders["Kernel Size"].get());
        k = k if k%2!=0 else k+1
        res = cv2.medianBlur(self.current_image_cv, k); 
        self.save_state(res);

    def apply_laplacian(self):
        gray = cv2.cvtColor(self.current_image_cv, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        res = cv2.convertScaleAbs(gray - lap)
        self.save_state(cv2.cvtColor(res, cv2.COLOR_GRAY2BGR))

    def apply_sobel(self):
        gray = cv2.cvtColor(self.current_image_cv, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0);
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1);
        res= cv2.magnitude(sx, sy)
        res = cv2.convertScaleAbs(cv2.magnitude(sx, sy))
        self.save_state(cv2.cvtColor(res, cv2.COLOR_GRAY2BGR))

    def apply_unsharp(self):
        blur = cv2.GaussianBlur(self.current_image_cv, (5,5), 0)
        mask = cv2.subtract(self.current_image_cv, blur)
        res = cv2.addWeighted(self.current_image_cv, 1.0, mask, 1.0, 0)
        self.save_state(res)

    def apply_canny(self):
        gray = cv2.cvtColor(self.current_image_cv, cv2.COLOR_BGR2GRAY)
        res = cv2.Canny(gray, 100, 200)
        self.save_state(cv2.cvtColor(res, cv2.COLOR_GRAY2BGR))

    def apply_negative(self):
        self.save_state(255 - self.current_image_cv)

    def apply_log(self):
        gray = cv2.cvtColor(self.current_image_cv, cv2.COLOR_BGR2GRAY).astype(np.float32)
        c = 255 / np.log(1 + np.max(gray))
        res = np.uint8(c * np.log(1 + gray))
        self.save_state(cv2.cvtColor(res, cv2.COLOR_GRAY2BGR))

    def apply_gamma(self):
        gamma = self.sliders["Gamma"].get()
        gray = cv2.cvtColor(self.current_image_cv, cv2.COLOR_BGR2GRAY)
        normalized = gray / 255.0
        res =np.power(normalized , gamma)
        res = np.uint8( res* 255)
        self.save_state(cv2.cvtColor(res, cv2.COLOR_GRAY2BGR))

    def apply_hist_equalization(self):
        gray = cv2.cvtColor(self.current_image_cv, cv2.COLOR_BGR2GRAY)
        res = cv2.equalizeHist(gray)
        self.save_state(cv2.cvtColor(res, cv2.COLOR_GRAY2BGR))

    def arithmetic_op_academic(self, op):
        if self.current_image_cv is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        try: 
            val = float(self.arithmetic_entry.get())
        except ValueError: 
            messagebox.showerror("Error", "Please enter a valid numeric value!")
            return
            
        img = self.current_image_cv.copy()
        if op == "add": res = cv2.add(img, np.array([val, val, val, 0]))
        elif op == "sub": res = cv2.subtract(img, np.array([val, val, val, 0]))
        else:
            img_f = img.astype(np.float32) / 255.0
            res_f = img_f * val if op == "mul" else img_f / (val if val!=0 else 0.01)
            res = (np.clip(res_f, 0, 1) * 255).astype(np.uint8)
        self.save_state(res)

    def apply_erosion(self):
        k = int(self.sliders["Kernel Size"].get())
        kernel = np.ones((k,k), np.uint8)
        erosion =cv2.erode(self.current_image_cv, kernel, iterations=1)
        self.save_state(erosion)

    def apply_dilation(self):
        k = int(self.sliders["Kernel Size"].get())
        kernel = np.ones((k,k), np.uint8)
        dilation =cv2.dilate(self.current_image_cv, kernel, iterations=1)
        self.save_state(dilation)

    def apply_opening(self):
        k = int(self.sliders["Kernel Size"].get())
        kernel = np.ones((k,k), np.uint8)
        opening = cv2.morphologyEx(self.current_image_cv, cv2.MORPH_OPEN, kernel)
        self.save_state(opening)

    def apply_closing(self):
        k = int(self.sliders["Kernel Size"].get())
        kernel = np.ones((k,k), np.uint8)
        closing= cv2.morphologyEx(self.current_image_cv, cv2.MORPH_CLOSE, kernel)
        self.save_state(closing)

    def apply_simple_threshold(self):
        if self.current_image_cv is None: return

        threshold_value = 127 

        gray = cv2.cvtColor(self.current_image_cv, cv2.COLOR_BGR2GRAY)

        res = np.where(gray > threshold_value, 255, 0).astype(np.uint8)

        res_bgr = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

        self.save_state(res_bgr)

        

if __name__ == "__main__":
    app = ModernImageApp()
    app.mainloop()
