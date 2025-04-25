import wx
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Corrected Gaussian function
def Gaussian(img, ker_size, k):
    var = ker_size / 6
    kernel = np.zeros((ker_size, ker_size), dtype=np.float32)
    center = ker_size // 2
    for s in range(ker_size):
        for t in range(ker_size):
            r2 = (s - center) ** 2 + (t - center) ** 2
            kernel[s, t] = k * np.exp(-r2 / (2 * var ** 2))
    kernel /= np.sum(kernel)
    pad = ker_size // 2
    padded = np.pad(img, pad, 'reflect')
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(padded[i:i+ker_size, j:j+ker_size] * kernel)
    return out

# Local histogram equalization on patches
def local_his_eq(img, ker_size):
    pad = ker_size // 2
    padded = np.pad(img, pad, 'reflect')
    result = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            patch = padded[i:i+ker_size, j:j+ker_size]
            cdf = cv2.calcHist([patch], [0], None, [256], [0, 256]).cumsum()
            cdf_normalized = cdf * 255 / cdf[-1]
            result[i, j] = cdf_normalized[patch[ker_size//2, ker_size//2]]
    return result

class ImageProcessingGUI(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(1000, 700),
                         style=wx.DEFAULT_FRAME_STYLE | wx.RESIZE_BORDER)
        self.original_img = None
        self.proc_img = None
        self.setup_ui()
        self.Show()

    def setup_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(self, label="Load an image and choose processing options."), 0, wx.ALL, 5)

        # Image display panels
        disp_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.before_bmp = wx.StaticBitmap(self, size=(200, 200))
        self.after_bmp = wx.StaticBitmap(self, size=(200, 200))
        for lbl, bmp in [("Before", self.before_bmp), ("After", self.after_bmp)]:
            box = wx.StaticBox(self, label=lbl)
            box_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
            box_sizer.Add(bmp, 1, wx.EXPAND | wx.ALL, 5)
            disp_sizer.Add(box_sizer, 0, wx.ALL, 5)
        sizer.Add(disp_sizer, 0, wx.CENTER)

        # Buttons
        btns = [
            ("Load Image", self.load_image),
            ("Enhance", self.enhance),
            ("Edge Detect", self.edge_detect),
            ("Threshold", self.threshold),
            ("Gray & Enhance", self.gray_enhance),
            ("Sharpen (Laplacian)", self.sharpen_laplacian),
            ("Custom Gaussian Blur", self.custom_gaussian),
            ("Local Histogram Equalization", self.local_hist_eq),
        ]

        # Calculate rows for 2 columns
        num_buttons = len(btns)
        cols = 2
        rows = (num_buttons + cols - 1) // cols  # Ensures enough rows

        btn_sizer = wx.GridSizer(rows, cols, 10, 10)
        for label, handler in btns:
            btn = wx.Button(self, label=label)
            btn.Bind(wx.EVT_BUTTON, handler)
            btn_sizer.Add(btn, 0, wx.EXPAND)
        sizer.Add(btn_sizer, 0, wx.ALL | wx.CENTER, 10)

        self.SetSizer(sizer)

    def display_image(self, img, bmp_widget):
        if img is None:
            return
        if len(img.shape) == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img.copy()

        # Resize to fit display
        h, w = img_bgr.shape[:2]
        max_dim = 200
        aspect_ratio = w / h
        if aspect_ratio > 1:
            new_w = max_dim
            new_h = int(max_dim / aspect_ratio)
        else:
            new_h = max_dim
            new_w = int(max_dim * aspect_ratio)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Centered in 200x200 canvas
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        y_off = (200 - new_h) // 2
        x_off = (200 - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        bmp = wx.Bitmap.FromBufferRGBA(width, height, cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA).tobytes())
        bmp_widget.SetBitmap(bmp)
        self.Layout()

    def load_image(self, event=None):
        with wx.FileDialog(self, "Open Image", "", "",
                           "Image Files (*.png;*.jpg;*.jpeg)|*.png;*.jpg;*.jpeg",
                           wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                img = cv2.imread(path)
                if img is not None:
                    self.original_img = img
                    self.proc_img = None
                    self.display_image(self.original_img, self.before_bmp)
                    self.display_image(self.original_img, self.after_bmp)

    def get_current_img(self):
        return self.proc_img if self.proc_img is not None else self.original_img

    def enhance(self, event):
        img = self.get_current_img()
        if img is None:
            wx.MessageBox("Load an image first.", "Error")
            return
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape)==2 else img.copy()
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(img_bgr, -1, kernel)
        blurred = cv2.GaussianBlur(img_bgr, (9, 9), 0)
        combined = cv2.addWeighted(sharpened, 0.4, cv2.addWeighted(img_bgr, 1.5, blurred, -0.5, 0), 0.6, 0)
        hsv = cv2.cvtColor(combined, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_eq = cv2.equalizeHist(v)
        final = cv2.cvtColor(cv2.merge([h, s, v_eq]), cv2.COLOR_HSV2BGR)
        self.proc_img = final
        self.display_image(self.proc_img, self.after_bmp)

    def edge_detect(self, event):
        img = self.get_current_img()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        edges = cv2.Canny(gray, 100, 200)
        self.proc_img = edges
        self.display_image(self.proc_img, self.after_bmp)

    def threshold(self, event):
        img = self.get_current_img()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        self.proc_img = thresh
        self.display_image(self.proc_img, self.after_bmp)

    def gray_enhance(self, event):
        img = self.get_current_img()
        if img is None:
            wx.MessageBox("Load an image first.", "Error")
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        self.proc_img = cv2.equalizeHist(gray)
        self.display_image(self.proc_img, self.after_bmp)

    def sharpen_laplacian(self, event):
        img = self.get_current_img()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharp = cv2.convertScaleAbs(gray + lap)
        self.proc_img = sharp
        self.display_image(self.proc_img, self.after_bmp)

    def custom_gaussian(self, event):
        img = self.get_current_img()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        blurred = Gaussian(gray, 43, 1)
        self.proc_img = cv2.cvtColor(blurred.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.display_image(self.proc_img, self.after_bmp)

    def local_hist_eq(self, event):
        img = self.get_current_img()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        result = local_his_eq(gray, 5)
        self.proc_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        self.display_image(self.proc_img, self.after_bmp)

if __name__ == '__main__':
    app = wx.App()
    frame = ImageProcessingGUI(None, 'Image Processing GUI')
    app.MainLoop()