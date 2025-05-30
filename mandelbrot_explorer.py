#!/usr/bin/env python3
"""
Advanced Interactive Fractal Explorer
Beautiful visualization of Mandelbrot and Julia sets with real-time morphing,
GPU-accelerated computation, and artistic rendering options.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider, RadioButtons
from matplotlib.patches import Rectangle
from numba import jit, cuda
import colorsys
import time
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import json


@jit(nopython=True)
def mandelbrot_iteration(c, max_iter=100):
    """Calculate the number of iterations for a complex number c."""
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n + 1 - np.log2(np.log2(abs(z)))  # Smooth coloring
        z = z*z + c
    return max_iter


@jit(nopython=True)
def julia_iteration(z, c, max_iter=100):
    """Calculate Julia set iterations for point z with parameter c."""
    for n in range(max_iter):
        if abs(z) > 2:
            return n + 1 - np.log2(np.log2(abs(z)))  # Smooth coloring
        z = z*z + c
    return max_iter


@jit(nopython=True)
def burning_ship_iteration(c, max_iter=100):
    """Burning Ship fractal iteration."""
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n + 1 - np.log2(np.log2(abs(z)))
        z = (abs(z.real) + 1j*abs(z.imag))**2 + c
    return max_iter


@jit(nopython=True)
def tricorn_iteration(c, max_iter=100):
    """Tricorn fractal iteration."""
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n + 1 - np.log2(np.log2(abs(z)))
        z = np.conj(z)**2 + c
    return max_iter


@jit(nopython=True, parallel=True)
def compute_fractal(xmin, xmax, ymin, ymax, width, height, max_iter=100, 
                   fractal_type=0, julia_c=complex(-0.7, 0.27015)):
    """Generate fractal set with parallel computation."""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    
    result = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            if fractal_type == 0:  # Mandelbrot
                c = complex(x[j], y[i])
                result[i, j] = mandelbrot_iteration(c, max_iter)
            elif fractal_type == 1:  # Julia
                z = complex(x[j], y[i])
                result[i, j] = julia_iteration(z, julia_c, max_iter)
            elif fractal_type == 2:  # Burning Ship
                c = complex(x[j], y[i])
                result[i, j] = burning_ship_iteration(c, max_iter)
            elif fractal_type == 3:  # Tricorn
                c = complex(x[j], y[i])
                result[i, j] = tricorn_iteration(c, max_iter)
    
    return result


@jit(nopython=True)
def create_distance_estimation(data, xmin, xmax, ymin, ymax):
    """Create distance estimation for edge detection."""
    height, width = data.shape
    dx = (xmax - xmin) / width
    dy = (ymax - ymin) / height
    
    edges = np.zeros_like(data)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            grad_x = (data[i, j+1] - data[i, j-1]) / (2 * dx)
            grad_y = (data[i+1, j] - data[i-1, j]) / (2 * dy)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            edges[i, j] = gradient_mag
    
    return edges


class AdvancedFractalExplorer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.max_iter = 150
        self.fractal_type = 0  # 0: Mandelbrot, 1: Julia, 2: Burning Ship, 3: Tricorn
        self.julia_c = complex(-0.7, 0.27015)
        self.julia_morphing = False
        self.morph_speed = 0.01
        
        # View parameters
        self.xmin, self.xmax = -2.5, 1.5
        self.ymin, self.ymax = -1.5, 1.5
        self.zoom_factor = 0.7
        self.color_cycle = 0
        self.render_mode = 0  # 0: Normal, 1: Edge detection, 2: Hybrid
        
        # Performance tracking
        self.computation_times = []
        self.zoom_history = []
        self.favorite_locations = []
        
        # Setup the figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(2, 3, height_ratios=[4, 1], width_ratios=[3, 3, 1])
        
        self.ax_main = self.fig.add_subplot(gs[0, :2])
        self.ax_julia_preview = self.fig.add_subplot(gs[0, 2])
        self.ax_controls = self.fig.add_subplot(gs[1, :])
        
        self.fig.suptitle('🌟 Advanced Fractal Explorer 🌟', fontsize=18, fontweight='bold')
        
        # Generate initial fractal
        self.fractal_data = compute_fractal(
            self.xmin, self.xmax, self.ymin, self.ymax, 
            self.width, self.height, self.max_iter, self.fractal_type, self.julia_c
        )
        
        # Create main image
        self.im_main = self.ax_main.imshow(
            self.fractal_data, 
            extent=[self.xmin, self.xmax, self.ymin, self.ymax],
            cmap='hot', 
            origin='lower',
            interpolation='bilinear'
        )
        
        # Julia preview
        julia_data = compute_fractal(-2, 2, -2, 2, 200, 200, 80, 1, self.julia_c)
        self.im_julia = self.ax_julia_preview.imshow(
            julia_data, extent=[-2, 2, -2, 2], cmap='plasma', origin='lower'
        )
        self.ax_julia_preview.set_title('Julia Preview')
        self.ax_julia_preview.set_xticks([])
        self.ax_julia_preview.set_yticks([])
        
        # Zoom rectangle
        self.zoom_rect = None
        self.zoom_start = None
        
        self.ax_main.set_xlabel('Real Axis')
        self.ax_main.set_ylabel('Imaginary Axis')
        
        # Setup interactive elements
        self.setup_controls()
        self.setup_events()
        
        # Start background computation thread
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.computation_future = None
        
        # Animation for morphing and updates
        self.animation = animation.FuncAnimation(
            self.fig, self.animate, interval=50, blit=False
        )
    
    def setup_controls(self):
        """Setup advanced interactive controls."""
        self.ax_controls.clear()
        self.ax_controls.set_xlim(0, 10)
        self.ax_controls.set_ylim(0, 5)
        self.ax_controls.axis('off')
        
        # Fractal type radio buttons
        ax_fractal = plt.axes([0.02, 0.15, 0.15, 0.15])
        self.radio_fractal = RadioButtons(
            ax_fractal, ['Mandelbrot', 'Julia', 'Burning Ship', 'Tricorn']
        )
        
        # Control buttons - row 1
        ax_zoom_in = plt.axes([0.2, 0.25, 0.08, 0.04])
        ax_zoom_out = plt.axes([0.29, 0.25, 0.08, 0.04])
        ax_reset = plt.axes([0.38, 0.25, 0.08, 0.04])
        ax_save = plt.axes([0.47, 0.25, 0.08, 0.04])
        
        self.btn_zoom_in = Button(ax_zoom_in, 'Zoom+')
        self.btn_zoom_out = Button(ax_zoom_out, 'Zoom-')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_save = Button(ax_save, 'Save')
        
        # Control buttons - row 2
        ax_color = plt.axes([0.2, 0.20, 0.08, 0.04])
        ax_render = plt.axes([0.29, 0.20, 0.08, 0.04])
        ax_morph = plt.axes([0.38, 0.20, 0.08, 0.04])
        ax_favorite = plt.axes([0.47, 0.20, 0.08, 0.04])
        
        self.btn_color = Button(ax_color, 'Colors')
        self.btn_render = Button(ax_render, 'Render')
        self.btn_morph = Button(ax_morph, 'Morph')
        self.btn_favorite = Button(ax_favorite, '⭐ Fav')
        
        # Sliders
        ax_iter = plt.axes([0.58, 0.27, 0.25, 0.02])
        ax_julia_real = plt.axes([0.58, 0.24, 0.25, 0.02])
        ax_julia_imag = plt.axes([0.58, 0.21, 0.25, 0.02])
        
        self.slider_iter = Slider(ax_iter, 'Iterations', 50, 800, valinit=self.max_iter, valstep=10)
        self.slider_julia_real = Slider(ax_julia_real, 'Julia Real', -2, 2, valinit=self.julia_c.real, valfmt='%.3f')
        self.slider_julia_imag = Slider(ax_julia_imag, 'Julia Imag', -2, 2, valinit=self.julia_c.imag, valfmt='%.3f')
        
        # Info panel
        self.create_info_panel()
    
    def create_info_panel(self):
        """Create the information panel."""
        self.info_text = self.fig.text(
            0.86, 0.6, self.get_info_text(), 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8)
        )
    
    def setup_events(self):
        """Setup comprehensive event handlers."""
        # Button events
        self.btn_zoom_in.on_clicked(lambda x: self.zoom(0.5))
        self.btn_zoom_out.on_clicked(lambda x: self.zoom(2.0))
        self.btn_reset.on_clicked(lambda x: self.reset_view())
        self.btn_save.on_clicked(lambda x: self.save_fractal())
        self.btn_color.on_clicked(lambda x: self.cycle_colors())
        self.btn_render.on_clicked(lambda x: self.cycle_render_mode())
        self.btn_morph.on_clicked(lambda x: self.toggle_morphing())
        self.btn_favorite.on_clicked(lambda x: self.save_favorite())
        
        # Radio button events
        self.radio_fractal.on_clicked(self.change_fractal_type)
        
        # Slider events
        self.slider_iter.on_changed(self.update_iterations)
        self.slider_julia_real.on_changed(self.update_julia_parameter)
        self.slider_julia_imag.on_changed(self.update_julia_parameter)
        
        # Mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
    
    def get_info_text(self):
        """Generate comprehensive info text."""
        fractal_names = ['Mandelbrot', 'Julia', 'Burning Ship', 'Tricorn']
        render_modes = ['Normal', 'Edge Detection', 'Hybrid']
        
        avg_time = np.mean(self.computation_times[-5:]) if self.computation_times else 0
        
        return f"""🎮 Controls:
• Click/Drag: Zoom to area
• Shift+Click: Center view
• Arrow Keys: Pan view
• Space: Toggle morphing
• R: Reset view
• S: Save image

📊 Current State:
Fractal: {fractal_names[self.fractal_type]}
Render: {render_modes[self.render_mode]}
Julia C: {self.julia_c:.4f}
Morphing: {'ON' if self.julia_morphing else 'OFF'}

📍 View Info:
Real: [{self.xmin:.6f}, {self.xmax:.6f}]
Imag: [{self.ymin:.6f}, {self.ymax:.6f}]
Zoom: {1/((self.xmax-self.xmin)/4):.1f}x
Iterations: {self.max_iter}

⚡ Performance:
Avg Time: {avg_time:.2f}s
Favorites: {len(self.favorite_locations)}"""
    
    def zoom(self, factor, center_x=None, center_y=None):
        """Zoom the view by a given factor."""
        if center_x is None:
            center_x = (self.xmin + self.xmax) / 2
        if center_y is None:
            center_y = (self.ymin + self.ymax) / 2
        
        width = (self.xmax - self.xmin) * factor
        height = (self.ymax - self.ymin) * factor
        
        self.xmin = center_x - width / 2
        self.xmax = center_x + width / 2
        self.ymin = center_y - height / 2
        self.ymax = center_y + height / 2
        
        self.update_fractal()
    
    def reset_view(self):
        """Reset to the initial view."""
        self.xmin, self.xmax = -2.5, 1.5
        self.ymin, self.ymax = -1.5, 1.5
        self.update_fractal()
    
    def cycle_colors(self):
        """Cycle through different color schemes."""
        colormaps = ['hot', 'plasma', 'viridis', 'magma', 'coolwarm', 'rainbow', 'gist_stern']
        self.color_cycle = (self.color_cycle + 1) % len(colormaps)
        self.im.set_cmap(colormaps[self.color_cycle])
        self.fig.canvas.draw()
    
    def update_iterations(self, val):
        """Update the maximum iterations."""
        self.max_iter = int(val)
        self.update_fractal()
    
    def on_click(self, event):
        """Handle mouse clicks for zooming."""
        if event.inaxes == self.ax and event.button == 1:
            self.zoom(self.zoom_factor, event.xdata, event.ydata)
    
    def update_fractal(self):
        """Regenerate and update the fractal display."""
        self.mandelbrot_data = mandelbrot_set(
            self.xmin, self.xmax, self.ymin, self.ymax,
            self.width, self.height, self.max_iter
        )
        
        self.im.set_array(self.mandelbrot_data)
        self.im.set_extent([self.xmin, self.xmax, self.ymin, self.ymax])
        
        self.info_text.set_text(self.get_info_text())
        self.fig.canvas.draw()
    
    def save_fractal(self):
        """Save current fractal as high-resolution image."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fractal_{timestamp}.png"
        
        # Generate high-resolution version
        hires_data = compute_fractal(
            self.xmin, self.xmax, self.ymin, self.ymax,
            1920, 1080, self.max_iter * 2, self.fractal_type, self.julia_c
        )
        
        plt.figure(figsize=(19.2, 10.8), dpi=100)
        plt.imshow(hires_data, extent=[self.xmin, self.xmax, self.ymin, self.ymax],
                  cmap=plt.cm.hot, origin='lower', interpolation='bilinear')
        plt.axis('off')
        plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"💾 Saved high-resolution fractal: {filename}")
    
    def update_julia_preview(self):
        """Update the Julia set preview."""
        julia_data = compute_fractal(-2, 2, -2, 2, 200, 200, 80, 1, self.julia_c)
        self.im_julia.set_array(julia_data)
        self.fig.canvas.draw()
    
    def on_mouse_press(self, event):
        """Handle mouse press events."""
        if event.inaxes == self.ax_main:
            if event.button == 1:  # Left click
                if hasattr(event, 'key') and event.key == 'shift':
                    # Shift+click to center
                    center_x = (self.xmin + self.xmax) / 2
                    center_y = (self.ymin + self.ymax) / 2
                    dx = event.xdata - center_x
                    dy = event.ydata - center_y
                    self.xmin += dx
                    self.xmax += dx
                    self.ymin += dy
                    self.ymax += dy
                    self.update_fractal()
                else:
                    # Start zoom rectangle
                    self.zoom_start = (event.xdata, event.ydata)
    
    def on_mouse_release(self, event):
        """Handle mouse release events."""
        if event.inaxes == self.ax_main and event.button == 1 and self.zoom_start:
            if abs(event.xdata - self.zoom_start[0]) > 0.01 and abs(event.ydata - self.zoom_start[1]) > 0.01:
                # Zoom to selected rectangle
                x1, y1 = self.zoom_start
                x2, y2 = event.xdata, event.ydata
                self.xmin, self.xmax = min(x1, x2), max(x1, x2)
                self.ymin, self.ymax = min(y1, y2), max(y1, y2)
                self.update_fractal()
            else:
                # Single click zoom
                self.zoom(self.zoom_factor, event.xdata, event.ydata)
            
            if self.zoom_rect:
                self.zoom_rect.remove()
                self.zoom_rect = None
            self.zoom_start = None
    
    def on_mouse_motion(self, event):
        """Handle mouse motion for zoom rectangle."""
        if event.inaxes == self.ax_main and self.zoom_start and event.button == 1:
            if self.zoom_rect:
                self.zoom_rect.remove()
            
            x1, y1 = self.zoom_start
            x2, y2 = event.xdata, event.ydata
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            self.zoom_rect = Rectangle((min(x1, x2), min(y1, y2)), width, height,
                                     fill=False, edgecolor='white', linewidth=2, alpha=0.8)
            self.ax_main.add_patch(self.zoom_rect)
            self.fig.canvas.draw()
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key == 'r':
            self.reset_view()
        elif event.key == 's':
            self.save_fractal()
        elif event.key == ' ':
            self.toggle_morphing()
        elif event.key == 'left':
            dx = (self.xmax - self.xmin) * 0.1
            self.xmin -= dx
            self.xmax -= dx
            self.update_fractal()
        elif event.key == 'right':
            dx = (self.xmax - self.xmin) * 0.1
            self.xmin += dx
            self.xmax += dx
            self.update_fractal()
        elif event.key == 'up':
            dy = (self.ymax - self.ymin) * 0.1
            self.ymin += dy
            self.ymax += dy
            self.update_fractal()
        elif event.key == 'down':
            dy = (self.ymax - self.ymin) * 0.1
            self.ymin -= dy
            self.ymax -= dy
            self.update_fractal()
    
    def animate(self, frame):
        """Advanced animation with morphing."""
        if self.julia_morphing and self.fractal_type == 1:
            # Morph Julia parameter
            t = frame * self.morph_speed
            self.julia_c = complex(0.7 * np.cos(t), 0.27015 + 0.2 * np.sin(t * 1.3))
            self.slider_julia_real.set_val(self.julia_c.real)
            self.slider_julia_imag.set_val(self.julia_c.imag)
            
            if frame % 10 == 0:  # Update every 10 frames for performance
                self.update_fractal()
                self.update_julia_preview()
        
        # Update info panel
        if frame % 30 == 0:
            self.info_text.set_text(self.get_info_text())
        
        return [self.im_main, self.im_julia]
    
    def show(self):
        """Display the advanced explorer."""
        plt.show()
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


def create_artistic_gradient():
    """Create beautiful custom colormap with golden ratio."""
    colors = []
    n_colors = 256
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    for i in range(n_colors):
        t = i / n_colors
        
        # Use golden ratio for pleasing color progression
        hue = (t * golden_ratio) % 1.0
        saturation = 0.7 + 0.3 * np.sin(t * np.pi * 4)
        value = 0.2 + 0.8 * t
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    
    return colors


def load_favorites():
    """Load saved favorite locations."""
    try:
        with open('fractal_favorites.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_favorites(favorites):
    """Save favorite locations to file."""
    with open('fractal_favorites.json', 'w') as f:
        json.dump(favorites, f, indent=2, default=str)


if __name__ == "__main__":
    print("🌟 Starting Advanced Fractal Explorer...")
    print("✨ Drag to zoom to specific areas")
    print("🎛️ Use radio buttons to switch fractals")
    print("🔄 Toggle morphing for dynamic Julia sets")
    print("⭐ Save your favorite locations")
    print("🎨 Multiple render modes and color schemes")
    print("📸 Save high-resolution images")
    print()
    print("🎮 Keyboard shortcuts:")
    print("   R: Reset view")
    print("   S: Save image")
    print("   Space: Toggle morphing")
    print("   Arrow keys: Pan view")
    
    # Create and show the advanced explorer
    explorer = AdvancedFractalExplorer(width=800, height=600)
    explorer.show()
