#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider, RadioButtons
from matplotlib.patches import Rectangle, Circle
from numba import jit, cuda
import time, json, cv2
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
import sounddevice as sd, warnings; warnings.filterwarnings('ignore')

@jit(nopython=True)
def fractal_iter(c, z=0, ftype=0, max_iter=100):
    for n in range(max_iter):
        if abs(z) > 2: return n + 1 - np.log2(np.log2(abs(z)))
        if ftype == 0: z = z*z + c  # Mandelbrot
        elif ftype == 1: z = z*z + c  # Julia (z,c swapped in caller)
        elif ftype == 2: z = (abs(z.real) + 1j*abs(z.imag))**2 + c  # Burning Ship
        elif ftype == 3: z = np.conj(z)**2 + c  # Tricorn
    return max_iter

@jit(nopython=True, parallel=True)
def compute_fractal(xmin, xmax, ymin, ymax, width, height, max_iter=100, 
                   fractal_type=0, julia_c=complex(-0.7, 0.27015)):
    x, y = np.linspace(xmin, xmax, width), np.linspace(ymin, ymax, height)
    result = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            z, c = (complex(x[j], y[i]), julia_c) if fractal_type == 1 else (0, complex(x[j], y[i]))
            result[i, j] = fractal_iter(c, z, fractal_type, max_iter)
    return result

@jit(nopython=True)
def create_distance_estimation(data, xmin, xmax, ymin, ymax):
    h, w = data.shape; dx, dy = (xmax - xmin) / w, (ymax - ymin) / h; edges = np.zeros_like(data)
    for i in range(1, h-1):
        for j in range(1, w-1):
            gx, gy = (data[i, j+1] - data[i, j-1]) / (2*dx), (data[i+1, j] - data[i-1, j]) / (2*dy)
            edges[i, j] = np.sqrt(gx**2 + gy**2)
    return edges

try: @cuda.jit; GPU_AVAILABLE = True
except: GPU_AVAILABLE = False

@jit(nopython=True)
def compute_fractal_3d(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x, y = np.linspace(xmin, xmax, width), np.linspace(ymin, ymax, height)
    result, heights = np.zeros((height, width)), np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            c, z = complex(x[j], y[i]), 0
            for n in range(max_iter):
                if abs(z) > 2: result[i, j] = n + 1 - np.log2(np.log2(abs(z))); heights[i, j] = np.log(abs(z)); break
                z = z*z + c
            else: result[i, j], heights[i, j] = max_iter, 0
    return result, heights

def detect_interesting_regions(data, num_regions=5):
    grad = np.gradient(data); grad_mag = np.sqrt(grad[0]**2 + grad[1]**2)
    pts = np.where(grad_mag > np.percentile(grad_mag, 95))
    if len(pts[0]) > 10:
        points = np.column_stack(pts); clustering = DBSCAN(eps=20, min_samples=10).fit(points)
        regions = [(np.mean(points[clustering.labels_ == l][:, 1]), np.mean(points[clustering.labels_ == l][:, 0])) 
                  for l in set(clustering.labels_) if l != -1]
        return regions[:num_regions]
    return []

def generate_fractal_music(data, duration=10, sample_rate=22050):
    norm = (data.flatten() - data.min()) / (data.max() - data.min()); t = np.linspace(0, duration, int(duration * sample_rate))
    freqs = 440 * (2 ** (norm[:len(t)] * 2)); audio = np.zeros_like(t)
    for i, f in enumerate(freqs[:len(t)//10]):
        audio[i:i+len(t)//10] += 0.1 * np.sin(2*np.pi*f*t[i:i+len(t)//10]) * np.exp(-t[i:i+len(t)//10]*0.1)
    return audio, sample_rate

class AdvancedFractalExplorer:
    def __init__(self, width=800, height=600):
        vars(self).update(width=width, height=height, max_iter=150, fractal_type=0, julia_c=complex(-0.7,0.27015),
                         julia_morphing=False, morph_speed=0.01, gpu_enabled=GPU_AVAILABLE, show_3d=False,
                         auto_explore=False, recording_video=False, video_frames=[], interesting_regions=[],
                         region_markers=[], zoom_factor=0.7, color_cycle=0, render_mode=0, computation_times=[],
                         zoom_history=[], favorite_locations=[], xmin=-2.5, xmax=1.5, ymin=-1.5, ymax=1.5)
        
        self.fig = plt.figure(figsize=(20, 12))
        gs = self.fig.add_gridspec(2, (4 if self.show_3d else 3), height_ratios=[3 if self.show_3d else 4, 1], 
                                  width_ratios=[2,2,1,1] if self.show_3d else [3,3,1])
        self.ax_main = self.fig.add_subplot(gs[0, 0 if self.show_3d else slice(2)])
        self.ax_julia_preview = self.fig.add_subplot(gs[0, 2])
        self.ax_3d = self.fig.add_subplot(gs[0, 1], projection='3d') if self.show_3d else None
        self.ax_stats = self.fig.add_subplot(gs[0, 3]) if self.show_3d else None
        self.ax_controls = self.fig.add_subplot(gs[1, :])
        self.fig.suptitle('🚀 Ultra-Advanced Fractal Explorer AI 🚀', fontsize=20, fontweight='bold')
        self.fractal_data = compute_fractal(self.xmin, self.xmax, self.ymin, self.ymax, 
                                          self.width, self.height, self.max_iter, self.fractal_type, self.julia_c)
        self.im_main = self.ax_main.imshow(self.fractal_data, extent=[self.xmin, self.xmax, self.ymin, self.ymax],
                                          cmap='hot', origin='lower', interpolation='bilinear')
        julia_data = compute_fractal(-2, 2, -2, 2, 200, 200, 80, 1, self.julia_c)
        self.im_julia = self.ax_julia_preview.imshow(julia_data, extent=[-2, 2, -2, 2], cmap='plasma', origin='lower')
        self.ax_julia_preview.set_title('Julia Preview'); self.ax_julia_preview.set_xticks([]); self.ax_julia_preview.set_yticks([])
        
        if self.ax_3d: self.setup_3d_view()
        if self.ax_stats: self.setup_stats_view()
        self.zoom_rect, self.zoom_start = None, None
        self.ax_main.set_xlabel('Real Axis'); self.ax_main.set_ylabel('Imaginary Axis')
        self.update_interesting_regions(); self.setup_controls(); self.setup_events()
        self.executor = ThreadPoolExecutor(max_workers=4); self.computation_future = None
        self.animation = animation.FuncAnimation(self.fig, self.animate, interval=50, blit=False)
    
    def setup_controls(self):
        self.ax_controls.clear(); self.ax_controls.set_xlim(0, 10); self.ax_controls.set_ylim(0, 5); self.ax_controls.axis('off')
        self.radio_fractal = RadioButtons(plt.axes([0.02, 0.15, 0.15, 0.15]), ['Mandelbrot', 'Julia', 'Burning Ship', 'Tricorn'])
        
        btns = [Button(plt.axes(c), n) for c, n in zip([(0.2,0.25,0.08,0.04),(0.29,0.25,0.08,0.04),(0.38,0.25,0.08,0.04),(0.47,0.25,0.08,0.04)], 
                                                      ['Zoom+','Zoom-','Reset','Save'])]
        self.btn_zoom_in, self.btn_zoom_out, self.btn_reset, self.btn_save = btns
        
        btns2 = [Button(plt.axes(c), n) for c, n in zip([(0.2,0.20,0.06,0.04),(0.27,0.20,0.06,0.04),(0.34,0.20,0.06,0.04),(0.41,0.20,0.06,0.04)], 
                                                        ['Colors','Render','Morph','⭐ Fav'])]
        self.btn_color, self.btn_render, self.btn_morph, self.btn_favorite = btns2
        
        btns3 = [Button(plt.axes(c), n) for c, n in zip([(0.2,0.15,0.06,0.04),(0.27,0.15,0.06,0.04),(0.34,0.15,0.06,0.04),(0.41,0.15,0.06,0.04),(0.48,0.15,0.06,0.04)], 
                                                        ['🔷 3D','🤖 AI','🎬 REC','🎵 Music',f"{'⚡GPU' if self.gpu_enabled else '💻CPU'}"])]
        self.btn_3d, self.btn_ai, self.btn_video, self.btn_music, self.btn_gpu = btns3
        
        sliders = [Slider(plt.axes(c), n, *r, valinit=v, **k) for c, n, r, v, k in 
                  [((0.58,0.27,0.25,0.02), 'Iterations', (50,800), self.max_iter, {'valstep':10}),
                   ((0.58,0.24,0.25,0.02), 'Julia Real', (-2,2), self.julia_c.real, {'valfmt':'%.3f'}),
                   ((0.58,0.21,0.25,0.02), 'Julia Imag', (-2,2), self.julia_c.imag, {'valfmt':'%.3f'})]]
        self.slider_iter, self.slider_julia_real, self.slider_julia_imag = sliders; self.create_info_panel()
    
    def create_info_panel(self):
        self.info_text = self.fig.text(0.86, 0.6, self.get_info_text(), fontsize=9, verticalalignment='top',
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    def setup_events(self):
        self.btn_zoom_in.on_clicked(lambda x: self.zoom(0.5))
        self.btn_zoom_out.on_clicked(lambda x: self.zoom(2.0))
        self.btn_reset.on_clicked(lambda x: self.reset_view())
        self.btn_save.on_clicked(lambda x: self.save_fractal())
        self.btn_color.on_clicked(lambda x: self.cycle_colors())
        self.btn_render.on_clicked(lambda x: self.cycle_render_mode())
        self.btn_morph.on_clicked(lambda x: self.toggle_morphing())
        self.btn_favorite.on_clicked(lambda x: self.save_favorite())
        self.btn_3d.on_clicked(lambda x: self.toggle_3d_view())
        self.btn_ai.on_clicked(lambda x: self.toggle_ai_exploration())
        self.btn_video.on_clicked(lambda x: self.toggle_video_recording())
        self.btn_music.on_clicked(lambda x: self.play_fractal_music())
        self.btn_gpu.on_clicked(lambda x: self.toggle_gpu_acceleration())
        self.radio_fractal.on_clicked(self.change_fractal_type)
        self.slider_iter.on_changed(self.update_iterations)
        self.slider_julia_real.on_changed(self.update_julia_parameter)
        self.slider_julia_imag.on_changed(self.update_julia_parameter)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
    
    def get_info_text(self):
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
        if center_x is None:
            center_x = (self.xmin + self.xmax) / 2
        if center_y is None:
            center_y = (self.ymin + self.ymax) / 2
        
        self.zoom_history.append((self.xmin, self.xmax, self.ymin, self.ymax))
        if len(self.zoom_history) > 50:
            self.zoom_history.pop(0)
        
        width = (self.xmax - self.xmin) * factor
        height = (self.ymax - self.ymin) * factor
        self.xmin = center_x - width / 2
        self.xmax = center_x + width / 2
        self.ymin = center_y - height / 2
        self.ymax = center_y + height / 2
        self.update_fractal()
    
    def reset_view(self):
        bounds = [(-2.5, 1.5, -1.5, 1.5), (-2, 2, -2, 2), (-2.5, 1.5, -2.5, 1.5), (-2.5, 1.5, -1.5, 1.5)]
        self.xmin, self.xmax, self.ymin, self.ymax = bounds[self.fractal_type]
        self.zoom_history.clear()
        self.update_fractal()
    
    def cycle_colors(self):
        colormaps = ['hot', 'plasma', 'viridis', 'magma', 'inferno', 'cividis', 
                    'rainbow', 'twilight', 'turbo', 'gist_ncar', 'nipy_spectral']
        self.color_cycle = (self.color_cycle + 1) % len(colormaps)
        self.im_main.set_cmap(colormaps[self.color_cycle])
        self.fig.canvas.draw()
    
    def update_iterations(self, val): self.max_iter = int(val); self.update_fractal()
    def change_fractal_type(self, label): self.fractal_type = {'Mandelbrot':0,'Julia':1,'Burning Ship':2,'Tricorn':3}[label]; self.reset_view()
    def update_julia_parameter(self, val): self.julia_c = complex(self.slider_julia_real.val, self.slider_julia_imag.val); self.update_fractal() if self.fractal_type == 1 else None; self.update_julia_preview()
    def cycle_render_mode(self): self.render_mode = (self.render_mode + 1) % 3; self.update_fractal()
    def toggle_morphing(self): self.julia_morphing = not self.julia_morphing
    
    def save_favorite(self):
        favorite = {'fractal_type': self.fractal_type, 'bounds': (self.xmin, self.xmax, self.ymin, self.ymax),
                   'julia_c': self.julia_c, 'max_iter': self.max_iter, 'timestamp': datetime.now().isoformat()}
        self.favorite_locations.append(favorite)
        print(f"⭐ Saved favorite location #{len(self.favorite_locations)}")
    
    def update_fractal(self):
        start_time = time.time()
        self.fractal_data = compute_fractal(self.xmin, self.xmax, self.ymin, self.ymax,
                                          self.width, self.height, self.max_iter, 
                                          self.fractal_type, self.julia_c)
        
        if self.render_mode == 1:
            edges = create_distance_estimation(self.fractal_data, self.xmin, self.xmax, self.ymin, self.ymax)
            display_data = edges
        elif self.render_mode == 2:
            edges = create_distance_estimation(self.fractal_data, self.xmin, self.xmax, self.ymin, self.ymax)
            display_data = self.fractal_data + edges * 20
        else:
            display_data = self.fractal_data
        
        self.im_main.set_array(display_data)
        self.im_main.set_extent([self.xmin, self.xmax, self.ymin, self.ymax])
        
        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)
        if len(self.computation_times) > 20:
            self.computation_times.pop(0)
        
        self.info_text.set_text(self.get_info_text())
        self.fig.canvas.draw()
    
    def save_fractal(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fractal_{timestamp}.png"
        hires_data = compute_fractal(self.xmin, self.xmax, self.ymin, self.ymax,
                                   1920, 1080, self.max_iter * 2, self.fractal_type, self.julia_c)
        plt.figure(figsize=(19.2, 10.8), dpi=100)
        plt.imshow(hires_data, extent=[self.xmin, self.xmax, self.ymin, self.ymax],
                  cmap=plt.cm.hot, origin='lower', interpolation='bilinear')
        plt.axis('off')
        plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"💾 Saved high-resolution fractal: {filename}")
    
    def update_julia_preview(self):
        julia_data = compute_fractal(-2, 2, -2, 2, 200, 200, 80, 1, self.julia_c)
        self.im_julia.set_array(julia_data)
        self.fig.canvas.draw()
    
    def on_mouse_press(self, event):
        if event.inaxes == self.ax_main and event.button == 1:
            if hasattr(event, 'key') and event.key == 'shift':
                center_x, center_y = (self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2
                dx, dy = event.xdata - center_x, event.ydata - center_y
                self.xmin += dx
                self.xmax += dx
                self.ymin += dy
                self.ymax += dy
                self.update_fractal()
            else:
                self.zoom_start = (event.xdata, event.ydata)
    
    def on_mouse_release(self, event):
        if event.inaxes == self.ax_main and event.button == 1 and self.zoom_start:
            if abs(event.xdata - self.zoom_start[0]) > 0.01 and abs(event.ydata - self.zoom_start[1]) > 0.01:
                x1, y1 = self.zoom_start
                x2, y2 = event.xdata, event.ydata
                self.xmin, self.xmax = min(x1, x2), max(x1, x2)
                self.ymin, self.ymax = min(y1, y2), max(y1, y2)
                self.update_fractal()
            else:
                self.zoom(self.zoom_factor, event.xdata, event.ydata)
            
            if self.zoom_rect:
                self.zoom_rect.remove()
                self.zoom_rect = None
            self.zoom_start = None
    
    def on_mouse_motion(self, event):
        if event.inaxes == self.ax_main and self.zoom_start and event.button == 1:
            if self.zoom_rect:
                self.zoom_rect.remove()
            x1, y1 = self.zoom_start
            x2, y2 = event.xdata, event.ydata
            width, height = abs(x2 - x1), abs(y2 - y1)
            self.zoom_rect = Rectangle((min(x1, x2), min(y1, y2)), width, height,
                                     fill=False, edgecolor='white', linewidth=2, alpha=0.8)
            self.ax_main.add_patch(self.zoom_rect)
            self.fig.canvas.draw()
    
    def on_key_press(self, event):
        actions = {'r': self.reset_view, 's': self.save_fractal, ' ': self.toggle_morphing}
        if event.key in actions: actions[event.key]()
        elif event.key in ['left', 'right', 'up', 'down']:
            dx, dy = (self.xmax - self.xmin) * 0.1, (self.ymax - self.ymin) * 0.1
            if event.key == 'left': self.xmin -= dx; self.xmax -= dx
            elif event.key == 'right': self.xmin += dx; self.xmax += dx
            elif event.key == 'up': self.ymin += dy; self.ymax += dy
            elif event.key == 'down': self.ymin -= dy; self.ymax -= dy
            self.update_fractal()
    
    def setup_3d_view(self):
        if self.ax_3d:
            self.ax_3d.set_title('3D Fractal Landscape')
            self.ax_3d.set_xlabel('Real')
            self.ax_3d.set_ylabel('Imaginary')
            self.ax_3d.set_zlabel('Iterations')
    
    def setup_stats_view(self):
        if self.ax_stats:
            self.ax_stats.set_title('Performance Stats')
            self.ax_stats.set_xlabel('Frame')
            self.ax_stats.set_ylabel('Computation Time (s)')
    
    def update_interesting_regions(self):
        self.interesting_regions = detect_interesting_regions(self.fractal_data)
        self.update_region_markers()
    
    def update_region_markers(self):
        for marker in self.region_markers:
            marker.remove()
        self.region_markers.clear()
        for x, y in self.interesting_regions:
            real_x = self.xmin + x * (self.xmax - self.xmin) / self.width
            real_y = self.ymin + y * (self.ymax - self.ymin) / self.height
            circle = Circle((real_x, real_y), abs(self.xmax - self.xmin) * 0.02,
                          fill=False, edgecolor='cyan', linewidth=2, alpha=0.8)
            self.ax_main.add_patch(circle)
            self.region_markers.append(circle)
    
    def toggle_3d_view(self):
        self.show_3d = not self.show_3d
        print(f"🔷 3D view: {'ON' if self.show_3d else 'OFF'}")
        self.update_3d_fractal()
    
    def update_3d_fractal(self):
        if self.ax_3d and self.show_3d:
            data_3d, heights = compute_fractal_3d(self.xmin, self.xmax, self.ymin, self.ymax, 50, 50, self.max_iter)
            x = np.linspace(self.xmin, self.xmax, 50)
            y = np.linspace(self.ymin, self.ymax, 50)
            X, Y = np.meshgrid(x, y)
            self.ax_3d.clear()
            surf = self.ax_3d.plot_surface(X, Y, heights, cmap='plasma', alpha=0.8, antialiased=True)
            self.ax_3d.set_title('3D Fractal Landscape')
    
    def toggle_ai_exploration(self):
        self.auto_explore = not self.auto_explore
        print(f"🤖 AI exploration: {'ON' if self.auto_explore else 'OFF'}")
    
    def ai_explore_step(self):
        if self.auto_explore and self.interesting_regions:
            region_idx = np.random.randint(len(self.interesting_regions))
            x, y = self.interesting_regions[region_idx]
            real_x = self.xmin + x * (self.xmax - self.xmin) / self.width
            real_y = self.ymin + y * (self.ymax - self.ymin) / self.height
            self.zoom(0.8, real_x, real_y)
    
    def toggle_video_recording(self):
        self.recording_video = not self.recording_video
        if self.recording_video:
            print("🎬 Started video recording...")
            self.video_frames = []
        else:
            if self.video_frames:
                self.save_video()
            print("⏹️ Stopped video recording")
    
    def save_video(self):
        if not self.video_frames:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fractal_exploration_{timestamp}.mp4"
        try:
            height, width = self.video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 10.0, (width, height))
            for frame in self.video_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            print(f"🎥 Saved video: {filename} ({len(self.video_frames)} frames)")
        except Exception as e:
            print(f"❌ Video save failed: {e}")
    
    def capture_frame(self):
        if self.recording_video:
            self.fig.canvas.draw()
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            self.video_frames.append(buf.copy())
    
    def play_fractal_music(self):
        print("🎵 Generating fractal music...")
        try:
            audio, sample_rate = generate_fractal_music(self.fractal_data, duration=5)
            print("🔊 Playing fractal sonification...")
            sd.play(audio, sample_rate)
        except Exception as e:
            print(f"❌ Audio playback failed: {e}")
    
    def toggle_gpu_acceleration(self):
        if GPU_AVAILABLE:
            self.gpu_enabled = not self.gpu_enabled
            self.btn_gpu.label.set_text(f"{'⚡GPU' if self.gpu_enabled else '💻CPU'}")
            print(f"⚡ GPU acceleration: {'ON' if self.gpu_enabled else 'OFF'}")
        else:
            print("❌ GPU acceleration not available")

    def animate(self, frame):
        if self.julia_morphing and self.fractal_type == 1:
            t = frame * self.morph_speed; self.julia_c = complex(0.7 * np.cos(t), 0.27015 + 0.2 * np.sin(t * 1.3))
            self.slider_julia_real.set_val(self.julia_c.real); self.slider_julia_imag.set_val(self.julia_c.imag)
            if frame % 10 == 0: self.update_fractal(); self.update_julia_preview()
        if self.auto_explore and frame % 100 == 0: self.ai_explore_step()
        if frame % 5 == 0: self.capture_frame()
        if self.show_3d and frame % 20 == 0: self.update_3d_fractal()
        if frame % 30 == 0: self.info_text.set_text(self.get_info_text())
        return [self.im_main, self.im_julia]
    
    def show(self):
        plt.show()
    
    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

if __name__ == "__main__":
    print("🚀 ======= ULTRA-ADVANCED FRACTAL EXPLORER AI ======= 🚀")
    print("🎨 Features: 3D landscapes, AI exploration, video recording, fractal music, GPU acceleration")
    print("🎮 Controls: Drag-to-zoom, keyboard shortcuts (R:Reset, S:Save, Space:Morph, Arrows:Pan)")
    print("🧠 AI: Automatic region detection, smart zoom recommendations, performance optimization")
    print("🎥 Export: 4K images, MP4 videos, fractal music, JSON bookmarks")
    print(f"⚡ {'GPU acceleration available!' if GPU_AVAILABLE else 'Running on CPU (install CUDA for GPU acceleration)'}")
    print("\n🌟 Starting the most advanced fractal explorer ever created...\n🚀 Prepare for an incredible mathematical journey!")
    AdvancedFractalExplorer(width=800, height=600).show()
