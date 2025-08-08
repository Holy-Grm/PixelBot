import cv2
import numpy as np
import pyautogui
import time
import random
import math
import threading
import keyboard
import os
import json
from PIL import Image, ImageGrab
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from dataclasses import dataclass
from typing import List, Tuple, Optional
import queue

@dataclass
class Template:
    """Classe pour stocker les informations d'un template"""
    name: str
    image_path: str
    enabled: bool = True
    threshold: float = 0.8
    priority: int = 1
    click_offset_x: int = 0
    click_offset_y: int = 0

class Mouse:
    """Classe pour gérer les mouvements naturels de la souris"""
    def __init__(self):
        self.move_speed = 0.1
        pyautogui.FAILSAFE = True
    
    def bezier_curve(self, start, end, control1=None, control2=None, steps=50):
        """Génère une courbe de Bézier pour un mouvement plus naturel"""
        if control1 is None:
            # Points de contrôle aléatoires pour courbe naturelle
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            control1 = (
                mid_x + random.randint(-50, 50),
                mid_y + random.randint(-50, 50)
            )
            control2 = (
                mid_x + random.randint(-30, 30),
                mid_y + random.randint(-30, 30)
            )
        
        points = []
        for t in np.linspace(0, 1, steps):
            x = (1-t)**3 * start[0] + 3*(1-t)**2*t * control1[0] + \
                3*(1-t)*t**2 * control2[0] + t**3 * end[0]
            y = (1-t)**3 * start[1] + 3*(1-t)**2*t * control1[1] + \
                3*(1-t)*t**2 * control2[1] + t**3 * end[1]
            points.append((int(x), int(y)))
        return points
    
    def move_to(self, x, y, use_bezier=True):
        """Déplacer vers une position avec mouvement naturel"""
        current_x, current_y = pyautogui.position()
        
        # Ajouter une légère imprécision à la cible
        x += random.randint(-2, 2)
        y += random.randint(-2, 2)
        
        if use_bezier:
            # Utiliser courbe de Bézier pour mouvement naturel
            points = self.bezier_curve((current_x, current_y), (x, y))
            for point in points:
                pyautogui.moveTo(point[0], point[1], duration=0)
                time.sleep(random.uniform(0.001, 0.003))
        else:
            # Mouvement standard avec point intermédiaire
            delta_x = x - current_x
            delta_y = y - current_y
            distance = math.sqrt(delta_x * delta_x + delta_y * delta_y)
            
            if distance > 100:
                rand_x = random.randint(-15, 15)
                rand_y = random.randint(-15, 15)
                mid_x = current_x + (delta_x * 0.6) + rand_x
                mid_y = current_y + (delta_y * 0.6) + rand_y
                pyautogui.moveTo(mid_x, mid_y, duration=self.move_speed)
                self.random_pause(0.03, 0.1)
            
            pyautogui.moveTo(x, y, duration=self.move_speed)
        
        self.random_pause(0.05, 0.15)
    
    def click(self, x, y):
        """Clic simple avec temps de maintien variable"""
        self.move_to(x, y)
        pyautogui.mouseDown()
        self.random_pause(0.05, 0.15)  # Temps de maintien variable
        pyautogui.mouseUp()
        self.random_pause(0.1, 0.3)
    
    def shift_click(self, x, y):
        """Clic avec Shift"""
        self.move_to(x, y)
        pyautogui.keyDown('shift')
        time.sleep(random.uniform(0.02, 0.05))
        pyautogui.mouseDown()
        self.random_pause(0.05, 0.15)
        pyautogui.mouseUp()
        time.sleep(random.uniform(0.02, 0.05))
        pyautogui.keyUp('shift')
        self.random_pause(0.1, 0.3)
    
    def random_pause(self, min_time, max_time):
        """Pause aléatoire avec distribution gaussienne"""
        mean = (min_time + max_time) / 2
        std = (max_time - min_time) / 4
        pause_time = np.random.normal(mean, std)
        pause_time = max(min_time, min(max_time, pause_time))
        time.sleep(pause_time)

class ImageDetector:
    """Classe pour la détection d'images"""
    def __init__(self):
        self.templates: List[Template] = []
        self.detection_region = None  # (x, y, width, height)
        self.last_detection_time = {}
        self.detection_cooldown = 2.0  # Secondes entre détections du même objet
    
    def add_template(self, template: Template):
        """Ajouter un template à détecter"""
        if os.path.exists(template.image_path):
            self.templates.append(template)
            return True
        return False
    
    def set_detection_region(self, x, y, width, height):
        """Définir la région de détection"""
        self.detection_region = (x, y, width, height)
    
    def capture_screen(self) -> np.ndarray:
        """Capturer l'écran ou une région spécifique"""
        if self.detection_region:
            x, y, w, h = self.detection_region
            screenshot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        else:
            screenshot = ImageGrab.grab()
        
        # Convertir en array numpy pour OpenCV
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        return screenshot
    
    def detect_template(self, screenshot: np.ndarray, template: Template) -> List[Tuple[int, int]]:
        """Détecter un template dans une capture d'écran"""
        if not template.enabled:
            return []
        
        # Vérifier le cooldown
        current_time = time.time()
        if template.name in self.last_detection_time:
            if current_time - self.last_detection_time[template.name] < self.detection_cooldown:
                return []
        
        # Charger le template
        template_img = cv2.imread(template.image_path)
        if template_img is None:
            return []
        
        # Template matching
        result = cv2.matchTemplate(screenshot, template_img, cv2.TM_CCOEFF_NORMED)
        
        # Trouver toutes les correspondances au-dessus du seuil
        locations = np.where(result >= template.threshold)
        matches = []
        
        h, w = template_img.shape[:2]
        
        # Convertir en liste de coordonnées
        for pt in zip(*locations[::-1]):
            center_x = pt[0] + w // 2 + template.click_offset_x
            center_y = pt[1] + h // 2 + template.click_offset_y
            
            # Ajuster si on a une région de détection
            if self.detection_region:
                center_x += self.detection_region[0]
                center_y += self.detection_region[1]
            
            matches.append((center_x, center_y))
        
        # Supprimer les doublons proches
        if matches:
            matches = self.remove_close_duplicates(matches, min_distance=30)
            self.last_detection_time[template.name] = current_time
        
        return matches
    
    def remove_close_duplicates(self, points: List[Tuple[int, int]], min_distance: int = 30) -> List[Tuple[int, int]]:
        """Supprimer les points trop proches les uns des autres"""
        if not points:
            return []
        
        filtered = [points[0]]
        for point in points[1:]:
            is_duplicate = False
            for filtered_point in filtered:
                distance = math.sqrt((point[0] - filtered_point[0])**2 + 
                                   (point[1] - filtered_point[1])**2)
                if distance < min_distance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(point)
        
        return filtered
    
    def detect_all(self) -> List[Tuple[Template, List[Tuple[int, int]]]]:
        """Détecter tous les templates actifs"""
        screenshot = self.capture_screen()
        results = []
        
        # Trier par priorité
        sorted_templates = sorted(self.templates, key=lambda t: t.priority, reverse=True)
        
        for template in sorted_templates:
            matches = self.detect_template(screenshot, template)
            if matches:
                results.append((template, matches))
        
        return results

class PixelBot:
    """Bot principal avec détection d'image"""
    def __init__(self):
        self.mouse = Mouse()
        self.detector = ImageDetector()
        self.is_running = False
        self.shift_mode = True  # Mode Shift pour activer
        self.auto_click_mode = False  # Mode auto-click alternatif
        self.config_file = "bot_config.json"
        self.templates_folder = "templates"
        self.setup_folders()
        self.setup_hotkeys()
        self.action_queue = queue.Queue()
        self.pending_actions = []  # Actions en attente
        self.last_shift_state = False
        self.stats = {
            'clicks': 0,
            'detections': 0,
            'start_time': None
        }
    
    def setup_folders(self):
        """Créer les dossiers nécessaires"""
        if not os.path.exists(self.templates_folder):
            os.makedirs(self.templates_folder)
    
    def setup_hotkeys(self):
        """Configurer les raccourcis clavier"""
        keyboard.add_hotkey('f1', self.toggle_bot)
        keyboard.add_hotkey('f2', self.stop_bot)
        keyboard.add_hotkey('f10', self.capture_template)
    
    def toggle_bot(self):
        """Démarrer/Arrêter le bot"""
        if self.is_running:
            self.stop_bot()
        else:
            self.start_bot()
    
    def start_bot(self):
        """Démarrer le bot"""
        if self.is_running:
            return
        
        if self.shift_mode:
            print("\n🟢 Bot démarré en mode SHIFT!")
            print("📌 Maintenez SHIFT pour activer les clics")
            print("Le bot analyse en arrière-plan et clique quand vous maintenez Shift")
        else:
            print("\n🟢 Bot démarré en mode AUTO!")
            print("Le bot clique automatiquement sur les éléments détectés")
        
        print("F1 : Toggle Bot | F2 : Arrêt | F10 : Capturer template")
        
        self.is_running = True
        self.stats['start_time'] = time.time()
        self.pending_actions = []
        
        # Thread de détection (toujours actif)
        detection_thread = threading.Thread(target=self.detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        
        # Thread d'action (contrôlé par Shift ou automatique)
        action_thread = threading.Thread(target=self.action_loop)
        action_thread.daemon = True
        action_thread.start()
        
        # Thread de monitoring Shift
        if self.shift_mode:
            shift_monitor_thread = threading.Thread(target=self.shift_monitor)
            shift_monitor_thread.daemon = True
            shift_monitor_thread.start()
    
    def stop_bot(self):
        """Arrêter le bot"""
        self.is_running = False
        print("\n🔴 Bot arrêté!")
        self.print_stats()
    
    def detection_loop(self):
        """Boucle de détection (tourne toujours en arrière-plan)"""
        while self.is_running:
            try:
                # Détecter tous les templates
                detections = self.detector.detect_all()
                
                if detections:
                    # Vider les anciennes détections
                    self.pending_actions = []
                    
                    # Ajouter les nouvelles détections
                    for template, matches in detections:
                        for match in matches:
                            self.pending_actions.append((template, match))
                            self.stats['detections'] += 1
                            
                            if self.shift_mode:
                                # En mode Shift, on affiche juste qu'on a détecté
                                print(f"👁️ Détecté: {template.name} à ({match[0]}, {match[1]}) - En attente de SHIFT")
                            else:
                                # En mode auto, on ajoute à la queue
                                self.action_queue.put((template, match))
                                print(f"🎯 Détecté: {template.name} à ({match[0]}, {match[1]})")
                
                # Pause entre détections
                time.sleep(random.uniform(0.3, 0.6))
                
            except Exception as e:
                print(f"❌ Erreur détection: {e}")
    
    def shift_monitor(self):
        """Surveille l'état de la touche Shift"""
        while self.is_running and self.shift_mode:
            try:
                shift_pressed = keyboard.is_pressed('shift')
                
                # Détection du changement d'état
                if shift_pressed and not self.last_shift_state:
                    print("⚡ SHIFT activé - Exécution des actions en attente...")
                    # Ajouter les actions en attente à la queue
                    for action in self.pending_actions:
                        self.action_queue.put(action)
                    self.pending_actions = []
                elif not shift_pressed and self.last_shift_state:
                    print("💤 SHIFT relâché - En attente...")
                    # Vider la queue si on relâche Shift
                    while not self.action_queue.empty():
                        try:
                            self.action_queue.get_nowait()
                        except:
                            break
                
                self.last_shift_state = shift_pressed
                time.sleep(0.05)  # Check rapide pour réactivité
                
            except Exception as e:
                print(f"❌ Erreur monitoring Shift: {e}")
    
    def action_loop(self):
        """Boucle d'exécution des actions"""
        while self.is_running:
            try:
                # En mode Shift, on n'exécute que si Shift est pressé
                if self.shift_mode and not keyboard.is_pressed('shift'):
                    time.sleep(0.05)
                    continue
                
                # Récupérer une action de la queue (timeout court pour réactivité)
                template, position = self.action_queue.get(timeout=0.1)
                
                # Double vérification pour le mode Shift
                if self.shift_mode and not keyboard.is_pressed('shift'):
                    continue
                
                # Exécuter l'action
                print(f"🖱️ Clic sur {template.name}")
                
                # En mode Shift, toujours faire un shift+clic
                if self.shift_mode or keyboard.is_pressed('shift'):
                    self.mouse.shift_click(position[0], position[1])
                else:
                    self.mouse.click(position[0], position[1])
                
                self.stats['clicks'] += 1
                
                # Pause après action (simule temps de réaction humain)
                # Plus court en mode Shift car l'utilisateur contrôle
                if self.shift_mode:
                    time.sleep(random.uniform(0.3, 0.6))
                else:
                    time.sleep(random.uniform(0.8, 1.5))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Erreur action: {e}")
    
    def capture_template(self):
        """Capturer une zone comme template"""
        print("\n📸 Mode capture - Sélectionnez la zone à capturer...")
        time.sleep(2)
        
        # Obtenir la sélection de l'utilisateur
        x1, y1, x2, y2 = self.get_selection()
        if x1 is None:
            return
        
        # Capturer la zone
        screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        
        # Sauvegarder
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.templates_folder}/template_{timestamp}.png"
        screenshot.save(filename)
        
        print(f"✅ Template sauvegardé: {filename}")
    
    def get_selection(self) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Obtenir une sélection de zone par l'utilisateur"""
        print("Cliquez et maintenez pour sélectionner une zone...")
        print("Appuyez sur ESC pour annuler")
        
        # Simplification: utiliser pyautogui pour obtenir deux points
        print("Cliquez sur le coin supérieur gauche...")
        time.sleep(0.5)
        
        # Attendre le clic
        while not pyautogui.mouseDown():
            if keyboard.is_pressed('esc'):
                print("❌ Sélection annulée")
                return None, None, None, None
            time.sleep(0.01)
        
        x1, y1 = pyautogui.position()
        print(f"Point 1: ({x1}, {y1})")
        
        print("Cliquez sur le coin inférieur droit...")
        time.sleep(1)
        
        while not pyautogui.mouseDown():
            if keyboard.is_pressed('esc'):
                print("❌ Sélection annulée")
                return None, None, None, None
            time.sleep(0.01)
        
        x2, y2 = pyautogui.position()
        print(f"Point 2: ({x2}, {y2})")
        
        # S'assurer que les coordonnées sont dans le bon ordre
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        return x1, y1, x2, y2
    
    def print_stats(self):
        """Afficher les statistiques"""
        if self.stats['start_time']:
            duration = time.time() - self.stats['start_time']
            print(f"\n📊 Statistiques:")
            print(f"  Durée: {duration:.1f}s")
            print(f"  Détections: {self.stats['detections']}")
            print(f"  Clics: {self.stats['clicks']}")
    
    def load_config(self):
        """Charger la configuration"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                
                # Charger le mode
                mode = config.get('mode', 'shift')
                self.shift_mode = (mode == 'shift')
                self.auto_click_mode = (mode == 'auto')
                
                # Charger les templates
                for template_data in config.get('templates', []):
                    template = Template(**template_data)
                    self.detector.add_template(template)
                
                # Charger la région de détection
                if 'detection_region' in config:
                    self.detector.set_detection_region(**config['detection_region'])
                
                print(f"✅ Configuration chargée: {len(self.detector.templates)} templates")
                print(f"📌 Mode: {'SHIFT' if self.shift_mode else 'AUTO'}")
    
    def save_config(self):
        """Sauvegarder la configuration"""
        config = {
            'mode': 'shift' if self.shift_mode else 'auto',
            'templates': [
                {
                    'name': t.name,
                    'image_path': t.image_path,
                    'enabled': t.enabled,
                    'threshold': t.threshold,
                    'priority': t.priority,
                    'click_offset_x': t.click_offset_x,
                    'click_offset_y': t.click_offset_y
                }
                for t in self.detector.templates
            ]
        }
        
        if self.detector.detection_region:
            config['detection_region'] = {
                'x': self.detector.detection_region[0],
                'y': self.detector.detection_region[1],
                'width': self.detector.detection_region[2],
                'height': self.detector.detection_region[3]
            }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration sauvegardée")

# Interface graphique
class BotGUI:
    """Interface graphique pour configurer le bot"""
    def __init__(self, bot: PixelBot):
        self.bot = bot
        self.root = tk.Tk()
        self.root.title("Pixel Bot - Configuration")
        self.root.geometry("850x650")
        
        self.mode_var = None  # Sera initialisé dans setup_ui
        self.setup_ui()
        self.refresh_templates()
    
    def setup_ui(self):
        """Créer l'interface"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Section Mode de fonctionnement
        mode_frame = ttk.LabelFrame(main_frame, text="Mode de fonctionnement", padding="10")
        mode_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.mode_var = tk.StringVar(value="shift")
        ttk.Radiobutton(mode_frame, text="🎮 Mode SHIFT (Maintenez Shift pour activer)", 
                       variable=self.mode_var, value="shift",
                       command=self.update_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="🤖 Mode AUTO (Clics automatiques)", 
                       variable=self.mode_var, value="auto",
                       command=self.update_mode).pack(anchor=tk.W)
        
        ttk.Label(mode_frame, text="En mode SHIFT : Le bot analyse en permanence mais ne clique que quand vous maintenez Shift", 
                 font=('Arial', 9), foreground='gray').pack(anchor=tk.W, pady=(5, 0))
        
        # Section Templates
        ttk.Label(main_frame, text="Templates", font=('Arial', 14, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=(10, 5))
        
        # Liste des templates
        self.templates_frame = ttk.Frame(main_frame)
        self.templates_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Boutons de contrôle
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        ttk.Button(control_frame, text="Ajouter Template", command=self.add_template).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Capturer Nouveau", command=self.capture_new_template).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Définir Zone", command=self.set_detection_zone).pack(side=tk.LEFT, padx=5)
        
        # Section Arbres spécifiques
        ttk.Label(main_frame, text="Arbres à récolter", font=('Arial', 12, 'bold')).grid(row=4, column=0, sticky=tk.W, pady=(20, 5))
        
        trees_frame = ttk.Frame(main_frame)
        trees_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Checkboxes pour les arbres
        self.tree_vars = {}
        trees = ['Frêne', 'Châtaignier', 'Noyer', 'Chêne', 'Érable', 'If', 'Ébène', 'Orme']
        
        for i, tree in enumerate(trees):
            var = tk.BooleanVar()
            self.tree_vars[tree] = var
            ttk.Checkbutton(trees_frame, text=tree, variable=var).grid(row=i//4, column=i%4, sticky=tk.W, padx=10)
        
        # Boutons d'action
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=6, column=0, columnspan=3, pady=20)
        
        self.start_button = ttk.Button(action_frame, text="▶ Démarrer Bot", command=self.toggle_bot)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_frame, text="💾 Sauvegarder Config", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="📂 Charger Config", command=self.load_config).pack(side=tk.LEFT, padx=5)
        
        # Status bar avec info mode
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Prêt - Mode SHIFT activé")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X)
        
        # Info supplémentaire
        info_label = ttk.Label(status_frame, text="💡 Astuce: En mode SHIFT, maintenez Shift pour que le bot clique sur les éléments détectés", 
                              font=('Arial', 9), foreground='blue')
        info_label.pack(pady=(5, 0))
    
    def refresh_templates(self):
        """Rafraîchir la liste des templates"""
        # Effacer les widgets existants
        for widget in self.templates_frame.winfo_children():
            widget.destroy()
        
        # Headers
        ttk.Label(self.templates_frame, text="Nom", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5)
        ttk.Label(self.templates_frame, text="Fichier", font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5)
        ttk.Label(self.templates_frame, text="Actif", font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=5)
        ttk.Label(self.templates_frame, text="Seuil", font=('Arial', 10, 'bold')).grid(row=0, column=3, padx=5)
        ttk.Label(self.templates_frame, text="Actions", font=('Arial', 10, 'bold')).grid(row=0, column=4, padx=5)
        
        # Afficher les templates
        for i, template in enumerate(self.bot.detector.templates, 1):
            ttk.Label(self.templates_frame, text=template.name).grid(row=i, column=0, padx=5)
            ttk.Label(self.templates_frame, text=os.path.basename(template.image_path)).grid(row=i, column=1, padx=5)
            
            # Checkbox actif
            var = tk.BooleanVar(value=template.enabled)
            var.trace('w', lambda *args, t=template, v=var: setattr(t, 'enabled', v.get()))
            ttk.Checkbutton(self.templates_frame, variable=var).grid(row=i, column=2, padx=5)
            
            # Seuil
            threshold_var = tk.StringVar(value=str(template.threshold))
            threshold_entry = ttk.Entry(self.templates_frame, textvariable=threshold_var, width=5)
            threshold_entry.grid(row=i, column=3, padx=5)
            threshold_var.trace('w', lambda *args, t=template, v=threshold_var: self.update_threshold(t, v))
            
            # Bouton supprimer
            ttk.Button(self.templates_frame, text="❌", 
                      command=lambda t=template: self.remove_template(t)).grid(row=i, column=4, padx=5)
    
    def update_threshold(self, template, var):
        """Mettre à jour le seuil d'un template"""
        try:
            template.threshold = float(var.get())
        except ValueError:
            pass
    
    def add_template(self):
        """Ajouter un template existant"""
        filename = filedialog.askopenfilename(
            title="Sélectionner un template",
            initialdir=self.bot.templates_folder,
            filetypes=[("Images", "*.png *.jpg *.jpeg")]
        )
        
        if filename:
            name = simpledialog.askstring("Nom du template", "Entrez un nom pour ce template:")
            if name:
                template = Template(name=name, image_path=filename)
                if self.bot.detector.add_template(template):
                    self.refresh_templates()
                    self.status_var.set(f"Template '{name}' ajouté")
    
    def capture_new_template(self):
        """Capturer un nouveau template"""
        self.root.withdraw()  # Cacher la fenêtre
        time.sleep(0.5)
        
        self.bot.capture_template()
        
        self.root.deiconify()  # Réafficher la fenêtre
        self.refresh_templates()
    
    def set_detection_zone(self):
        """Définir la zone de détection"""
        self.root.withdraw()
        time.sleep(0.5)
        
        print("\n🎯 Définir la zone de détection...")
        x1, y1, x2, y2 = self.bot.get_selection()
        
        if x1 is not None:
            self.bot.detector.set_detection_region(x1, y1, x2 - x1, y2 - y1)
            self.status_var.set(f"Zone définie: {x2-x1}x{y2-y1} à ({x1},{y1})")
        
        self.root.deiconify()
    
    def remove_template(self, template):
        """Supprimer un template"""
        self.bot.detector.templates.remove(template)
        self.refresh_templates()
    
    def update_mode(self):
        """Mettre à jour le mode de fonctionnement"""
        mode = self.mode_var.get()
        self.bot.shift_mode = (mode == "shift")
        self.bot.auto_click_mode = (mode == "auto")
        
        if self.bot.shift_mode:
            self.status_var.set("Mode SHIFT activé - Maintenez Shift pour cliquer")
        else:
            self.status_var.set("Mode AUTO activé - Clics automatiques")
        
        # Si le bot est en cours, le redémarrer avec le nouveau mode
        if self.bot.is_running:
            self.bot.stop_bot()
            time.sleep(0.5)
            self.bot.start_bot()
    
    def toggle_bot(self):
        """Démarrer/Arrêter le bot"""
        if self.bot.is_running:
            self.bot.stop_bot()
            self.start_button.config(text="▶ Démarrer Bot")
            self.status_var.set("Bot arrêté")
        else:
            # Mettre à jour les templates selon les arbres sélectionnés
            for template in self.bot.detector.templates:
                for tree_name, var in self.tree_vars.items():
                    if tree_name.lower() in template.name.lower():
                        template.enabled = var.get()
            
            # Appliquer le mode sélectionné
            mode = self.mode_var.get()
            self.bot.shift_mode = (mode == "shift")
            self.bot.auto_click_mode = (mode == "auto")
            
            self.bot.start_bot()
            self.start_button.config(text="⏸ Arrêter Bot")
            
            if self.bot.shift_mode:
                self.status_var.set("Bot actif - Mode SHIFT (Maintenez Shift pour activer)")
            else:
                self.status_var.set("Bot actif - Mode AUTO")
    
    def save_config(self):
        """Sauvegarder la configuration"""
        self.bot.save_config()
        self.status_var.set("Configuration sauvegardée")
    
    def load_config(self):
        """Charger la configuration"""
        self.bot.load_config()
        self.refresh_templates()
        
        # Mettre à jour le mode dans l'interface
        if self.bot.shift_mode:
            self.mode_var.set("shift")
            self.status_var.set("Configuration chargée - Mode SHIFT")
        else:
            self.mode_var.set("auto")
            self.status_var.set("Configuration chargée - Mode AUTO")
    
    def run(self):
        """Lancer l'interface"""
        self.root.mainloop()

# Programme principal
if __name__ == "__main__":
    print("🤖 Pixel Bot avec reconnaissance d'image")
    print("=" * 50)
    
    # Créer le bot
    bot = PixelBot()
    
    # Charger la configuration si elle existe
    bot.load_config()
    
    # Lancer l'interface graphique
    gui = BotGUI(bot)
    
    # Thread pour les raccourcis clavier
    keyboard_thread = threading.Thread(target=lambda: keyboard.wait('ctrl+q'))
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    # Lancer l'interface
    gui.run()