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
from typing import List, Tuple, Optional, Set
import queue
from collections import deque

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
        # Vitesse augmentée pour des mouvements plus rapides
        self.move_speed = 0.001  # Beaucoup plus rapide
        self.use_acceleration = True
        pyautogui.FAILSAFE = True
        pyautogui.MINIMUM_DURATION = 0  # Désactiver la durée minimale
        pyautogui.PAUSE = 0  # Pas de pause automatique
    
    def bezier_curve(self, start, end, control1=None, control2=None, steps=3):
        """Génère une courbe de Bézier simplifiée pour mouvement rapide"""
        if control1 is None:
            # Réduire la courbure pour des mouvements plus directs
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            control1 = (
                mid_x + random.randint(-20, 20),
                mid_y + random.randint(-20, 20)
            )
            control2 = control1  # Un seul point de contrôle pour simplicité
        
        points = []
        for t in np.linspace(0, 1, steps):
            x = (1-t)**3 * start[0] + 3*(1-t)**2*t * control1[0] + \
                3*(1-t)*t**2 * control2[0] + t**3 * end[0]
            y = (1-t)**3 * start[1] + 3*(1-t)**2*t * control1[1] + \
                3*(1-t)*t**2 * control2[1] + t**3 * end[1]
            points.append((int(x), int(y)))
        return points
    
    def move_to(self, x, y, use_bezier=True, speed_multiplier=1.0):
        """Déplacer vers une position avec mouvement rapide et fluide"""
        current_x, current_y = pyautogui.position()
        
        # Petite imprécision pour l'aspect humain
        x += random.randint(-1, 1)
        y += random.randint(-1, 1)
        
        distance = math.sqrt((x - current_x)**2 + (y - current_y)**2)
        
        if distance < 50:  # Mouvement très court - direct
            pyautogui.moveTo(x, y, duration=0)
        elif use_bezier and distance > 100:
            # Utiliser Bézier seulement pour les longues distances
            points = self.bezier_curve((current_x, current_y), (x, y))
            for point in points[:-1]:  # Sauter le dernier point
                pyautogui.moveTo(point[0], point[1], duration=0)
                time.sleep(0.001 * speed_multiplier)  # Très court délai
            pyautogui.moveTo(x, y, duration=0)  # Point final précis
        else:
            # Mouvement direct avec accélération
            # Calculer une durée basée sur la distance (max 0.2 secondes)
            duration = min(0.2, distance / 5000) * speed_multiplier
            pyautogui.moveTo(x, y, duration=duration)
        
        # Micro-pause après mouvement
        time.sleep(random.uniform(0.01, 0.03) * speed_multiplier)
    
    def click(self, x, y):
        """Clic simple et rapide"""
        self.move_to(x, y, speed_multiplier=0.5)  # Mouvement encore plus rapide
        pyautogui.click()
        time.sleep(random.uniform(0.05, 0.1))
    
    def shift_click(self, x, y):
        """Clic avec Shift - optimisé pour rapidité"""
        self.move_to(x, y, speed_multiplier=0.5)
        # Vérifier si Shift est déjà pressé
        if not keyboard.is_pressed('shift'):
            pyautogui.keyDown('shift')
            time.sleep(0.01)
        pyautogui.click()
        time.sleep(random.uniform(0.05, 0.1))
    
    def random_pause(self, min_time, max_time):
        """Pause aléatoire réduite"""
        pause_time = random.uniform(min_time, max_time)
        time.sleep(pause_time)

class ImageDetector:
    """Classe pour la détection d'images avec système anti-doublons"""
    def __init__(self):
        self.templates: List[Template] = []
        self.detection_region = None  # (x, y, width, height)
        self.clicked_positions: Set[Tuple[int, int]] = set()  # Positions déjà cliquées
        self.position_history = deque(maxlen=50)  # Historique des dernières positions
        self.click_cooldown = {}  # Cooldown par position
        self.min_distance_between_clicks = 50  # Distance minimale entre deux clics
        self.position_timeout = 10.0  # Temps avant de pouvoir recliquer au même endroit
    
    def add_template(self, template: Template):
        """Ajouter un template à détecter"""
        if os.path.exists(template.image_path):
            self.templates.append(template)
            return True
        return False
    
    def set_detection_region(self, x, y, width, height):
        """Définir la région de détection"""
        self.detection_region = (x, y, width, height)
    
    def reset_clicked_positions(self):
        """Réinitialiser les positions cliquées"""
        self.clicked_positions.clear()
        self.click_cooldown.clear()
        self.position_history.clear()
    
    def is_position_valid(self, x: int, y: int) -> bool:
        """Vérifier si une position peut être cliquée"""
        current_time = time.time()
        
        # Nettoyer les anciennes positions du cooldown
        positions_to_remove = []
        for pos, timestamp in self.click_cooldown.items():
            if current_time - timestamp > self.position_timeout:
                positions_to_remove.append(pos)
        
        for pos in positions_to_remove:
            del self.click_cooldown[pos]
            if pos in self.clicked_positions:
                self.clicked_positions.remove(pos)
        
        # Vérifier si la position est trop proche d'une position récente
        for clicked_x, clicked_y in self.clicked_positions:
            distance = math.sqrt((x - clicked_x)**2 + (y - clicked_y)**2)
            if distance < self.min_distance_between_clicks:
                return False
        
        # Vérifier le cooldown spécifique
        for (cx, cy), _ in self.click_cooldown.items():
            if abs(x - cx) < 30 and abs(y - cy) < 30:
                return False
        
        return True
    
    def mark_position_clicked(self, x: int, y: int):
        """Marquer une position comme cliquée"""
        # Arrondir la position pour grouper les positions proches
        rounded_x = (x // 30) * 30
        rounded_y = (y // 30) * 30
        position = (rounded_x, rounded_y)
        
        self.clicked_positions.add(position)
        self.click_cooldown[position] = time.time()
        self.position_history.append((x, y))
    
    def capture_screen(self) -> np.ndarray:
        """Capturer l'écran ou une région spécifique"""
        if self.detection_region:
            x, y, w, h = self.detection_region
            screenshot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        else:
            screenshot = ImageGrab.grab()
        
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        return screenshot
    
    def detect_template(self, screenshot: np.ndarray, template: Template) -> List[Tuple[int, int]]:
        """Détecter un template avec filtrage des positions déjà cliquées"""
        if not template.enabled:
            return []
        
        template_img = cv2.imread(template.image_path)
        if template_img is None:
            return []
        
        result = cv2.matchTemplate(screenshot, template_img, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= template.threshold)
        matches = []
        
        h, w = template_img.shape[:2]
        
        for pt in zip(*locations[::-1]):
            center_x = pt[0] + w // 2 + template.click_offset_x
            center_y = pt[1] + h // 2 + template.click_offset_y
            
            if self.detection_region:
                center_x += self.detection_region[0]
                center_y += self.detection_region[1]
            
            # Vérifier si cette position est valide (pas déjà cliquée)
            if self.is_position_valid(center_x, center_y):
                matches.append((center_x, center_y))
        
        # Supprimer les doublons proches et trier par distance depuis la dernière position
        if matches:
            matches = self.remove_close_duplicates(matches, min_distance=30)
            matches = self.sort_by_optimal_path(matches)
        
        return matches
    
    def sort_by_optimal_path(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Trier les points pour un parcours optimal (plus proche voisin)"""
        if len(points) <= 1:
            return points
        
        # Obtenir la position actuelle de la souris
        current_x, current_y = pyautogui.position()
        
        sorted_points = []
        remaining = points.copy()
        current_pos = (current_x, current_y)
        
        while remaining:
            # Trouver le point le plus proche
            min_dist = float('inf')
            closest_point = None
            closest_idx = -1
            
            for idx, point in enumerate(remaining):
                dist = math.sqrt((point[0] - current_pos[0])**2 + 
                               (point[1] - current_pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = point
                    closest_idx = idx
            
            if closest_point:
                sorted_points.append(closest_point)
                remaining.pop(closest_idx)
                current_pos = closest_point
        
        return sorted_points
    
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
        """Détecter tous les templates actifs avec système de rotation"""
        screenshot = self.capture_screen()
        results = []
        
        sorted_templates = sorted(self.templates, key=lambda t: t.priority, reverse=True)
        
        for template in sorted_templates:
            matches = self.detect_template(screenshot, template)
            if matches:
                results.append((template, matches))
        
        return results

class PixelBot:
    """Bot principal avec détection d'image amélioré"""
    def __init__(self):
        self.mouse = Mouse()
        self.detector = ImageDetector()
        self.is_running = False
        self.shift_mode = True
        self.auto_click_mode = False
        self.config_file = "bot_config.json"
        self.templates_folder = "templates"
        self.setup_folders()
        self.setup_hotkeys()
        self.action_queue = queue.Queue()
        self.pending_actions = []
        self.last_shift_state = False
        self.shift_released = threading.Event()  # Event pour détecter le relâchement de Shift
        self.current_action_thread = None
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
        
        # Réinitialiser les positions cliquées au démarrage
        self.detector.reset_clicked_positions()
        
        if self.shift_mode:
            print("\n🟢 Bot démarré en mode SHIFT!")
            print("📌 Maintenez SHIFT pour activer les clics")
            print("📍 Système de rotation activé - évite les doublons")
        else:
            print("\n🟢 Bot démarré en mode AUTO!")
        
        print("F1 : Toggle Bot | F2 : Arrêt | F10 : Capturer template")
        
        self.is_running = True
        self.stats['start_time'] = time.time()
        self.pending_actions = []
        self.shift_released.clear()
        
        # Thread de détection
        detection_thread = threading.Thread(target=self.detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        
        # Thread d'action
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
        self.shift_released.set()  # Débloquer les threads en attente
        print("\n🔴 Bot arrêté!")
        self.print_stats()
    
    def detection_loop(self):
        """Boucle de détection optimisée"""
        while self.is_running:
            try:
                # Détecter tous les templates
                detections = self.detector.detect_all()
                
                if detections:
                    # Créer une liste triée de toutes les détections
                    all_detections = []
                    for template, matches in detections:
                        for match in matches:
                            all_detections.append((template, match))
                    
                    # Trier par distance optimale
                    if all_detections:
                        self.pending_actions = all_detections
                        self.stats['detections'] += len(all_detections)
                        
                        if self.shift_mode:
                            print(f"👁️ {len(all_detections)} cibles détectées - En attente de SHIFT")
                
                # Pause courte entre détections
                time.sleep(random.uniform(0.2, 0.4))
                
            except Exception as e:
                print(f"❌ Erreur détection: {e}")
    
    def shift_monitor(self):
        """Surveille l'état de la touche Shift avec arrêt immédiat"""
        while self.is_running and self.shift_mode:
            try:
                shift_pressed = keyboard.is_pressed('shift')
                
                if shift_pressed and not self.last_shift_state:
                    print("⚡ SHIFT activé - Exécution des actions...")
                    self.shift_released.clear()
                    # Ajouter les actions à la queue
                    for action in self.pending_actions:
                        self.action_queue.put(action)
                    self.pending_actions = []
                    
                elif not shift_pressed and self.last_shift_state:
                    print("💤 SHIFT relâché - Arrêt immédiat")
                    # Signal pour arrêter immédiatement
                    self.shift_released.set()
                    # Vider la queue
                    while not self.action_queue.empty():
                        try:
                            self.action_queue.get_nowait()
                        except:
                            break
                    # Réinitialiser pour le prochain cycle
                    self.detector.reset_clicked_positions()
                
                self.last_shift_state = shift_pressed
                time.sleep(0.01)  # Check très rapide
                
            except Exception as e:
                print(f"❌ Erreur monitoring Shift: {e}")
    
    def action_loop(self):
        """Boucle d'exécution des actions avec arrêt immédiat"""
        while self.is_running:
            try:
                # En mode Shift, vérifier constamment si Shift est relâché
                if self.shift_mode:
                    if not keyboard.is_pressed('shift'):
                        time.sleep(0.01)
                        continue
                    
                    # Vérifier si on doit s'arrêter
                    if self.shift_released.is_set():
                        continue
                
                # Récupérer une action avec timeout très court
                try:
                    template, position = self.action_queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                
                # Vérification finale avant le clic
                if self.shift_mode:
                    if not keyboard.is_pressed('shift') or self.shift_released.is_set():
                        continue
                
                # Marquer la position comme cliquée AVANT le clic
                self.detector.mark_position_clicked(position[0], position[1])
                
                # Exécuter l'action
                print(f"🖱️ Clic sur {template.name} à ({position[0]}, {position[1]})")
                
                # Faire le clic rapidement
                if self.shift_mode or keyboard.is_pressed('shift'):
                    self.mouse.shift_click(position[0], position[1])
                else:
                    self.mouse.click(position[0], position[1])
                
                self.stats['clicks'] += 1
                
                # Vérifier encore si on doit s'arrêter après le clic
                if self.shift_mode and (not keyboard.is_pressed('shift') or self.shift_released.is_set()):
                    print("⏸️ Arrêt après clic - Shift relâché")
                    # Vider la queue
                    while not self.action_queue.empty():
                        try:
                            self.action_queue.get_nowait()
                        except:
                            break
                    continue
                
                # Micro-pause entre actions
                time.sleep(random.uniform(0.1, 0.2))
                
            except Exception as e:
                print(f"❌ Erreur action: {e}")
    
    def capture_template(self):
        """Capturer une zone comme template"""
        print("\n📸 Mode capture - Sélectionnez la zone à capturer...")
        time.sleep(2)
        
        x1, y1, x2, y2 = self.get_selection()
        if x1 is None:
            return
        
        screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.templates_folder}/template_{timestamp}.png"
        screenshot.save(filename)
        
        print(f"✅ Template sauvegardé: {filename}")
    
    def get_selection(self) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Obtenir une sélection de zone par l'utilisateur"""
        print("Cliquez et maintenez pour sélectionner une zone...")
        print("Appuyez sur ESC pour annuler")
        
        print("Cliquez sur le coin supérieur gauche...")
        time.sleep(0.5)
        
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
            print(f"  Positions uniques: {len(self.detector.clicked_positions)}")
    
    def load_config(self):
        """Charger la configuration"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                
                mode = config.get('mode', 'shift')
                self.shift_mode = (mode == 'shift')
                self.auto_click_mode = (mode == 'auto')
                
                for template_data in config.get('templates', []):
                    template = Template(**template_data)
                    self.detector.add_template(template)
                
                if 'detection_region' in config:
                    self.detector.set_detection_region(**config['detection_region'])
                
                # Charger les paramètres de vitesse si présents
                settings = config.get('settings', {})
                if 'move_speed' in settings:
                    self.mouse.move_speed = settings['move_speed']
                
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
            ],
            'settings': {
                'move_speed': self.mouse.move_speed,
                'min_distance_between_clicks': self.detector.min_distance_between_clicks,
                'position_timeout': self.detector.position_timeout
            }
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

# Interface graphique (inchangée mais avec ajout du bouton reset)
class BotGUI:
    """Interface graphique pour configurer le bot"""
    def __init__(self, bot: PixelBot):
        self.bot = bot
        self.root = tk.Tk()
        self.root.title("Pixel Bot - Configuration")
        self.root.geometry("850x700")
        
        self.mode_var = None
        self.setup_ui()
        self.refresh_templates()
    
    def setup_ui(self):
        """Créer l'interface"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Section Mode
        mode_frame = ttk.LabelFrame(main_frame, text="Mode de fonctionnement", padding="10")
        mode_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.mode_var = tk.StringVar(value="shift")
        ttk.Radiobutton(mode_frame, text="🎮 Mode SHIFT (Maintenez Shift pour activer)", 
                       variable=self.mode_var, value="shift",
                       command=self.update_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="🤖 Mode AUTO (Clics automatiques)", 
                       variable=self.mode_var, value="auto",
                       command=self.update_mode).pack(anchor=tk.W)
        
        ttk.Label(mode_frame, text="✨ Système anti-doublons: évite de cliquer plusieurs fois au même endroit", 
                 font=('Arial', 9), foreground='green').pack(anchor=tk.W, pady=(5, 0))
        ttk.Label(mode_frame, text="⚡ Mouvements rapides et fluides (< 1 seconde pour traverser l'écran)", 
                 font=('Arial', 9), foreground='blue').pack(anchor=tk.W)
        
        # Section Templates
        ttk.Label(main_frame, text="Templates", font=('Arial', 14, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=(10, 5))
        
        self.templates_frame = ttk.Frame(main_frame)
        self.templates_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Boutons de contrôle
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        ttk.Button(control_frame, text="Ajouter Template", command=self.add_template).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Capturer Nouveau", command=self.capture_new_template).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Définir Zone", command=self.set_detection_zone).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🔄 Reset Positions", command=self.reset_positions).pack(side=tk.LEFT, padx=5)
        
        # Section Arbres
        ttk.Label(main_frame, text="Bucheron", font=('Arial', 12, 'bold')).grid(row=4, column=0, sticky=tk.W, pady=(20, 5))
        
        trees_frame = ttk.Frame(main_frame)
        trees_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.tree_vars = {}
        trees = ['Frene', 'Chataignier', 'Ortie'] #, 'Noyer', 'Chene', 'Bombu', 'Erable', 'Oliviolet', 'Pin', 'If', 'Bambou', 'Merisier', 'Noisetier', 'Ebene', 'Kaliptus', 'Charme', 'Bambou Sacre', 'Aquajou', 'Tremble']
        
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
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Prêt - Mode SHIFT activé")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X)
        
        info_label = ttk.Label(status_frame, text="💡 Relâchez SHIFT = Arrêt immédiat | Système de rotation intelligent activé", 
                              font=('Arial', 9), foreground='blue')
        info_label.pack(pady=(5, 0))
    
    def reset_positions(self):
        """Réinitialiser les positions cliquées"""
        self.bot.detector.reset_clicked_positions()
        self.status_var.set("Positions réinitialisées - Prêt à scanner toute la zone")
    
    def refresh_templates(self):
        """Rafraîchir la liste des templates"""
        for widget in self.templates_frame.winfo_children():
            widget.destroy()
        
        ttk.Label(self.templates_frame, text="Nom", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5)
        ttk.Label(self.templates_frame, text="Fichier", font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5)
        ttk.Label(self.templates_frame, text="Actif", font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=5)
        ttk.Label(self.templates_frame, text="Seuil", font=('Arial', 10, 'bold')).grid(row=0, column=3, padx=5)
        ttk.Label(self.templates_frame, text="Actions", font=('Arial', 10, 'bold')).grid(row=0, column=4, padx=5)
        
        for i, template in enumerate(self.bot.detector.templates, 1):
            ttk.Label(self.templates_frame, text=template.name).grid(row=i, column=0, padx=5)
            ttk.Label(self.templates_frame, text=os.path.basename(template.image_path)).grid(row=i, column=1, padx=5)
            
            var = tk.BooleanVar(value=template.enabled)
            var.trace('w', lambda *args, t=template, v=var: setattr(t, 'enabled', v.get()))
            ttk.Checkbutton(self.templates_frame, variable=var).grid(row=i, column=2, padx=5)
            
            threshold_var = tk.StringVar(value=str(template.threshold))
            threshold_entry = ttk.Entry(self.templates_frame, textvariable=threshold_var, width=5)
            threshold_entry.grid(row=i, column=3, padx=5)
            threshold_var.trace('w', lambda *args, t=template, v=threshold_var: self.update_threshold(t, v))
            
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
        self.root.withdraw()
        time.sleep(0.5)
        self.bot.capture_template()
        self.root.deiconify()
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
            self.status_var.set("Mode SHIFT - Relâcher = Arrêt immédiat")
        else:
            self.status_var.set("Mode AUTO - Clics automatiques")
        
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
            for template in self.bot.detector.templates:
                for tree_name, var in self.tree_vars.items():
                    if tree_name.lower() in template.name.lower():
                        template.enabled = var.get()
            
            mode = self.mode_var.get()
            self.bot.shift_mode = (mode == "shift")
            self.bot.auto_click_mode = (mode == "auto")
            
            self.bot.start_bot()
            self.start_button.config(text="⏸ Arrêter Bot")
            
            if self.bot.shift_mode:
                self.status_var.set("Bot actif - SHIFT pour activer | Relâcher = Stop")
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
        
        if self.bot.shift_mode:
            self.mode_var.set("shift")
            self.status_var.set("Config chargée - Mode SHIFT")
        else:
            self.mode_var.set("auto")
            self.status_var.set("Config chargée - Mode AUTO")
    
    def run(self):
        """Lancer l'interface"""
        self.root.mainloop()

# Programme principal
if __name__ == "__main__":
    print("🤖 Pixel Bot v2.0 - Système intelligent")
    print("=" * 50)
    print("✨ Nouveautés:")
    print("  • Système anti-doublons avec rotation intelligente")
    print("  • Arrêt immédiat au relâchement de SHIFT")
    print("  • Mouvements ultra-rapides et fluides")
    print("=" * 50)
    
    bot = PixelBot()
    bot.load_config()
    
    gui = BotGUI(bot)
    
    keyboard_thread = threading.Thread(target=lambda: keyboard.wait('ctrl+q'))
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    gui.run()