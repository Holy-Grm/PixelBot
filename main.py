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
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Set, Callable, Dict, Any
from enum import Enum
import queue
from collections import deque
from abc import ABC, abstractmethod

# ================== CONSTANTS & CONFIGURATION ==================

class BotMode(Enum):
    SHIFT = "shift"
    AUTO = "auto"

class Constants:
    """Constantes globales du bot"""
    # Fichiers et dossiers
    CONFIG_FILE = "bot_config.json"
    TEMPLATES_FOLDER = "templates"
    
    # Paramètres souris
    DEFAULT_MOVE_SPEED = 0.001
    BEZIER_STEPS = 3
    MOUSE_PRECISION_OFFSET = 1
    
    # Paramètres détection
    DEFAULT_THRESHOLD = 0.8
    DEFAULT_PRIORITY = 1
    MIN_DISTANCE_BETWEEN_CLICKS = 50
    POSITION_TIMEOUT = 10.0
    DETECTION_PAUSE_MIN = 0.2
    DETECTION_PAUSE_MAX = 0.4
    
    # Interface
    WINDOW_WIDTH = 850
    WINDOW_HEIGHT = 600
    
    # Raccourcis clavier
    HOTKEY_TOGGLE = 'f1'
    HOTKEY_STOP = 'f2'
    HOTKEY_CAPTURE = 'f10'
    HOTKEY_EXIT = 'ctrl+q'

@dataclass
class BotConfig:
    """Configuration centralisée du bot"""
    mode: BotMode = BotMode.SHIFT
    move_speed: float = Constants.DEFAULT_MOVE_SPEED
    min_distance_between_clicks: int = Constants.MIN_DISTANCE_BETWEEN_CLICKS
    position_timeout: float = Constants.POSITION_TIMEOUT
    detection_region: Optional[Tuple[int, int, int, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        config = asdict(self)
        config['mode'] = self.mode.value
        return config
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BotConfig':
        if 'mode' in data:
            data['mode'] = BotMode(data['mode'])
        return cls(**data)

@dataclass
class Template:
    """Template pour la détection d'image"""
    name: str
    image_path: str
    enabled: bool = True
    threshold: float = Constants.DEFAULT_THRESHOLD
    priority: int = Constants.DEFAULT_PRIORITY
    click_offset_x: int = 0
    click_offset_y: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Template':
        return cls(**data)

# ================== EVENT SYSTEM ==================

class EventType(Enum):
    """Types d'événements du système"""
    BOT_STARTED = "bot_started"
    BOT_STOPPED = "bot_stopped"
    TEMPLATE_DETECTED = "template_detected"
    CLICK_EXECUTED = "click_executed"
    SHIFT_PRESSED = "shift_pressed"
    SHIFT_RELEASED = "shift_released"
    ERROR_OCCURRED = "error_occurred"
    STATUS_CHANGED = "status_changed"

class EventManager:
    """Gestionnaire d'événements pour découpler les composants"""
    def __init__(self):
        self.listeners: Dict[EventType, List[Callable]] = {}
    
    def subscribe(self, event_type: EventType, callback: Callable):
        """S'abonner à un type d'événement"""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)
    
    def emit(self, event_type: EventType, data: Any = None):
        """Émettre un événement"""
        if event_type in self.listeners:
            for callback in self.listeners[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"❌ Erreur dans event listener {event_type}: {e}")

# ================== LOGGING SYSTEM ==================

class Logger:
    """Système de logging centralisé"""
    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager
    
    def info(self, message: str):
        print(f"ℹ️ {message}")
        self.event_manager.emit(EventType.STATUS_CHANGED, message)
    
    def success(self, message: str):
        print(f"✅ {message}")
        self.event_manager.emit(EventType.STATUS_CHANGED, message)
    
    def warning(self, message: str):
        print(f"⚠️ {message}")
        self.event_manager.emit(EventType.STATUS_CHANGED, message)
    
    def error(self, message: str):
        print(f"❌ {message}")
        self.event_manager.emit(EventType.ERROR_OCCURRED, message)

# ================== SCREEN CAPTURE SERVICE ==================

class ScreenCaptureOverlay:
    """Overlay pour sélection visuelle de zone d'écran"""
    def __init__(self):
        self.root = None
        self.canvas = None
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.selection = None
        self.is_selecting = False
    
    def select_area(self) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Ouvrir l'overlay de sélection et retourner les coordonnées"""
        self.selection = None
        self._create_overlay()
        self._setup_bindings()
        
        # Attendre la sélection
        self.root.wait_window()
        
        return self.selection
    
    def _create_overlay(self):
        """Créer l'overlay plein écran"""
        self.root = tk.Toplevel()
        self.root.title("Sélection de zone")
        
        # Configuration fenêtre plein écran
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.3)  # Transparence
        self.root.configure(bg='black')
        self.root.focus_set()
        
        # Canvas pour dessiner
        self.canvas = tk.Canvas(
            self.root, 
            bg='black', 
            highlightthickness=0,
            cursor='crosshair'
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        instruction_text = "Cliquez et glissez pour sélectionner une zone • ESC pour annuler"
        self.canvas.create_text(
            screen_width // 2, 50,
            text=instruction_text,
            fill='white',
            font=('Arial', 14, 'bold'),
            tags='instructions'
        )
    
    def _setup_bindings(self):
        """Configurer les événements souris et clavier"""
        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.root.bind('<Escape>', self._on_cancel)
        self.root.bind('<Return>', self._on_confirm)
    
    def _on_click(self, event):
        """Début de sélection"""
        self.start_x = event.x
        self.start_y = event.y
        self.is_selecting = True
        
        # Supprimer l'ancien rectangle
        if self.current_rect:
            self.canvas.delete(self.current_rect)
    
    def _on_drag(self, event):
        """Mise à jour du rectangle pendant le glissement"""
        if not self.is_selecting:
            return
        
        # Supprimer l'ancien rectangle
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        
        # Dessiner le nouveau rectangle
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='red',
            width=2,
            fill='red',
            stipple='gray25'  # Motif semi-transparent
        )
        
        # Afficher les dimensions
        width = abs(event.x - self.start_x)
        height = abs(event.y - self.start_y)
        
        # Supprimer l'ancien texte de dimensions
        self.canvas.delete('dimensions')
        
        # Afficher nouvelles dimensions
        center_x = (self.start_x + event.x) // 2
        center_y = (self.start_y + event.y) // 2
        
        self.canvas.create_text(
            center_x, center_y,
            text=f"{width} × {height}",
            fill='white',
            font=('Arial', 12, 'bold'),
            tags='dimensions'
        )
    
    def _on_release(self, event):
        """Fin de sélection"""
        if not self.is_selecting:
            return
        
        self.is_selecting = False
        
        # Calculer les coordonnées finales
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        
        # Vérifier taille minimale
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            self.canvas.delete(self.current_rect)
            self.canvas.delete('dimensions')
            return
        
        # Afficher confirmation
        self._show_confirmation(x1, y1, x2, y2)
    
    def _show_confirmation(self, x1, y1, x2, y2):
        """Afficher la confirmation de sélection"""
        # Supprimer les instructions
        self.canvas.delete('instructions')
        
        # Afficher texte de confirmation
        screen_width = self.root.winfo_screenwidth()
        confirmation_text = f"Zone sélectionnée: {x2-x1}×{y2-y1} • ENTRÉE pour confirmer • ESC pour annuler"
        
        self.canvas.create_text(
            screen_width // 2, 50,
            text=confirmation_text,
            fill='yellow',
            font=('Arial', 14, 'bold'),
            tags='confirmation'
        )
        
        # Stocker la sélection temporaire
        self._temp_selection = (x1, y1, x2, y2)
    
    def _on_confirm(self, event=None):
        """Confirmer la sélection"""
        if hasattr(self, '_temp_selection'):
            self.selection = self._temp_selection
        self.root.destroy()
    
    def _on_cancel(self, event=None):
        """Annuler la sélection"""
        self.selection = (None, None, None, None)
        self.root.destroy()

class ScreenCaptureService:
    """Service pour la capture et sélection d'écran"""
    def __init__(self, logger: Logger):
        self.logger = logger
    
    def select_screen_area(self) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Sélectionner une zone d'écran avec interface graphique"""
        self.logger.info("Sélection de zone - Interface graphique activée")
        
        try:
            overlay = ScreenCaptureOverlay()
            return overlay.select_area()
        except Exception as e:
            self.logger.error(f"Erreur sélection zone: {e}")
            return None, None, None, None
    
    def capture_area(self, x1: int, y1: int, x2: int, y2: int, save_path: str) -> bool:
        """Capturer une zone spécifique et la sauvegarder"""
        try:
            screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
            screenshot.save(save_path)
            self.logger.success(f"Zone capturée: {save_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erreur capture: {e}")
            return False

# ================== SERVICES ==================

class MouseService:
    """Service pour la gestion des mouvements de souris"""
    def __init__(self, config: BotConfig):
        self.config = config
        self._setup_pyautogui()
    
    def _setup_pyautogui(self):
        pyautogui.FAILSAFE = True
        pyautogui.MINIMUM_DURATION = 0
        pyautogui.PAUSE = 0
    
    def move_to(self, x: int, y: int, use_bezier: bool = True, speed_multiplier: float = 1.0):
        """Déplacer la souris vers une position"""
        current_x, current_y = pyautogui.position()
        
        # Ajouter une petite imprécision
        x += random.randint(-Constants.MOUSE_PRECISION_OFFSET, Constants.MOUSE_PRECISION_OFFSET)
        y += random.randint(-Constants.MOUSE_PRECISION_OFFSET, Constants.MOUSE_PRECISION_OFFSET)
        
        distance = math.sqrt((x - current_x)**2 + (y - current_y)**2)
        
        if distance < 50:
            pyautogui.moveTo(x, y, duration=0)
        elif use_bezier and distance > 100:
            self._move_bezier(current_x, current_y, x, y, speed_multiplier)
        else:
            duration = min(0.2, distance / 5000) * speed_multiplier
            pyautogui.moveTo(x, y, duration=duration)
        
        time.sleep(random.uniform(0.01, 0.03) * speed_multiplier)
    
    def _move_bezier(self, start_x: int, start_y: int, end_x: int, end_y: int, speed_multiplier: float):
        """Mouvement en courbe de Bézier"""
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        control_x = mid_x + random.randint(-20, 20)
        control_y = mid_y + random.randint(-20, 20)
        
        for t in np.linspace(0, 1, Constants.BEZIER_STEPS):
            x = (1-t)**2 * start_x + 2*(1-t)*t * control_x + t**2 * end_x
            y = (1-t)**2 * start_y + 2*(1-t)*t * control_y + t**2 * end_y
            pyautogui.moveTo(int(x), int(y), duration=0)
            time.sleep(0.001 * speed_multiplier)
    
    def click(self, x: int, y: int):
        """Clic simple"""
        self.move_to(x, y, speed_multiplier=0.5)
        pyautogui.click()
        time.sleep(random.uniform(0.05, 0.1))
    
    def shift_click(self, x: int, y: int):
        """Clic avec Shift"""
        self.move_to(x, y, speed_multiplier=0.5)
        if not keyboard.is_pressed('shift'):
            pyautogui.keyDown('shift')
            time.sleep(0.01)
        pyautogui.click()
        time.sleep(random.uniform(0.05, 0.1))

class TemplateService:
    """Service pour la gestion des templates"""
    def __init__(self, event_manager: EventManager, logger: Logger):
        self.templates: List[Template] = []
        self.event_manager = event_manager
        self.logger = logger
    
    def add_template(self, template: Template) -> bool:
        """Ajouter un template"""
        if not os.path.exists(template.image_path):
            self.logger.error(f"Fichier template introuvable: {template.image_path}")
            return False
        
        self.templates.append(template)
        self.logger.success(f"Template '{template.name}' ajouté")
        return True
    
    def remove_template(self, template: Template):
        """Supprimer un template"""
        if template in self.templates:
            self.templates.remove(template)
            self.logger.info(f"Template '{template.name}' supprimé")
    
    def get_enabled_templates(self) -> List[Template]:
        """Obtenir les templates actifs"""
        return [t for t in self.templates if t.enabled]
    
    def clear_all(self):
        """Supprimer tous les templates"""
        self.templates.clear()

class DetectionService:
    """Service pour la détection d'images"""
    def __init__(self, config: BotConfig, template_service: TemplateService, 
                 event_manager: EventManager, logger: Logger):
        self.config = config
        self.template_service = template_service
        self.event_manager = event_manager
        self.logger = logger
        
        # Système anti-doublons
        self.clicked_positions: Set[Tuple[int, int]] = set()
        self.position_history = deque(maxlen=50)
        self.click_cooldown: Dict[Tuple[int, int], float] = {}
    
    def reset_positions(self):
        """Réinitialiser les positions cliquées"""
        self.clicked_positions.clear()
        self.click_cooldown.clear()
        self.position_history.clear()
        self.logger.info("Positions réinitialisées")
    
    def is_position_valid(self, x: int, y: int) -> bool:
        """Vérifier si une position peut être cliquée"""
        current_time = time.time()
        
        # Nettoyer les anciennes positions
        positions_to_remove = []
        for pos, timestamp in self.click_cooldown.items():
            if current_time - timestamp > self.config.position_timeout:
                positions_to_remove.append(pos)
        
        for pos in positions_to_remove:
            del self.click_cooldown[pos]
            self.clicked_positions.discard(pos)
        
        # Vérifier la distance minimale
        for clicked_x, clicked_y in self.clicked_positions:
            distance = math.sqrt((x - clicked_x)**2 + (y - clicked_y)**2)
            if distance < self.config.min_distance_between_clicks:
                return False
        
        return True
    
    def mark_position_clicked(self, x: int, y: int):
        """Marquer une position comme cliquée"""
        rounded_x = (x // 30) * 30
        rounded_y = (y // 30) * 30
        position = (rounded_x, rounded_y)
        
        self.clicked_positions.add(position)
        self.click_cooldown[position] = time.time()
        self.position_history.append((x, y))
    
    def capture_screen(self) -> np.ndarray:
        """Capturer l'écran"""
        if self.config.detection_region:
            x, y, w, h = self.config.detection_region
            screenshot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        else:
            screenshot = ImageGrab.grab()
        
        screenshot = np.array(screenshot)
        return cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    
    def detect_template(self, screenshot: np.ndarray, template: Template) -> List[Tuple[int, int]]:
        """Détecter un template dans l'écran"""
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
            
            if self.config.detection_region:
                center_x += self.config.detection_region[0]
                center_y += self.config.detection_region[1]
            
            if self.is_position_valid(center_x, center_y):
                matches.append((center_x, center_y))
        
        if matches:
            matches = self._remove_duplicates(matches)
            matches = self._sort_by_distance(matches)
        
        return matches
    
    def _remove_duplicates(self, points: List[Tuple[int, int]], min_distance: int = 30) -> List[Tuple[int, int]]:
        """Supprimer les points trop proches"""
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
    
    def _sort_by_distance(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Trier par distance optimale"""
        if len(points) <= 1:
            return points
        
        current_x, current_y = pyautogui.position()
        sorted_points = []
        remaining = points.copy()
        current_pos = (current_x, current_y)
        
        while remaining:
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
    
    def detect_all(self) -> List[Tuple[Template, List[Tuple[int, int]]]]:
        """Détecter tous les templates actifs"""
        screenshot = self.capture_screen()
        results = []
        
        templates = sorted(self.template_service.get_enabled_templates(), 
                         key=lambda t: t.priority, reverse=True)
        
        for template in templates:
            matches = self.detect_template(screenshot, template)
            if matches:
                results.append((template, matches))
                self.event_manager.emit(EventType.TEMPLATE_DETECTED, 
                                      {'template': template, 'matches': matches})
        
        return results

class ActionService:
    """Service pour l'exécution des actions"""
    def __init__(self, mouse_service: MouseService, detection_service: DetectionService,
                 event_manager: EventManager, logger: Logger):
        self.mouse_service = mouse_service
        self.detection_service = detection_service
        self.event_manager = event_manager
        self.logger = logger
        
        self.action_queue = queue.Queue()
        self.stats = {'clicks': 0, 'detections': 0, 'start_time': None}
    
    def add_action(self, template: Template, position: Tuple[int, int]):
        """Ajouter une action à la queue"""
        self.action_queue.put((template, position))
    
    def execute_click(self, template: Template, position: Tuple[int, int], use_shift: bool = False):
        """Exécuter un clic"""
        x, y = position
        self.detection_service.mark_position_clicked(x, y)
        
        self.logger.info(f"Clic sur {template.name} à ({x}, {y})")
        
        if use_shift:
            self.mouse_service.shift_click(x, y)
        else:
            self.mouse_service.click(x, y)
        
        self.stats['clicks'] += 1
        self.event_manager.emit(EventType.CLICK_EXECUTED, 
                              {'template': template, 'position': position})
    
    def clear_queue(self):
        """Vider la queue d'actions"""
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except queue.Empty:
                break
    
    def reset_stats(self):
        """Réinitialiser les statistiques"""
        self.stats = {'clicks': 0, 'detections': 0, 'start_time': time.time()}

class ConfigService:
    """Service pour la gestion de la configuration"""
    def __init__(self, logger: Logger):
        self.logger = logger
        self._ensure_folders()
    
    def _ensure_folders(self):
        """Créer les dossiers nécessaires"""
        if not os.path.exists(Constants.TEMPLATES_FOLDER):
            os.makedirs(Constants.TEMPLATES_FOLDER)
    
    def save_config(self, config: BotConfig, templates: List[Template]):
        """Sauvegarder la configuration"""
        data = {
            'config': config.to_dict(),
            'templates': [t.to_dict() for t in templates]
        }
        
        try:
            with open(Constants.CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.success("Configuration sauvegardée")
            return True
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde: {e}")
            return False
    
    def load_config(self) -> Tuple[BotConfig, List[Template]]:
        """Charger la configuration"""
        config = BotConfig()
        templates = []
        
        if not os.path.exists(Constants.CONFIG_FILE):
            return config, templates
        
        try:
            with open(Constants.CONFIG_FILE, 'r') as f:
                data = json.load(f)
            
            if 'config' in data:
                config = BotConfig.from_dict(data['config'])
            
            if 'templates' in data:
                templates = [Template.from_dict(t) for t in data['templates']]
                # Filtrer les templates avec fichiers manquants
                templates = [t for t in templates if os.path.exists(t.image_path)]
            
            self.logger.success(f"Configuration chargée: {len(templates)} templates")
            
        except Exception as e:
            self.logger.error(f"Erreur chargement: {e}")
        
        return config, templates

# ================== MAIN CONTROLLER ==================

class BotController:
    """Contrôleur principal du bot"""
    def __init__(self):
        # Système d'événements et logging
        self.event_manager = EventManager()
        self.logger = Logger(self.event_manager)
        
        # Configuration et services
        self.config = BotConfig()
        self.config_service = ConfigService(self.logger)
        
        self.template_service = TemplateService(self.event_manager, self.logger)
        self.mouse_service = MouseService(self.config)
        self.detection_service = DetectionService(self.config, self.template_service, 
                                                 self.event_manager, self.logger)
        self.action_service = ActionService(self.mouse_service, self.detection_service,
                                           self.event_manager, self.logger)
        self.screen_capture_service = ScreenCaptureService(self.logger)
        
        # État du bot
        self.is_running = False
        self.pending_actions = []
        self.threads = []
        
        # Events Shift
        self.last_shift_state = False
        self.shift_released = threading.Event()
        
        self._setup_hotkeys()
        self._load_initial_config()
    
    def _setup_hotkeys(self):
        """Configurer les raccourcis clavier"""
        keyboard.add_hotkey(Constants.HOTKEY_TOGGLE, self.toggle_bot)
        keyboard.add_hotkey(Constants.HOTKEY_STOP, self.stop_bot)
        keyboard.add_hotkey(Constants.HOTKEY_CAPTURE, self.capture_template)
    
    def _load_initial_config(self):
        """Charger la configuration initiale"""
        config, templates = self.config_service.load_config()
        self.config = config
        
        for template in templates:
            self.template_service.add_template(template)
        
        # Appliquer la région de détection
        if self.config.detection_region:
            self.detection_service.config = self.config
    
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
        
        if not self.template_service.get_enabled_templates():
            self.logger.warning("Aucun template actif!")
            return
        
        self.detection_service.reset_positions()
        self.action_service.reset_stats()
        
        self.is_running = True
        self.shift_released.clear()
        self.pending_actions = []
        
        mode_text = "SHIFT" if self.config.mode == BotMode.SHIFT else "AUTO"
        self.logger.success(f"Bot démarré en mode {mode_text}")
        
        # Démarrer les threads
        self.threads = [
            threading.Thread(target=self._detection_loop, daemon=True),
            threading.Thread(target=self._action_loop, daemon=True)
        ]
        
        if self.config.mode == BotMode.SHIFT:
            self.threads.append(threading.Thread(target=self._shift_monitor, daemon=True))
        
        for thread in self.threads:
            thread.start()
        
        self.event_manager.emit(EventType.BOT_STARTED)
    
    def stop_bot(self):
        """Arrêter le bot"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.shift_released.set()
        self.action_service.clear_queue()
        
        self.logger.success("Bot arrêté")
        self._print_stats()
        
        self.event_manager.emit(EventType.BOT_STOPPED)
    
    def _detection_loop(self):
        """Boucle de détection"""
        while self.is_running:
            try:
                detections = self.detection_service.detect_all()
                
                if detections:
                    all_detections = []
                    for template, matches in detections:
                        for match in matches:
                            all_detections.append((template, match))
                    
                    if all_detections:
                        self.pending_actions = all_detections
                        self.action_service.stats['detections'] += len(all_detections)
                        
                        if self.config.mode == BotMode.SHIFT:
                            self.logger.info(f"{len(all_detections)} cibles - En attente SHIFT")
                        else:
                            # Mode AUTO - ajouter immédiatement
                            for template, position in all_detections:
                                self.action_service.add_action(template, position)
                
                time.sleep(random.uniform(Constants.DETECTION_PAUSE_MIN, 
                                        Constants.DETECTION_PAUSE_MAX))
                
            except Exception as e:
                self.logger.error(f"Erreur détection: {e}")
    
    def _shift_monitor(self):
        """Surveillance de la touche Shift"""
        while self.is_running and self.config.mode == BotMode.SHIFT:
            try:
                shift_pressed = keyboard.is_pressed('shift')
                
                if shift_pressed and not self.last_shift_state:
                    self.logger.info("SHIFT activé - Exécution")
                    self.shift_released.clear()
                    self.event_manager.emit(EventType.SHIFT_PRESSED)
                    
                    # Ajouter actions à la queue
                    for template, position in self.pending_actions:
                        self.action_service.add_action(template, position)
                    self.pending_actions = []
                
                elif not shift_pressed and self.last_shift_state:
                    self.logger.info("SHIFT relâché - Arrêt")
                    self.shift_released.set()
                    self.action_service.clear_queue()
                    self.detection_service.reset_positions()
                    self.event_manager.emit(EventType.SHIFT_RELEASED)
                
                self.last_shift_state = shift_pressed
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Erreur monitoring Shift: {e}")
    
    def _action_loop(self):
        """Boucle d'exécution des actions"""
        while self.is_running:
            try:
                # Vérifications mode SHIFT
                if self.config.mode == BotMode.SHIFT:
                    if not keyboard.is_pressed('shift') or self.shift_released.is_set():
                        time.sleep(0.01)
                        continue
                
                # Récupérer action
                try:
                    template, position = self.action_service.action_queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                
                # Vérification finale avant clic
                if self.config.mode == BotMode.SHIFT:
                    if not keyboard.is_pressed('shift') or self.shift_released.is_set():
                        continue
                
                # Exécuter le clic
                use_shift = (self.config.mode == BotMode.SHIFT) or keyboard.is_pressed('shift')
                self.action_service.execute_click(template, position, use_shift)
                
                # Pause entre actions
                time.sleep(random.uniform(0.1, 0.2))
                
            except Exception as e:
                self.logger.error(f"Erreur action: {e}")
    
    def capture_template(self):
        """Capturer un nouveau template"""
        self.logger.info("Mode capture - Interface graphique")
        
        # Utiliser l'interface graphique pour sélectionner la zone
        x1, y1, x2, y2 = self.screen_capture_service.select_screen_area()
        
        if x1 is None:
            self.logger.warning("Capture annulée")
            return
        
        # Générer nom de fichier
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{Constants.TEMPLATES_FOLDER}/template_{timestamp}.png"
        
        # Capturer la zone
        if self.screen_capture_service.capture_area(x1, y1, x2, y2, filename):
            self.logger.success(f"Template sauvegardé: {filename}")
            
            # Optionnel: Demander si on veut ajouter le template immédiatement
            try:
                import tkinter.messagebox as msgbox
                if msgbox.askyesno("Ajouter Template", "Voulez-vous ajouter ce template à la liste ?"):
                    name = simpledialog.askstring("Nom du template", "Entrez un nom:")
                    if name:
                        template = Template(name=name, image_path=filename)
                        self.template_service.add_template(template)
            except:
                pass
    
    def set_detection_region_interactive(self):
        """Définir la région de détection avec interface graphique"""
        self.logger.info("Sélection de zone de détection")
        
        x1, y1, x2, y2 = self.screen_capture_service.select_screen_area()
        
        if x1 is not None:
            self.set_detection_region(x1, y1, x2 - x1, y2 - y1)
        else:
            self.logger.warning("Sélection annulée")
    
    def set_detection_region(self, x: int, y: int, width: int, height: int):
        """Définir la région de détection"""
        self.config.detection_region = (x, y, width, height)
        self.detection_service.config = self.config
        self.logger.success(f"Zone définie: {width}x{height} à ({x},{y})")
    
    def save_config(self):
        """Sauvegarder la configuration"""
        self.config_service.save_config(self.config, self.template_service.templates)
    
    def _print_stats(self):
        """Afficher les statistiques"""
        stats = self.action_service.stats
        if stats['start_time']:
            duration = time.time() - stats['start_time']
            print(f"\n📊 Statistiques:")
            print(f"  Durée: {duration:.1f}s")
            print(f"  Détections: {stats['detections']}")
            print(f"  Clics: {stats['clicks']}")

# ================== GUI ==================

class BotGUI:
    """Interface graphique moderne et modulaire"""
    def __init__(self, controller: BotController):
        self.controller = controller
        self.root = tk.Tk()
        self.mode_var = tk.StringVar(value=self.controller.config.mode.value)
        
        self._setup_window()
        self._setup_ui()
        self._bind_events()
        self._refresh_templates()
    
    def _setup_window(self):
        """Configurer la fenêtre"""
        self.root.title("Pixel Bot v2.0 - Refactorisé")
        self.root.geometry(f"{Constants.WINDOW_WIDTH}x{Constants.WINDOW_HEIGHT}")
    
    def _setup_ui(self):
        """Créer l'interface utilisateur"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self._create_mode_section(main_frame)
        self._create_templates_section(main_frame)
        self._create_control_buttons(main_frame)
        self._create_action_buttons(main_frame)
        self._create_status_bar(main_frame)
    
    def _create_mode_section(self, parent):
        """Section mode de fonctionnement"""
        mode_frame = ttk.LabelFrame(parent, text="Mode de fonctionnement", padding="10")
        mode_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(mode_frame, text="🎮 Mode SHIFT", 
                       variable=self.mode_var, value="shift",
                       command=self._update_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="🤖 Mode AUTO", 
                       variable=self.mode_var, value="auto",
                       command=self._update_mode).pack(anchor=tk.W)
        
        ttk.Label(mode_frame, text="✨ Interface de sélection visuelle comme l'outil de capture d'écran", 
                 font=('Arial', 9), foreground='green').pack(anchor=tk.W, pady=(5, 0))
    
    def _create_templates_section(self, parent):
        """Section templates"""
        ttk.Label(parent, text="Templates", font=('Arial', 14, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=(10, 5))
        
        self.templates_frame = ttk.Frame(parent)
        self.templates_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
    
    def _create_control_buttons(self, parent):
        """Boutons de contrôle"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        ttk.Button(control_frame, text="Ajouter Template", command=self._add_template).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📸 Capturer Zone", command=self._capture_template).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🎯 Définir Zone", command=self._set_zone).pack(side=tk.LEFT, padx=5)
    
    def _create_action_buttons(self, parent):
        """Boutons d'action"""
        action_frame = ttk.Frame(parent)
        action_frame.grid(row=4, column=0, columnspan=3, pady=20)
        
        self.start_button = ttk.Button(action_frame, text="▶ Démarrer", command=self._toggle_bot)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_frame, text="💾 Sauvegarder", command=self._save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="📂 Charger", command=self._load_config).pack(side=tk.LEFT, padx=5)
    
    def _create_status_bar(self, parent):
        """Barre de statut"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Prêt - Architecture refactorisée")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X)
    
    def _bind_events(self):
        """Lier les événements"""
        self.controller.event_manager.subscribe(EventType.STATUS_CHANGED, 
                                              lambda msg: self.status_var.set(msg))
        self.controller.event_manager.subscribe(EventType.BOT_STARTED, 
                                              lambda _: self.start_button.config(text="⏸ Arrêter"))
        self.controller.event_manager.subscribe(EventType.BOT_STOPPED, 
                                              lambda _: self.start_button.config(text="▶ Démarrer"))
    
    def _refresh_templates(self):
        """Rafraîchir la liste des templates"""
        for widget in self.templates_frame.winfo_children():
            widget.destroy()
        
        # Headers
        headers = ["Nom", "Fichier", "Actif", "Seuil", "Actions"]
        for i, header in enumerate(headers):
            ttk.Label(self.templates_frame, text=header, font=('Arial', 10, 'bold')).grid(row=0, column=i, padx=5)
        
        # Templates
        for i, template in enumerate(self.controller.template_service.templates, 1):
            self._create_template_row(i, template)
    
    def _create_template_row(self, row: int, template: Template):
        """Créer une ligne de template"""
        ttk.Label(self.templates_frame, text=template.name).grid(row=row, column=0, padx=5)
        ttk.Label(self.templates_frame, text=os.path.basename(template.image_path)).grid(row=row, column=1, padx=5)
        
        # Checkbox actif
        var = tk.BooleanVar(value=template.enabled)
        var.trace('w', lambda *args: setattr(template, 'enabled', var.get()))
        ttk.Checkbutton(self.templates_frame, variable=var).grid(row=row, column=2, padx=5)
        
        # Seuil
        threshold_var = tk.StringVar(value=str(template.threshold))
        threshold_entry = ttk.Entry(self.templates_frame, textvariable=threshold_var, width=5)
        threshold_entry.grid(row=row, column=3, padx=5)
        threshold_var.trace('w', lambda *args: self._update_threshold(template, threshold_var))
        
        # Bouton supprimer
        ttk.Button(self.templates_frame, text="❌", 
                  command=lambda: self._remove_template(template)).grid(row=row, column=4, padx=5)
    
    def _update_threshold(self, template: Template, var: tk.StringVar):
        """Mettre à jour le seuil"""
        try:
            template.threshold = float(var.get())
        except ValueError:
            pass
    
    def _add_template(self):
        """Ajouter un template"""
        filename = filedialog.askopenfilename(
            title="Sélectionner un template",
            initialdir=Constants.TEMPLATES_FOLDER,
            filetypes=[("Images", "*.png *.jpg *.jpeg")]
        )
        
        if filename:
            name = simpledialog.askstring("Nom", "Nom du template:")
            if name:
                template = Template(name=name, image_path=filename)
                if self.controller.template_service.add_template(template):
                    self._refresh_templates()
    
    def _capture_template(self):
        """Capturer un template avec interface graphique"""
        self.controller.capture_template()
        self._refresh_templates()
    
    def _set_zone(self):
        """Définir la zone de détection avec interface graphique"""
        self.controller.set_detection_region_interactive()
    
    def _reset_positions(self):
        """Réinitialiser les positions"""
        self.controller.detection_service.reset_positions()
    
    def _remove_template(self, template: Template):
        """Supprimer un template"""
        self.controller.template_service.remove_template(template)
        self._refresh_templates()
    
    def _update_mode(self):
        """Mettre à jour le mode"""
        self.controller.config.mode = BotMode(self.mode_var.get())
        if self.controller.is_running:
            self.controller.stop_bot()
            time.sleep(0.5)
            self.controller.start_bot()
    
    def _toggle_bot(self):
        """Toggle bot"""
        self.controller.toggle_bot()
    
    def _save_config(self):
        """Sauvegarder config"""
        self.controller.save_config()
    
    def _load_config(self):
        """Charger config"""
        self.controller._load_initial_config()
        self.mode_var.set(self.controller.config.mode.value)
        self._refresh_templates()
    
    def run(self):
        """Lancer l'interface"""
        self.root.mainloop()

# ================== MAIN ==================

if __name__ == "__main__":
    print("🤖 Pixel Bot v2.0 - Architecture Refactorisée")
    print("=" * 60)
    print("✨ Améliorations:")
    print("  • Architecture modulaire avec services")
    print("  • Système d'événements découplé")
    print("  • Configuration centralisée")
    print("  • Logging structuré")
    print("  • Code maintenable et extensible")
    print("=" * 60)
    
    controller = BotController()
    gui = BotGUI(controller)
    
    # Thread pour exit
    exit_thread = threading.Thread(target=lambda: keyboard.wait(Constants.HOTKEY_EXIT), daemon=True)
    exit_thread.start()
    
    try:
        gui.run()
    except KeyboardInterrupt:
        controller.stop_bot()
        print("\n👋 Arrêt propre du bot")