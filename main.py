#!/usr/bin/env python3
"""
Pixel Bot v3.0 - Architecture Refactorisée
Auteur: Votre nom
Date: 2024

Bot de détection et clic automatique avec interface graphique moderne.
"""

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
from typing import List, Tuple, Optional, Set, Callable, Dict, Any, Protocol
from enum import Enum
import queue
from collections import deque
from abc import ABC, abstractmethod
import logging
from contextlib import contextmanager
from pathlib import Path

# ================== CONFIGURATION & CONSTANTS ==================

class BotMode(Enum):
    SHIFT = "shift"
    AUTO = "auto"

@dataclass
class AppConfig:
    """Configuration principale de l'application"""
    # Fichiers et dossiers
    CONFIG_FILE: Path = Path("bot_config.json")
    TEMPLATES_FOLDER: Path = Path("templates")
    LOG_FILE: Path = Path("bot.log")
    
    # Paramètres souris
    DEFAULT_MOVE_SPEED: float = 0.001
    BEZIER_STEPS: int = 3
    MOUSE_PRECISION_OFFSET: int = 1
    
    # Paramètres détection
    DEFAULT_THRESHOLD: float = 0.9
    DEFAULT_PRIORITY: int = 1
    MIN_DISTANCE_BETWEEN_CLICKS: int = 50
    POSITION_TIMEOUT: float = 10.0
    DETECTION_PAUSE_MIN: float = 0.2
    DETECTION_PAUSE_MAX: float = 0.4
    
    # Interface
    WINDOW_WIDTH: int = 900
    WINDOW_HEIGHT: int = 700
    
    # Raccourcis clavier
    HOTKEY_TOGGLE: str = 'f3'
    HOTKEY_STOP: str = 'f2'
    HOTKEY_CAPTURE: str = 'f10'
    HOTKEY_EXIT: str = 'ctrl+q'
    
    def __post_init__(self):
        """Créer les dossiers nécessaires"""
        self.TEMPLATES_FOLDER.mkdir(exist_ok=True)

@dataclass
class BotConfig:
    """Configuration du bot"""
    mode: BotMode = BotMode.SHIFT
    move_speed: float = 0.001
    min_distance_between_clicks: int = 50
    position_timeout: float = 10.0
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
    threshold: float = 0.9
    priority: int = 1
    click_offset_x: int = 0
    click_offset_y: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Template':
        return cls(**data)
    
    @property
    def exists(self) -> bool:
        """Vérifier si le fichier template existe"""
        return Path(self.image_path).exists()

# ================== PROTOCOLS & INTERFACES ==================

class EventListener(Protocol):
    """Interface pour les écouteurs d'événements"""
    def __call__(self, data: Any = None) -> None: ...

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
    CONFIG_LOADED = "config_loaded"
    CONFIG_SAVED = "config_saved"

class EventBus:
    """Bus d'événements centralisé avec gestion d'erreurs"""
    
    def __init__(self, logger: logging.Logger):
        self._listeners: Dict[EventType, List[EventListener]] = {}
        self._logger = logger
        self._lock = threading.Lock()
    
    def subscribe(self, event_type: EventType, callback: EventListener) -> None:
        """S'abonner à un type d'événement de manière thread-safe"""
        with self._lock:
            if event_type not in self._listeners:
                self._listeners[event_type] = []
            self._listeners[event_type].append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: EventListener) -> None:
        """Se désabonner d'un événement"""
        with self._lock:
            if event_type in self._listeners:
                try:
                    self._listeners[event_type].remove(callback)
                except ValueError:
                    pass
    
    def emit(self, event_type: EventType, data: Any = None) -> None:
        """Émettre un événement avec gestion d'erreurs"""
        with self._lock:
            listeners = self._listeners.get(event_type, []).copy()
        
        for listener in listeners:
            try:
                listener(data)
            except Exception as e:
                self._logger.error(f"Erreur dans event listener {event_type}: {e}")

# ================== LOGGING SYSTEM ==================

class BotLogger:
    """Système de logging avancé"""
    
    def __init__(self, app_config: AppConfig, event_bus: EventBus):
        self.event_bus = event_bus
        self._setup_logging(app_config)
    
    def _setup_logging(self, config: AppConfig):
        """Configurer le système de logging"""
        self.logger = logging.getLogger('PixelBot')
        self.logger.setLevel(logging.INFO)
        
        # Éviter les doublons de handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Handler console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Handler fichier
        try:
            file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"Impossible de créer le fichier de log: {e}")
    
    def info(self, message: str, emit_event: bool = True):
        self.logger.info(message)
        if emit_event:
            self.event_bus.emit(EventType.STATUS_CHANGED, f"ℹ️ {message}")
    
    def success(self, message: str, emit_event: bool = True):
        self.logger.info(f"SUCCESS: {message}")
        if emit_event:
            self.event_bus.emit(EventType.STATUS_CHANGED, f"✅ {message}")
    
    def warning(self, message: str, emit_event: bool = True):
        self.logger.warning(message)
        if emit_event:
            self.event_bus.emit(EventType.STATUS_CHANGED, f"⚠️ {message}")
    
    def error(self, message: str, emit_event: bool = True):
        self.logger.error(message)
        if emit_event:
            self.event_bus.emit(EventType.ERROR_OCCURRED, f"❌ {message}")

# ================== EXCEPTIONS ==================

class BotException(Exception):
    """Exception de base pour le bot"""
    pass

class ConfigurationError(BotException):
    """Erreur de configuration"""
    pass

class TemplateError(BotException):
    """Erreur liée aux templates"""
    pass

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
        self.root.attributes('-alpha', 0.3)
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
            stipple='gray25'
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
    
    def __init__(self, logger: BotLogger):
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

class ConfigService:
    """Service de gestion de la configuration avec validation"""
    
    def __init__(self, app_config: AppConfig, logger: BotLogger, event_bus: EventBus):
        self.app_config = app_config
        self.logger = logger
        self.event_bus = event_bus
    
    def save_config(self, bot_config: BotConfig, templates: List[Template]) -> bool:
        """Sauvegarder la configuration avec validation"""
        try:
            data = {
                'config': bot_config.to_dict(),
                'templates': [t.to_dict() for t in templates]
            }
            
            with open(self.app_config.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.success("Configuration sauvegardée")
            self.event_bus.emit(EventType.CONFIG_SAVED, data)
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde config: {e}")
            return False
    
    def load_config(self) -> Tuple[BotConfig, List[Template]]:
        """Charger la configuration avec validation"""
        bot_config = BotConfig()
        templates = []
        
        if not self.app_config.CONFIG_FILE.exists():
            self.logger.info("Pas de fichier de configuration, utilisation des valeurs par défaut")
            return bot_config, templates
        
        try:
            with open(self.app_config.CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'config' in data:
                bot_config = BotConfig.from_dict(data['config'])
            
            if 'templates' in data:
                valid_templates = []
                for template_data in data['templates']:
                    template = Template.from_dict(template_data)
                    if template.exists:
                        valid_templates.append(template)
                    else:
                        self.logger.warning(f"Template ignoré (fichier manquant): {template.image_path}")
                
                templates = valid_templates
            
            self.logger.success(f"Configuration chargée: {len(templates)} templates")
            self.event_bus.emit(EventType.CONFIG_LOADED, {'config': bot_config, 'templates': templates})
            
        except Exception as e:
            self.logger.error(f"Erreur chargement config: {e}")
        
        return bot_config, templates

class MouseService:
    """Service de gestion de la souris avec amélioration des mouvements"""
    
    def __init__(self, config: BotConfig, logger: BotLogger):
        self.config = config
        self.logger = logger
        self._setup_pyautogui()
        self._movement_lock = threading.Lock()
    
    def _setup_pyautogui(self):
        """Configuration de pyautogui"""
        pyautogui.FAILSAFE = True
        pyautogui.MINIMUM_DURATION = 0
        pyautogui.PAUSE = 0
    
    def move_to(self, x: int, y: int, use_bezier: bool = True, speed_multiplier: float = 1.0) -> None:
        """Déplacer la souris avec mouvement naturel"""
        with self._movement_lock:
            try:
                current_x, current_y = pyautogui.position()
                
                # Ajouter imprécision humaine
                target_x = x + random.randint(-1, 1)
                target_y = y + random.randint(-1, 1)
                
                distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
                
                if distance < 5:
                    return
                elif distance < 50:
                    pyautogui.moveTo(target_x, target_y, duration=0)
                elif use_bezier and distance > 100:
                    self._move_bezier(current_x, current_y, target_x, target_y, speed_multiplier)
                else:
                    duration = min(0.3, distance / 3000) * speed_multiplier
                    pyautogui.moveTo(target_x, target_y, duration=duration)
                
                # Pause naturelle
                time.sleep(random.uniform(0.01, 0.03) * speed_multiplier)
                
            except Exception as e:
                self.logger.error(f"Erreur mouvement souris: {e}")
    
    def _move_bezier(self, start_x: int, start_y: int, end_x: int, end_y: int, speed_multiplier: float):
        """Mouvement en courbe de Bézier plus naturel"""
        # Points de contrôle avec variabilité
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        # Variation du point de contrôle basée sur la distance
        distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        control_variance = min(50, distance / 10)
        
        control_x = mid_x + random.uniform(-control_variance, control_variance)
        control_y = mid_y + random.uniform(-control_variance, control_variance)
        
        steps = max(5, int(distance / 50))
        
        for t in np.linspace(0, 1, steps):
            x = (1-t)**2 * start_x + 2*(1-t)*t * control_x + t**2 * end_x
            y = (1-t)**2 * start_y + 2*(1-t)*t * control_y + t**2 * end_y
            pyautogui.moveTo(int(x), int(y), duration=0)
            time.sleep(0.002 * speed_multiplier)
    
    def click(self, x: int, y: int) -> None:
        """Clic simple avec validation"""
        try:
            self.move_to(x, y, speed_multiplier=0.8)
            pyautogui.click()
            time.sleep(random.uniform(0.05, 0.15))
        except Exception as e:
            self.logger.error(f"Erreur clic: {e}")
    
    def shift_click(self, x: int, y: int) -> None:
        """Clic avec Shift maintenu"""
        try:
            self.move_to(x, y, speed_multiplier=0.8)
            
            shift_already_pressed = keyboard.is_pressed('shift')
            if not shift_already_pressed:
                pyautogui.keyDown('shift')
                time.sleep(0.01)
            
            pyautogui.click()
            time.sleep(random.uniform(0.05, 0.15))
            
            if not shift_already_pressed:
                pyautogui.keyUp('shift')
                
        except Exception as e:
            self.logger.error(f"Erreur shift-clic: {e}")

class TemplateManager:
    """Gestionnaire de templates avec validation et cache"""
    
    def __init__(self, event_bus: EventBus, logger: BotLogger):
        self._templates: List[Template] = []
        self._template_cache: Dict[str, np.ndarray] = {}
        self.event_bus = event_bus
        self.logger = logger
        self._lock = threading.RLock()
    
    @property
    def templates(self) -> List[Template]:
        """Obtenir la liste des templates (lecture seule)"""
        with self._lock:
            return self._templates.copy()
    
    def add_template(self, template: Template) -> bool:
        """Ajouter un template avec validation"""
        try:
            if not template.exists:
                raise TemplateError(f"Fichier template introuvable: {template.image_path}")
            
            # Tester le chargement de l'image
            self._load_template_image(template)
            
            with self._lock:
                base_name = template.name
                counter = 1
                while any(t.name == template.name for t in self._templates):
                    template.name = f"{base_name}{counter}"
                    counter += 1
                
                self._templates.append(template)
            
            self.logger.success(f"Template '{template.name}' ajouté")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur ajout template: {e}")
            return False
    
    def remove_template(self, template: Template) -> bool:
        """Supprimer un template"""
        with self._lock:
            if template in self._templates:
                self._templates.remove(template)
                # Nettoyer le cache
                cache_key = template.image_path
                self._template_cache.pop(cache_key, None)
                
                self.logger.info(f"Template '{template.name}' supprimé")
                return True
        return False
    
    def get_enabled_templates(self) -> List[Template]:
        """Obtenir les templates actifs"""
        with self._lock:
            return [t for t in self._templates if t.enabled and t.exists]
    
    def get_template_image(self, template: Template) -> Optional[np.ndarray]:
        """Obtenir l'image d'un template avec cache"""
        cache_key = template.image_path
        
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]
        
        try:
            image = self._load_template_image(template)
            self._template_cache[cache_key] = image
            return image
        except Exception as e:
            self.logger.error(f"Erreur chargement image template {template.name}: {e}")
            return None
    
    def _load_template_image(self, template: Template) -> np.ndarray:
        """Charger l'image d'un template"""
        image = cv2.imread(template.image_path)
        if image is None:
            raise TemplateError(f"Impossible de charger l'image: {template.image_path}")
        return image
    
    def clear_cache(self):
        """Vider le cache des images"""
        self._template_cache.clear()
        self.logger.info("Cache des templates vidé")
    
    def clear_all(self):
        """Supprimer tous les templates"""
        with self._lock:
            self._templates.clear()
            self.clear_cache()

class DetectionService:
    """Service pour la détection d'images"""
    
    def __init__(self, config: BotConfig, template_manager: TemplateManager, 
                 event_bus: EventBus, logger: BotLogger):
        self.config = config
        self.template_manager = template_manager
        self.event_bus = event_bus
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
        try:
            if self.config.detection_region:
                x, y, w, h = self.config.detection_region
                screenshot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            else:
                screenshot = ImageGrab.grab()
            
            screenshot = np.array(screenshot)
            return cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.logger.error(f"Erreur capture écran: {e}")
            return np.array([])
    
    def detect_template(self, screenshot: np.ndarray, template: Template) -> List[Tuple[int, int]]:
        """Détecter un template dans l'écran"""
        if not template.enabled or screenshot.size == 0:
            return []
        
        template_img = self.template_manager.get_template_image(template)
        if template_img is None:
            return []
        
        try:
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
            
        except Exception as e:
            self.logger.error(f"Erreur détection template {template.name}: {e}")
            return []
    
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
        
        try:
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
        except Exception as e:
            self.logger.error(f"Erreur tri par distance: {e}")
            return points
    
    def detect_all(self) -> List[Tuple[Template, List[Tuple[int, int]]]]:
        """Détecter tous les templates actifs"""
        screenshot = self.capture_screen()
        if screenshot.size == 0:
            return []
            
        results = []
        
        templates = sorted(self.template_manager.get_enabled_templates(), 
                         key=lambda t: t.priority, reverse=True)
        
        for template in templates:
            matches = self.detect_template(screenshot, template)
            if matches:
                results.append((template, matches))
                self.event_bus.emit(EventType.TEMPLATE_DETECTED, 
                                  {'template': template, 'matches': matches})
        
        return results

class ActionService:
    """Service pour l'exécution des actions"""
    
    def __init__(self, mouse_service: MouseService, detection_service: DetectionService,
                 event_bus: EventBus, logger: BotLogger):
        self.mouse_service = mouse_service
        self.detection_service = detection_service
        self.event_bus = event_bus
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
        self.event_bus.emit(EventType.CLICK_EXECUTED, 
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

# ================== MAIN CONTROLLER ==================

class BotController:
    """Contrôleur principal avec gestion d'état améliorée"""
    
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        
        # Initialisation des composants de base
        self.event_bus = EventBus(logging.getLogger('PixelBot'))
        self.logger = BotLogger(app_config, self.event_bus)
        
        # Configuration
        self.bot_config = BotConfig()
        self.config_service = ConfigService(app_config, self.logger, self.event_bus)
        
        # Services
        self.template_manager = TemplateManager(self.event_bus, self.logger)
        self.mouse_service = MouseService(self.bot_config, self.logger)
        self.detection_service = DetectionService(self.bot_config, self.template_manager, 
                                                 self.event_bus, self.logger)
        self.action_service = ActionService(self.mouse_service, self.detection_service,
                                           self.event_bus, self.logger)
        self.screen_capture_service = ScreenCaptureService(self.logger)
        
        # État du bot
        self._running = False
        self._activated = False
        self._threads: List[threading.Thread] = []
        self._shutdown_event = threading.Event()
        self.pending_actions = []
        
        self._setup_event_handlers()
        self._setup_hotkeys()
        self._load_initial_config()
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def is_activated(self) -> bool:
        return self._activated
    
    def _setup_event_handlers(self):
        """Configurer les gestionnaires d'événements"""
        self.event_bus.subscribe(EventType.CONFIG_LOADED, self._on_config_loaded)
    
    def _on_config_loaded(self, data: Dict[str, Any]):
        """Gestionnaire de chargement de configuration"""
        if data and 'config' in data:
            self.bot_config = data['config']
            self.mouse_service.config = self.bot_config
            self.detection_service.config = self.bot_config
    
    def _setup_hotkeys(self):
        """Configurer les raccourcis clavier"""
        try:
            keyboard.add_hotkey(self.app_config.HOTKEY_TOGGLE, self.toggle_bot)
            keyboard.add_hotkey(self.app_config.HOTKEY_STOP, self.stop_bot)
            keyboard.add_hotkey(self.app_config.HOTKEY_CAPTURE, self.capture_template)
        except Exception as e:
            self.logger.error(f"Erreur configuration hotkeys: {e}")
    
    def _load_initial_config(self):
        """Charger la configuration initiale"""
        try:
            bot_config, templates = self.config_service.load_config()
            self.bot_config = bot_config
            
            for template in templates:
                self.template_manager.add_template(template)
                
        except Exception as e:
            self.logger.error(f"Erreur configuration: {e}")
    
    def start_bot(self) -> bool:
        """Démarrer le bot avec validation"""
        if self._running:
            self.logger.warning("Bot déjà en cours d'exécution")
            return False
        
        enabled_templates = self.template_manager.get_enabled_templates()
        if not enabled_templates:
            self.logger.error("Aucun template actif!")
            return False
        
        try:
            self._running = True
            self._activated = False
            self._shutdown_event.clear()
            self.pending_actions = []
            
            self.detection_service.reset_positions()
            self.action_service.reset_stats()
            
            mode_text = "SHIFT+ESPACE" if self.bot_config.mode == BotMode.SHIFT else "AUTO"
            self.logger.success(f"Bot démarré en mode {mode_text}")
            self.event_bus.emit(EventType.BOT_STARTED)
            
            # Démarrer les threads de travail
            self._start_worker_threads()
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur démarrage bot: {e}")
            self._running = False
            return False
    
    def stop_bot(self) -> bool:
        """Arrêter le bot proprement"""
        if not self._running:
            return True
        
        try:
            self._running = False
            self._activated = False
            self._shutdown_event.set()
            
            self.action_service.clear_queue()
            
            # Attendre les threads
            for thread in self._threads:
                if thread.is_alive():
                    thread.join(timeout=2.0)
            
            self._threads.clear()
            
            self.logger.success("Bot arrêté")
            self._print_stats()
            self.event_bus.emit(EventType.BOT_STOPPED)
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur arrêt bot: {e}")
            return False
    
    def toggle_bot(self) -> bool:
        """Basculer l'état du bot"""
        if self._running:
            return self.stop_bot()
        else:
            return self.start_bot()
    
    def _start_worker_threads(self):
        """Démarrer les threads de travail"""
        self._threads = [
            threading.Thread(target=self._detection_worker, daemon=True, name="Detection"),
            threading.Thread(target=self._action_worker, daemon=True, name="Action")
        ]
        
        if self.bot_config.mode == BotMode.SHIFT:
            self._threads.append(
                threading.Thread(target=self._shift_monitor, daemon=True, name="ShiftMonitor")
            )
        
        for thread in self._threads:
            thread.start()
    
    def _detection_worker(self):
        """Worker de détection"""
        while self._running and not self._shutdown_event.is_set():
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
                        
                        if self.bot_config.mode == BotMode.SHIFT:
                            self.logger.info(f"{len(all_detections)} cibles - En attente SHIFT + ESPACE")
                        else:
                            # Mode AUTO - ajouter immédiatement
                            for template, position in all_detections:
                                self.action_service.add_action(template, position)
                
                time.sleep(random.uniform(self.app_config.DETECTION_PAUSE_MIN, 
                                        self.app_config.DETECTION_PAUSE_MAX))
                
            except Exception as e:
                self.logger.error(f"Erreur détection: {e}")
    
    def _action_worker(self):
        """Worker d'exécution des actions"""
        while self._running and not self._shutdown_event.is_set():
            try:
                # Vérifications mode SHIFT
                if self.bot_config.mode == BotMode.SHIFT:
                    # Vérifier que SHIFT est pressé ET que le bot est activé
                    if not keyboard.is_pressed('shift') or not self._activated or self._shutdown_event.is_set():
                        time.sleep(0.01)
                        continue
                
                # Récupérer action
                try:
                    template, position = self.action_service.action_queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                
                # Vérification finale avant clic
                if self.bot_config.mode == BotMode.SHIFT:
                    if not keyboard.is_pressed('shift') or not self._activated or self._shutdown_event.is_set():
                        continue
                
                # Exécuter le clic
                use_shift = (self.bot_config.mode == BotMode.SHIFT and self._activated) or keyboard.is_pressed('shift')
                self.action_service.execute_click(template, position, use_shift)
                
                # Pause entre actions
                time.sleep(random.uniform(0.1, 0.2))
                
            except Exception as e:
                self.logger.error(f"Erreur action: {e}")
    
    def _shift_monitor(self):
        """Surveillance de la combinaison Shift + Espace"""
        while self._running and self.bot_config.mode == BotMode.SHIFT and not self._shutdown_event.is_set():
            try:
                shift_pressed = keyboard.is_pressed('shift')
                space_pressed = keyboard.is_pressed('space')
                shift_space_combo = shift_pressed and space_pressed
                
                # Déclenchement : SHIFT + ESPACE pressés et bot pas encore activé
                if shift_space_combo and not self._activated:
                    self.logger.info("SHIFT + ESPACE activé - Exécution")
                    self._activated = True
                    self.event_bus.emit(EventType.SHIFT_PRESSED)
                    
                    # Ajouter actions à la queue
                    for template, position in self.pending_actions:
                        self.action_service.add_action(template, position)
                    self.pending_actions = []
                
                # Arrêt : SHIFT relâché (peu importe l'état d'ESPACE) et bot était activé
                elif not shift_pressed and self._activated:
                    self.logger.info("SHIFT relâché - Arrêt")
                    self._activated = False
                    self.action_service.clear_queue()
                    self.detection_service.reset_positions()
                    self.event_bus.emit(EventType.SHIFT_RELEASED)
                
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Erreur monitoring Shift+Espace: {e}")
    
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
        filename = f"{self.app_config.TEMPLATES_FOLDER}/template_{timestamp}.png"
        
        # Capturer la zone
        if self.screen_capture_service.capture_area(x1, y1, x2, y2, filename):
            self.logger.success(f"Template sauvegardé: {filename}")
            
            # Optionnel: Demander si on veut ajouter le template immédiatement
            try:
                name = simpledialog.askstring("Nom du template", "Entrez un nom:")
                if name:
                    template = Template(name=name, image_path=filename)
                    self.template_manager.add_template(template)
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
        self.bot_config.detection_region = (x, y, width, height)
        self.detection_service.config = self.bot_config
        self.logger.success(f"Zone définie: {width}x{height} à ({x},{y})")
    
    def save_config(self) -> bool:
        """Sauvegarder la configuration"""
        return self.config_service.save_config(self.bot_config, self.template_manager.templates)
    
    def _print_stats(self):
        """Afficher les statistiques"""
        stats = self.action_service.stats
        if stats['start_time']:
            duration = time.time() - stats['start_time']
            print(f"\n📊 Statistiques:")
            print(f"  Durée: {duration:.1f}s")
            print(f"  Détections: {stats['detections']}")
            print(f"  Clics: {stats['clicks']}")
    
    def shutdown(self):
        """Arrêt complet de l'application"""
        self.stop_bot()
        self.logger.info("Arrêt de l'application")

# ================== GUI ==================

class BotGUI:
    """Interface graphique moderne et modulaire"""
    
    def __init__(self, controller: BotController):
        self.controller = controller
        self.root = tk.Tk()
        self.mode_var = tk.StringVar(value=self.controller.bot_config.mode.value)
        
        self._setup_window()
        self._setup_ui()
        self._bind_events()
        self._refresh_templates()
    
    def _setup_window(self):
        """Configurer la fenêtre"""
        self.root.title("Pixel Bot v3.0 - Architecture Refactorisée")
        self.root.geometry(f"{self.controller.app_config.WINDOW_WIDTH}x{self.controller.app_config.WINDOW_HEIGHT}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Configurer la grille principale
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def _setup_ui(self):
        """Créer l'interface utilisateur"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        
        self._create_mode_section(main_frame)
        self._create_control_section(main_frame)
        self._create_templates_section(main_frame)
        self._create_action_buttons(main_frame)
        self._create_status_bar(main_frame)
    
    def _create_mode_section(self, parent):
        """Section mode de fonctionnement"""
        mode_frame = ttk.LabelFrame(parent, text="Mode de fonctionnement", padding="10")
        mode_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Radiobutton(mode_frame, text="🎮 Mode SHIFT + ESPACE", 
                       variable=self.mode_var, value="shift",
                       command=self._update_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="🤖 Mode AUTO", 
                       variable=self.mode_var, value="auto",
                       command=self._update_mode).pack(anchor=tk.W)
        
        # Description
        desc_text = "SHIFT + ESPACE : Déclenche le bot | Lâcher SHIFT : Arrête le bot"
        ttk.Label(mode_frame, text=desc_text, font=('Arial', 9), 
                 foreground='blue').pack(anchor=tk.W, pady=(5, 0))
    
    def _create_control_section(self, parent):
        """Section contrôles"""
        control_frame = ttk.LabelFrame(parent, text="Contrôles", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(fill=tk.X)
        
        ttk.Button(buttons_frame, text="📸 Capturer Template", 
                  command=self._capture_template).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buttons_frame, text="🎯 Définir Zone", 
                  command=self._set_zone).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="🔄 Reset Positions", 
                  command=self._reset_positions).pack(side=tk.LEFT, padx=5)
        
        # Info zone de détection
        if self.controller.bot_config.detection_region:
            region = self.controller.bot_config.detection_region
            zone_text = f"Zone actuelle: {region[2]}x{region[3]} à ({region[0]},{region[1]})"
        else:
            zone_text = "Zone: Écran complet"
        
        self.zone_label = ttk.Label(control_frame, text=zone_text, font=('Arial', 9))
        self.zone_label.pack(pady=(5, 0))
    
    def _create_templates_section(self, parent):
        """Section templates"""
        templates_frame = ttk.LabelFrame(parent, text="Templates", padding="10")
        templates_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        templates_frame.columnconfigure(0, weight=1)
        templates_frame.rowconfigure(1, weight=1)
        
        # Boutons templates
        template_buttons = ttk.Frame(templates_frame)
        template_buttons.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Button(template_buttons, text="➕ Ajouter Template", 
                  command=self._add_template).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(template_buttons, text="🗂️ Charger depuis fichier", 
                  command=self._load_config).pack(side=tk.LEFT, padx=5)
        
        # Liste templates avec scrollbar
        list_frame = ttk.Frame(templates_frame)
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Scrollable frame pour templates
        canvas = tk.Canvas(list_frame)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        self.templates_frame = ttk.Frame(canvas)
        
        self.templates_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.templates_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    def _create_action_buttons(self, parent):
        """Boutons d'action"""
        action_frame = ttk.Frame(parent)
        action_frame.grid(row=3, column=0, pady=(0, 10))
        
        self.start_button = ttk.Button(action_frame, text="▶ Démarrer", 
                                      command=self._toggle_bot, 
                                      style="Accent.TButton")
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(action_frame, text="💾 Sauvegarder", 
                  command=self._save_config).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_frame, text="📊 Statistiques", 
                  command=self._show_stats).pack(side=tk.LEFT, padx=5)
    
    def _create_status_bar(self, parent):
        """Barre de statut"""
        self.status_var = tk.StringVar(value="Prêt - Architecture refactorisée v3.0")
        status_bar = ttk.Label(parent, textvariable=self.status_var, 
                              relief=tk.SUNKEN, padding="5")
        status_bar.grid(row=4, column=0, sticky=(tk.W, tk.E))
    
    def _bind_events(self):
        """Lier les événements"""
        self.controller.event_bus.subscribe(EventType.STATUS_CHANGED, self._update_status)
        self.controller.event_bus.subscribe(EventType.BOT_STARTED, 
                                          lambda _: self.start_button.config(text="⏸ Arrêter"))
        self.controller.event_bus.subscribe(EventType.BOT_STOPPED, 
                                          lambda _: self.start_button.config(text="▶ Démarrer"))
        self.controller.event_bus.subscribe(EventType.CONFIG_LOADED, 
                                          lambda _: self._refresh_templates())
    
    def _refresh_templates(self):
        """Rafraîchir la liste des templates"""
        # Nettoyer les widgets existants
        for widget in self.templates_frame.winfo_children():
            widget.destroy()
        
        templates = self.controller.template_manager.templates
        if not templates:
            ttk.Label(self.templates_frame, 
                     text="Aucun template. Cliquez sur 'Capturer Template' pour commencer.",
                     foreground='gray').pack(pady=20)
            return
        
        # Headers
        header_frame = ttk.Frame(self.templates_frame)
        header_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(header_frame, text="Nom", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(header_frame, text="Actif", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(header_frame, text="Seuil", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(header_frame, text="Actions", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        
        # Separator
        ttk.Separator(self.templates_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Templates
        for template in templates:
            self._create_template_row(template)
        
        # Mettre à jour la zone de détection
        self._update_zone_label()
    
    def _create_template_row(self, template: Template):
        """Créer une ligne de template"""
        row_frame = ttk.Frame(self.templates_frame)
        row_frame.pack(fill=tk.X, pady=2)
        
        # Nom du template (avec indicateur d'existence)
        name_text = template.name
        if not template.exists:
            name_text += " ❌"
        
        name_label = ttk.Label(row_frame, text=name_text, width=20)
        name_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # Checkbox actif
        enabled_var = tk.BooleanVar(value=template.enabled)
        enabled_var.trace('w', lambda *args: setattr(template, 'enabled', enabled_var.get()))
        ttk.Checkbutton(row_frame, variable=enabled_var).pack(side=tk.LEFT, padx=(0, 30))
        
        # Seuil
        threshold_var = tk.StringVar(value=f"{template.threshold:.2f}")
        threshold_entry = ttk.Entry(row_frame, textvariable=threshold_var, width=8)
        threshold_entry.pack(side=tk.LEFT, padx=(0, 20))
        threshold_var.trace('w', lambda *args: self._update_threshold(template, threshold_var))
        
        # Boutons d'action
        actions_frame = ttk.Frame(row_frame)
        actions_frame.pack(side=tk.LEFT)
        
        ttk.Button(actions_frame, text="🗑️", width=3,
                  command=lambda: self._remove_template(template)).pack(side=tk.LEFT, padx=2)
        
        if template.exists:
            ttk.Button(actions_frame, text="👁️", width=3,
                      command=lambda: self._preview_template(template)).pack(side=tk.LEFT, padx=2)
    
    def _update_threshold(self, template: Template, var: tk.StringVar):
        """Mettre à jour le seuil"""
        try:
            value = float(var.get())
            if 0.0 <= value <= 1.0:
                template.threshold = value
        except ValueError:
            pass
    
    def _preview_template(self, template: Template):
        """Prévisualiser un template"""
        try:
            img = Image.open(template.image_path)
            img.show()
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir l'image: {e}")
    
    def _add_template(self):
        """Ajouter un template"""
        filename = filedialog.askopenfilename(
            title="Sélectionner un template",
            initialdir=str(self.controller.app_config.TEMPLATES_FOLDER),
            filetypes=[("Images", "*.png *.jpg *.jpeg")]
        )
        
        if filename:
            name = simpledialog.askstring("Nom du template", "Nom du template:")
            if name:
                template = Template(name=name, image_path=filename)
                if self.controller.template_manager.add_template(template):
                    self._refresh_templates()
    
    def _remove_template(self, template: Template):
        """Supprimer un template"""
        if messagebox.askyesno("Confirmation", f"Supprimer le template '{template.name}' ?"):
            self.controller.template_manager.remove_template(template)
            self._refresh_templates()
    
    def _capture_template(self):
        """Capturer un template"""
        self.controller.capture_template()
        self._refresh_templates()
    
    def _set_zone(self):
        """Définir la zone de détection"""
        self.controller.set_detection_region_interactive()
        self._update_zone_label()
    
    def _reset_positions(self):
        """Réinitialiser les positions"""
        self.controller.detection_service.reset_positions()
    
    def _update_mode(self):
        """Mettre à jour le mode"""
        self.controller.bot_config.mode = BotMode(self.mode_var.get())
        if self.controller.is_running:
            self.controller.stop_bot()
            time.sleep(0.5)
            self.controller.start_bot()
    
    def _toggle_bot(self):
        """Toggle bot"""
        self.controller.toggle_bot()
    
    def _save_config(self):
        """Sauvegarder config"""
        if self.controller.save_config():
            messagebox.showinfo("Succès", "Configuration sauvegardée!")
    
    def _load_config(self):
        """Charger config"""
        self.controller._load_initial_config()
        self.mode_var.set(self.controller.bot_config.mode.value)
        self._refresh_templates()
    
    def _show_stats(self):
        """Afficher les statistiques"""
        stats = self.controller.action_service.stats
        if stats['start_time']:
            duration = time.time() - stats['start_time']
            stats_text = f"""Statistiques de session:
            
Durée: {duration:.1f} secondes
Détections: {stats['detections']}
Clics exécutés: {stats['clicks']}
Templates actifs: {len(self.controller.template_manager.get_enabled_templates())}
Mode: {self.controller.bot_config.mode.value.upper()}"""
        else:
            stats_text = "Aucune session active."
        
        messagebox.showinfo("Statistiques", stats_text)
    
    def _update_status(self, message: str):
        """Mettre à jour le statut"""
        self.status_var.set(message)
    
    def _update_zone_label(self):
        """Mettre à jour le label de la zone"""
        if hasattr(self, 'zone_label'):
            if self.controller.bot_config.detection_region:
                region = self.controller.bot_config.detection_region
                zone_text = f"Zone: {region[2]}x{region[3]} à ({region[0]},{region[1]})"
            else:
                zone_text = "Zone: Écran complet"
            self.zone_label.config(text=zone_text)
    
    def _on_closing(self):
        """Gestionnaire de fermeture de fenêtre"""
        if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter ?"):
            self.controller.shutdown()
            self.root.destroy()
    
    def run(self):
        """Lancer l'interface"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.controller.shutdown()

# ================== MAIN ==================

def main():
    """Point d'entrée principal"""
    print("🤖 Pixel Bot v3.0 - Architecture Refactorisée")
    print("=" * 60)
    print("✨ Améliorations v3.0:")
    print("  • Gestion d'erreurs robuste")
    print("  • Logging avancé avec fichiers")
    print("  • Services avec interfaces")
    print("  • Cache des templates optimisé")
    print("  • Configuration validée")
    print("  • Threads workers sécurisés")
    print("  • Event bus thread-safe")
    print("  • Interface graphique améliorée")
    print("  • Capture d'écran interactive")
    print("=" * 60)
    print(f"📂 Dossier templates: {Path('templates').absolute()}")
    print(f"⚙️ Fichier config: {Path('bot_config.json').absolute()}")
    print(f"📋 Fichier log: {Path('bot.log').absolute()}")
    print("=" * 60)
    
    try:
        app_config = AppConfig()
        controller = BotController(app_config)
        gui = BotGUI(controller)
        
        # Thread de surveillance d'arrêt
        def exit_monitor():
            try:
                keyboard.wait(app_config.HOTKEY_EXIT)
                controller.shutdown()
            except:
                pass
        
        exit_thread = threading.Thread(target=exit_monitor, daemon=True, name="ExitMonitor")
        exit_thread.start()
        
        print("🚀 Interface graphique lancée!")
        print("=" * 60)
        print("💡 Raccourcis clavier:")
        print(f"  {app_config.HOTKEY_TOGGLE} - Toggle bot")
        print(f"  {app_config.HOTKEY_STOP} - Arrêter bot")
        print(f"  {app_config.HOTKEY_CAPTURE} - Capturer template")
        print(f"  {app_config.HOTKEY_EXIT} - Quitter")
        print("=" * 60)
        
        gui.run()
        
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        logging.exception("Erreur fatale de l'application")

if __name__ == "__main__":
    main()