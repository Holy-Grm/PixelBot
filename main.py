import pyautogui
import time
import random
import math
import threading
import keyboard
import os
import re

class Mouse:
    def __init__(self):
        self.move_speed = 0.1  # Durée du mouvement
        pyautogui.FAILSAFE = True  # Sécurité PyAutoGUI
    
    def move_to(self, x, y):
        """Déplacer vers une position avec mouvement naturel"""
        current_x, current_y = pyautogui.position()
        
        # Calcul distance
        delta_x = x - current_x
        delta_y = y - current_y
        distance = math.sqrt(delta_x * delta_x + delta_y * delta_y)
        
        # Point intermédiaire pour longues distances
        if distance > 100:
            rand_x = random.randint(-15, 15)
            rand_y = random.randint(-15, 15)
            mid_x = current_x + (delta_x * 0.6) + rand_x
            mid_y = current_y + (delta_y * 0.6) + rand_y
            pyautogui.moveTo(mid_x, mid_y, duration=self.move_speed)
            self.random_pause(0.03, 0.1)
        
        # Mouvement final avec légère imprécision
        rand_x = random.randint(-2, 2)
        rand_y = random.randint(-2, 2)
        pyautogui.moveTo(x + rand_x, y + rand_y, duration=self.move_speed)
        self.random_pause(0.05, 0.15)
    
    def click(self, x, y):
        """Clic simple"""
        self.move_to(x, y)
        pyautogui.mouseDown()
        self.random_pause(0.01, 0.03)
        pyautogui.mouseUp()
        self.random_pause(0.05, 0.15)
    
    def shift_click(self, x, y):
        """Clic avec Shift"""
        self.move_to(x, y)
        pyautogui.keyDown('shift')
        time.sleep(0.01)
        pyautogui.mouseDown()
        self.random_pause(0.01, 0.03)
        pyautogui.mouseUp()
        pyautogui.keyUp('shift')
        self.random_pause(0.05, 0.15)
    
    def random_pause(self, min_time, max_time):
        """Pause aléatoire"""
        pause_time = random.uniform(min_time, max_time)
        time.sleep(pause_time)

class Sequence:
    def __init__(self):
        self.actions = []
        self.is_running = False
        self.should_stop = False
    
    def add(self, action_type, x=0, y=0, param=""):
        """Ajouter une action"""
        action = {
            'type': action_type,
            'x': x,
            'y': y,
            'param': param
        }
        self.actions.append(action)
    
    def load_from_file(self, filename):
        """Charger depuis un fichier"""
        self.clear()
        
        if not os.path.exists(filename):
            print(f"Erreur: Fichier non trouvé - {filename}")
            return False
        
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.parse_line(line)
            print(f"Séquence chargée: {len(self.actions)} actions depuis {filename}")
            return True
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier: {e}")
            return False
    
    def parse_line(self, line):
        """Parser une ligne"""
        # shift clic (x,y)
        match = re.search(r'shift\s+clic\s*\((\d+)\s*,\s*(\d+)\)', line, re.IGNORECASE)
        if match:
            self.add("shift_click", int(match.group(1)), int(match.group(2)))
            return
        
        # clic (x,y)
        match = re.search(r'clic\s*\((\d+)\s*,\s*(\d+)\)', line, re.IGNORECASE)
        if match:
            self.add("click", int(match.group(1)), int(match.group(2)))
            return
        
        # pause (ms) ou pause (min,max)
        match = re.search(r'pause\s*\((\d+)(?:\s*,\s*(\d+))?\)', line, re.IGNORECASE)
        if match:
            min_time = int(match.group(1))
            max_time = int(match.group(2)) if match.group(2) else min_time
            self.add("pause", min_time, max_time)
            return
    
    def execute(self, loop=False):
        """Exécuter la séquence"""
        if len(self.actions) == 0:
            print("Attention: Aucune action à exécuter!")
            return
        
        self.is_running = True
        self.should_stop = False
        mouse = Mouse()
        
        print("=" * 50)
        print("🤖 BOT DÉMARRÉ")
        print("=" * 50)
        print("Appuyez sur F2 pour arrêter le bot")
        
        try:
            cycle = 1
            while True:
                if loop:
                    print(f"\n--- Cycle {cycle} ---")
                
                for index, action in enumerate(self.actions):
                    print(f"Action {index+1}/{len(self.actions)}: ", end="")
                    
                    if not self.is_running or self.should_stop:
                        print("\n❌ Bot arrêté par l'utilisateur")
                        self.stop()
                        return
                    
                    # Exécuter l'action
                    if action['type'] == "click":
                        print(f"Clic en ({action['x']}, {action['y']})")
                        mouse.click(action['x'], action['y'])
                    elif action['type'] == "shift_click":
                        print(f"Shift+Clic en ({action['x']}, {action['y']})")
                        mouse.shift_click(action['x'], action['y'])
                    elif action['type'] == "pause":
                        sleep_time = random.randint(action['x'], action['y']) / 1000.0
                        print(f"Pause {sleep_time:.2f}s")
                        time.sleep(sleep_time)
                    
                    # Pause entre actions
                    mouse.random_pause(0.1, 0.25)
                
                if not loop:
                    break
                
                cycle += 1
                # Pause entre cycles
                print("Pause entre cycles...")
                mouse.random_pause(1.0, 2.0)
            
            self.is_running = False
            print("\n✅ Bot terminé avec succès!")
                
        except Exception as e:
            self.is_running = False
            print(f"\n❌ Erreur dans execute: {e}")
    
    def stop(self):
        """Arrêter"""
        self.is_running = False
        self.should_stop = True
    
    def clear(self):
        """Vider"""
        self.actions = []

class DofusBot:
    def __init__(self):
        self.sequence = Sequence()
        self.saved_positions = []
        self.is_running_sequence = False
        self.setup_hotkeys()
    
    def setup_hotkeys(self):
        """Configurer les raccourcis clavier"""
        keyboard.add_hotkey('f2', self.stop_bot)
        keyboard.add_hotkey('f3', self.run_sequence)
        keyboard.add_hotkey('f11', self.export_positions)
        keyboard.add_hotkey('f12', self.save_position)
    
    def run_sequence(self):
        """Exécuter la séquence depuis le fichier 0407.txt"""
        if self.is_running_sequence:
            print("⚠️  Une séquence est déjà en cours d'exécution!")
            return
            
        filename = "0407.txt"
        if self.sequence.load_from_file(filename):
            def execute_sequence():
                self.is_running_sequence = True
                try:
                    self.sequence.execute(False)  # False = pas de loop
                except Exception as e:
                    print(f"Erreur lors de l'exécution: {e}")
                finally:
                    self.is_running_sequence = False
            
            # Exécuter dans un thread séparé
            thread = threading.Thread(target=execute_sequence)
            thread.daemon = True
            thread.start()
    
    def stop_bot(self):
        """Arrêter le bot"""
        print("\n🛑 Arrêt du bot demandé...")
        self.sequence.stop()
        
        # Beep de confirmation
        try:
            import winsound
            winsound.Beep(440, 200)
        except:
            print("🔔 Beep!")
    
    def save_position(self):
        """Enregistrer la position actuelle de la souris"""
        x, y = pyautogui.position()
        self.saved_positions.append({'x': x, 'y': y})
        print(f"📍 Position enregistrée: ({x}, {y}) - Total: {len(self.saved_positions)}")
    
    def export_positions(self):
        """Exporter les positions enregistrées"""
        if len(self.saved_positions) == 0:
            print("ℹ️  Aucune position enregistrée.")
            return
        
        text = ""
        for i, pos in enumerate(self.saved_positions, 1):
            text += f"shift clic ({pos['x']}, {pos['y']})  # Frêne {i}\n"
        
        print(f"\n📋 Export de {len(self.saved_positions)} positions:")
        print("-" * 40)
        print(text)
        print("-" * 40)
        
        # Copier dans le presse-papier et coller
        try:
            import pyperclip
            pyperclip.copy(text)
            print("✅ Texte copié dans le presse-papier!")
            
            # Attendre un peu puis coller
            time.sleep(0.5)
            pyautogui.hotkey('ctrl', 'v')
            print("✅ Texte collé!")
            
        except ImportError:
            print("⚠️  pyperclip non installé - texte affiché ci-dessus")
        except Exception as e:
            print(f"⚠️  Erreur lors de l'export: {e}")
    
    def show_help(self):
        """Afficher l'aide"""
        print("\n" + "=" * 60)
        print("🎮 BOT DOFUS - COMMANDES")
        print("=" * 60)
        print("F2  : Arrêter le bot")
        print("F3  : Exécuter la séquence (0407.txt)")
        print("F11 : Exporter les positions enregistrées")
        print("F12 : Enregistrer la position actuelle de la souris")
        print("Ctrl+C : Quitter le programme")
        print("=" * 60)
        print("📁 Fichier de séquence attendu: 0407.txt")
        print("📝 Format: clic (x,y) | shift clic (x,y) | pause (ms)")
        print("=" * 60)
    
    def run(self):
        """Démarrer le bot"""
        print("🚀 Bot Dofus Python démarré!")
        self.show_help()
        
        try:
            print("\n⏳ En attente de commandes...")
            print("💡 Conseil: Créez un fichier '0407.txt' avec vos actions")
            keyboard.wait('ctrl+c')
        except KeyboardInterrupt:
            print("\n👋 Bot arrêté proprement.")

if __name__ == "__main__":
    bot = DofusBot()
    bot.run()