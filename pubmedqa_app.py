"""
Application Tkinter pour PubMedQA - BioBERT Fine-tuné
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import threading
from datetime import datetime
import json
import os

class PubMedQAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PubMedQA - Assistant Médical BioBERT")
        self.root.geometry("1200x800")
        
        # Configuration du style
        self.setup_styles()
        
        # Variables
        self.model = None
        self.tokenizer = None
        self.is_model_loaded = False
        self.history = []
        
        # Création de l'interface
        self.create_widgets()
        
        # Chargement automatique du modèle
        self.root.after(100, self.load_model)
    
    def setup_styles(self):
        """Configure les styles ttk pour une interface moderne"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Couleurs
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'light': '#ecf0f1',
            'dark': '#2c3e50',
            'white': '#ffffff'
        }
        
        # Configuration des styles
        style.configure('Title.TLabel', font=('Helvetica', 18, 'bold'), foreground=self.colors['primary'])
        style.configure('Heading.TLabel', font=('Helvetica', 12, 'bold'), foreground=self.colors['dark'])
        style.configure('Result.TLabel', font=('Helvetica', 14), padding=10)
        style.configure('Success.TButton', font=('Helvetica', 10, 'bold'))
        style.configure('History.TFrame', background=self.colors['light'])
    
    def create_widgets(self):
        """Crée tous les widgets de l'interface"""
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration du grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # === En-tête ===
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="🧬 PubMedQA - Assistant Médical BioBERT", 
                                style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Statut du modèle
        self.status_label = ttk.Label(header_frame, text="⏳ Chargement du modèle...", 
                                      font=('Helvetica', 10))
        self.status_label.pack(side=tk.RIGHT)
        
        # Barre de progression
        self.progress = ttk.Progressbar(header_frame, mode='indeterminate', length=200)
        self.progress.pack(side=tk.RIGHT, padx=10)
        
        # === Panneau de gauche : Saisie ===
        left_frame = ttk.LabelFrame(main_frame, text="📝 Saisie de la question", padding="10")
        left_frame.grid(row=1, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        left_frame.rowconfigure(2, weight=1)
        
        # Question
        ttk.Label(left_frame, text="Question médicale :", style='Heading.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.question_text = scrolledtext.ScrolledText(left_frame, height=4, wrap=tk.WORD, 
                                                       font=('Helvetica', 10))
        self.question_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Contexte
        ttk.Label(left_frame, text="Contexte (abstract PubMed) :", style='Heading.TLabel').grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        
        self.context_text = scrolledtext.ScrolledText(left_frame, height=15, wrap=tk.WORD,
                                                      font=('Helvetica', 10))
        self.context_text.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Boutons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=4, column=0, pady=10)
        
        self.predict_button = ttk.Button(button_frame, text="🔍 Analyser", 
                                         command=self.predict_async, 
                                         state=tk.DISABLED)
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="🗑️ Effacer", 
                  command=self.clear_inputs).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="📁 Charger exemple", 
                  command=self.load_example).pack(side=tk.LEFT, padx=5)
        
        # === Panneau de droite : Résultats ===
        right_frame = ttk.LabelFrame(main_frame, text="📊 Résultats de l'analyse", padding="10")
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_frame.columnconfigure(0, weight=1)
        
        # Frame pour le résultat principal
        result_main_frame = ttk.Frame(right_frame)
        result_main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=10)
        
        self.result_label = ttk.Label(result_main_frame, text="En attente d'une question...", 
                                      style='Result.TLabel')
        self.result_label.pack()
        
        # Frame pour les probabilités
        prob_frame = ttk.LabelFrame(right_frame, text="Probabilités par classe", padding="10")
        prob_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)
        prob_frame.columnconfigure(0, weight=1)
        
        # Barres de progression pour les probabilités
        self.prob_bars = {}
        self.prob_labels = {}
        
        for i, label in enumerate(['Yes (Oui)', 'No (Non)', 'Maybe (Peut-être)']):
            ttk.Label(prob_frame, text=label, font=('Helvetica', 10, 'bold')).grid(row=i, column=0, sticky=tk.W, pady=2)
            
            prob_bar = ttk.Progressbar(prob_frame, length=300, mode='determinate')
            prob_bar.grid(row=i, column=1, padx=10, pady=2, sticky=(tk.W, tk.E))
            
            prob_label = ttk.Label(prob_frame, text="0%", width=8)
            prob_label.grid(row=i, column=2, padx=5)
            
            self.prob_bars[label] = prob_bar
            self.prob_labels[label] = prob_label
        
        prob_frame.columnconfigure(1, weight=1)
        
        # Confiance globale
        confidence_frame = ttk.Frame(right_frame)
        confidence_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(confidence_frame, text="Confiance :", font=('Helvetica', 10)).pack(side=tk.LEFT)
        self.confidence_label = ttk.Label(confidence_frame, text="--", font=('Helvetica', 12, 'bold'))
        self.confidence_label.pack(side=tk.LEFT, padx=10)
        
        # === Historique ===
        history_frame = ttk.LabelFrame(main_frame, text="📜 Historique des requêtes", padding="10")
        history_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0), pady=(10, 0))
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)
        
        # Treeview pour l'historique
        columns = ('timestamp', 'question', 'answer', 'confidence')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='tree headings', height=8)
        
        self.history_tree.heading('#0', text='')
        self.history_tree.heading('timestamp', text='Heure')
        self.history_tree.heading('question', text='Question')
        self.history_tree.heading('answer', text='Réponse')
        self.history_tree.heading('confidence', text='Confiance')
        
        self.history_tree.column('#0', width=0, stretch=False)
        self.history_tree.column('timestamp', width=100)
        self.history_tree.column('question', width=300)
        self.history_tree.column('answer', width=100)
        self.history_tree.column('confidence', width=100)
        
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        self.history_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Boutons d'historique
        history_button_frame = ttk.Frame(history_frame)
        history_button_frame.grid(row=1, column=0, pady=5)
        
        ttk.Button(history_button_frame, text="📤 Exporter l'historique", 
                  command=self.export_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(history_button_frame, text="🗑️ Effacer l'historique", 
                  command=self.clear_history).pack(side=tk.LEFT, padx=5)
        
        # === Barre de statut ===
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_text = ttk.Label(status_frame, text="Prêt", font=('Helvetica', 9))
        self.status_text.pack(side=tk.LEFT)
    
    def load_model(self):
        """Charge le modèle BioBERT fine-tuné"""
        def load_task():
            try:
                self.progress.start()
                self.status_label.config(text="⏳ Chargement du modèle...")
                self.status_text.config(text="Chargement du modèle BioBERT...")
                
                # Utiliser directement le modèle de base pour éviter les problèmes
                model_path = "C:/Users/naima/pubmedqa_app/pubmedqa_model"                
                self.status_text.config(text=f"Chargement depuis : {model_path}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=3,
                    ignore_mismatched_sizes=True
                )
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(device)
                self.model.eval()
                
                self.is_model_loaded = True
                
                self.root.after(0, self.on_model_loaded)
                
            except Exception as ex:
                error_msg = str(ex)
                self.root.after(0, lambda msg=error_msg: self.on_model_error(msg))
        
        thread = threading.Thread(target=load_task)
        thread.daemon = True
        thread.start()
    
    def on_model_loaded(self):
        """Callback quand le modèle est chargé"""
        self.progress.stop()
        self.status_label.config(text="✅ Modèle chargé")
        self.status_text.config(text="Modèle BioBERT chargé avec succès")
        self.predict_button.config(state=tk.NORMAL)
        
        messagebox.showinfo("Modèle chargé", 
                           "Le modèle BioBERT a été chargé avec succès !\n\n"
                           "Vous pouvez maintenant poser des questions médicales.")
    
    def on_model_error(self, error_msg):
        """Callback en cas d'erreur de chargement"""
        self.progress.stop()
        self.status_label.config(text="❌ Erreur")
        self.status_text.config(text=f"Erreur : {error_msg[:50]}...")
        
        messagebox.showerror("Erreur de chargement", 
                            f"Impossible de charger le modèle.\n\nErreur : {error_msg}")
    
    def predict_async(self):
        """Lance la prédiction de manière asynchrone"""
        if not self.is_model_loaded:
            messagebox.showwarning("Modèle non chargé", "Veuillez attendre le chargement du modèle.")
            return
        
        question = self.question_text.get("1.0", tk.END).strip()
        context = self.context_text.get("1.0", tk.END).strip()
        
        if not question:
            messagebox.showwarning("Question vide", "Veuillez entrer une question.")
            return
        
        if not context:
            messagebox.showwarning("Contexte vide", "Veuillez entrer un contexte (abstract PubMed).")
            return
        
        self.predict_button.config(state=tk.DISABLED)
        self.status_text.config(text="Analyse en cours...")
        self.progress.start()
        
        thread = threading.Thread(target=lambda: self.predict(question, context))
        thread.daemon = True
        thread.start()
    
    def predict(self, question, context):
        """Effectue la prédiction"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            encoding = self.tokenizer(
                question,
                context,
                max_length=512,
                padding='max_length',
                truncation='only_second',
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(logits, dim=-1)
            
            probs = probabilities[0].cpu().numpy()
            pred_label = ['yes', 'no', 'maybe'][prediction[0].item()]
            
            result = {
                'prediction': pred_label,
                'confidence': float(probs.max()),
                'probabilities': {
                    'Yes (Oui)': float(probs[0]),
                    'No (Non)': float(probs[1]),
                    'Maybe (Peut-être)': float(probs[2])
                }
            }
            
            self.root.after(0, lambda: self.display_results(result, question))
            
        except Exception as ex:
            error_msg = str(ex)
            self.root.after(0, lambda msg=error_msg: self.on_prediction_error(msg))
    
    def display_results(self, result, question):
        """Affiche les résultats de la prédiction"""
        self.progress.stop()
        self.predict_button.config(state=tk.NORMAL)
        self.status_text.config(text="Analyse terminée")
        
        pred_text = {
            'yes': '✅ OUI - La réponse est positive',
            'no': '❌ NON - La réponse est négative',
            'maybe': '⚠️ PEUT-ÊTRE - Résultat incertain'
        }
        
        self.result_label.config(text=pred_text.get(result['prediction'], result['prediction']))
        
        for label, prob in result['probabilities'].items():
            bar = self.prob_bars[label]
            label_widget = self.prob_labels[label]
            
            bar['value'] = prob * 100
            label_widget.config(text=f"{prob*100:.1f}%")
        
        self.confidence_label.config(text=f"{result['confidence']*100:.1f}%")
        self.add_to_history(question, result)
    
    def on_prediction_error(self, error_msg):
        """Gère les erreurs de prédiction"""
        self.progress.stop()
        self.predict_button.config(state=tk.NORMAL)
        self.status_text.config(text=f"Erreur : {error_msg}")
        messagebox.showerror("Erreur de prédiction", f"Une erreur est survenue :\n{error_msg}")
    
    def add_to_history(self, question, result):
        """Ajoute une entrée à l'historique"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        short_question = question[:50] + "..." if len(question) > 50 else question
        
        self.history_tree.insert('', 0, values=(
            timestamp,
            short_question,
            result['prediction'].upper(),
            f"{result['confidence']*100:.1f}%"
        ))
        
        self.history.append({
            'timestamp': timestamp,
            'question': question,
            'answer': result['prediction'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        })
    
    def clear_inputs(self):
        """Efface les champs de saisie"""
        self.question_text.delete("1.0", tk.END)
        self.context_text.delete("1.0", tk.END)
        self.status_text.config(text="Champs effacés")
    
    def load_example(self):
        """Charge un exemple pré-défini"""
        example = {
            "question": "Does aspirin reduce the risk of cardiovascular events in patients with diabetes?",
            "context": "A randomized controlled trial of 15,480 patients with diabetes showed that aspirin 100mg daily reduced the risk of serious vascular events by 12% (rate ratio 0.88, 95% CI 0.79-0.97). However, the absolute reduction was small and was counterbalanced by an increased risk of major bleeding."
        }
        
        self.question_text.delete("1.0", tk.END)
        self.question_text.insert("1.0", example['question'])
        self.context_text.delete("1.0", tk.END)
        self.context_text.insert("1.0", example['context'])
        self.status_text.config(text="Exemple chargé")
    
    def export_history(self):
        """Exporte l'historique en JSON"""
        if not self.history:
            messagebox.showinfo("Historique vide", "Aucune entrée dans l'historique.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"pubmedqa_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            
            self.status_text.config(text=f"Historique exporté : {filename}")
            messagebox.showinfo("Export réussi", f"Historique sauvegardé dans :\n{filename}")
    
    def clear_history(self):
        """Efface l'historique"""
        if messagebox.askyesno("Confirmation", "Voulez-vous vraiment effacer tout l'historique ?"):
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            self.history = []
            self.status_text.config(text="Historique effacé")

def main():
    root = tk.Tk()
    app = PubMedQAApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()