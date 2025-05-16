import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional
import minimax
import alphabeta
from evaluation import evaluate as good_evaluate
from evaluationBad import simple_evaluate as bad_evaluate

Board = List[List[Optional[str]]]
CELL_SIZE = 80
ROWS, COLS = 6, 7
ANIMATION_DELAY = 120

class Connect4GUI:
    def __init__(self, master):
        self.master = master
        master.title("Puissance 4")
        master.geometry(f"{COLS*CELL_SIZE}x{ROWS*CELL_SIZE}")
        master.resizable(False, False)
        self._create_styles()
        self._create_widgets()

    def _create_styles(self):
        style = ttk.Style(self.master)
        style.theme_use('clam')
        style.configure('TButton', font=('Helvetica', 12), padding=6)
        style.configure('TLabel', font=('Helvetica', 11))
        style.configure('TRadiobutton', font=('Helvetica', 11))
        style.configure('Spinbox.TSpinbox', font=('Helvetica', 11), padding=4)

    def _create_widgets(self):
        self.container = ttk.Frame(self.master)
        self.container.pack(fill='both', expand=True)
        # Menu
        self.menu_frame = ttk.Frame(self.container, padding=20, relief='ridge')
        self.menu_frame.grid(row=0, column=0, sticky='ns')
        ttk.Label(self.menu_frame, text="Paramètres du jeu", font=('Helvetica', 16, 'bold')).grid(row=0, column=0, pady=(0,10))
        # Heuristique
        ttk.Label(self.menu_frame, text="Évaluation:").grid(row=1, column=0, sticky='w')
        self.eval_var = tk.StringVar(value='good')
        ttk.Radiobutton(self.menu_frame, text='Classique (bonne)', variable=self.eval_var, value='good').grid(row=2, column=0, sticky='w')
        ttk.Radiobutton(self.menu_frame, text='Simple (mauvaise)', variable=self.eval_var, value='bad').grid(row=3, column=0, sticky='w')
        # Algo
        ttk.Label(self.menu_frame, text="Algo IA:").grid(row=4, column=0, pady=(10,0), sticky='w')
        self.algo_var = tk.StringVar(value='minimax')
        ttk.Radiobutton(self.menu_frame, text='Minimax', variable=self.algo_var, value='minimax').grid(row=5, column=0, sticky='w')
        ttk.Radiobutton(self.menu_frame, text='Alpha-Beta', variable=self.algo_var, value='alphabeta').grid(row=6, column=0, sticky='w')
        # Modes
        ttk.Label(self.menu_frame, text="Joueur X:").grid(row=7, column=0, pady=(10,0), sticky='w')
        self.mode_x = tk.StringVar(value='A')
        ttk.Radiobutton(self.menu_frame, text='Humain', variable=self.mode_x, value='H').grid(row=8, column=0, sticky='w')
        ttk.Radiobutton(self.menu_frame, text='IA', variable=self.mode_x, value='A').grid(row=9, column=0, sticky='w')
        ttk.Label(self.menu_frame, text="Joueur O:").grid(row=10, column=0, pady=(10,0), sticky='w')
        self.mode_o = tk.StringVar(value='A')
        ttk.Radiobutton(self.menu_frame, text='Humain', variable=self.mode_o, value='H').grid(row=11, column=0, sticky='w')
        ttk.Radiobutton(self.menu_frame, text='IA', variable=self.mode_o, value='A').grid(row=12, column=0, sticky='w')
        # Depth
        ttk.Label(self.menu_frame, text="Profondeur IA:").grid(row=13, column=0, pady=(10,0), sticky='w')
        self.depth_spin = ttk.Spinbox(self.menu_frame, from_=1, to=8, width=5, style='Spinbox.TSpinbox')
        self.depth_spin.set(4)
        self.depth_spin.grid(row=14, column=0, sticky='w')
        # Play
        ttk.Button(self.menu_frame, text="Jouer", command=self.start_game).grid(row=15, column=0, pady=20)
        # Game
        self.game_frame = ttk.Frame(self.container)
        self.canvas = tk.Canvas(self.game_frame, width=COLS*CELL_SIZE, height=ROWS*CELL_SIZE, bg='#0044cc', highlightthickness=0)
        self.canvas.pack(side='left')
        self.canvas.bind("<Button-1>", self.on_click)

    def start_game(self):
        # Settings
        minimax.evaluate = good_evaluate if self.eval_var.get()=='good' else bad_evaluate
        self.use_ab = (self.algo_var.get()=='alphabeta')
        self.modes = {'X':self.mode_x.get(),'O':self.mode_o.get()}
        try:
            self.depth = int(self.depth_spin.get())
        except ValueError:
            self.depth = 4
        # Init
        self.board:Board=[[None]*COLS for _ in range(ROWS)]
        self.current='X'
        # Switch view
        self.menu_frame.grid_forget()
        self.game_frame.grid(row=0,column=1)
        self.draw_board()
        if self.modes[self.current]=='A':
            self.master.after(200,self.ai_move)

    def draw_board(self):
        self.canvas.delete('all')
        for r in range(ROWS):
            for c in range(COLS):
                x0,y0=c*CELL_SIZE+5,r*CELL_SIZE+5
                x1,y1=x0+CELL_SIZE-10,y0+CELL_SIZE-10
                fill='white'
                if self.board[r][c]=='X': fill='red'
                elif self.board[r][c]=='O': fill='yellow'
                self.canvas.create_oval(x0,y0,x1,y1,fill=fill,outline='black')

    def on_click(self,event):
        if self.modes[self.current]=='H':
            col=event.x//CELL_SIZE
            self.animate_drop(col)

    def animate_drop(self,col):
        if col<0 or col>=COLS or self.board[0][col] is not None: return
        # find target row
        for target in range(ROWS-1,-1,-1):
            if self.board[target][col] is None: break
        # animate
        def step(r):
            self.draw_board()
            x0,y0=col*CELL_SIZE+5,r*CELL_SIZE+5
            x1,y1=x0+CELL_SIZE-10,y0+CELL_SIZE-10
            color='red' if self.current=='X' else 'yellow'
            self.canvas.create_oval(x0,y0,x1,y1,fill=color,outline='black')
            if r<target:
                self.master.after(ANIMATION_DELAY,lambda:step(r+1))
            else:
                self.board[target][col]=self.current
                if self.check_end(): return
                self.switch_player()
                if self.modes[self.current]=='A':
                    self.master.after(200,self.ai_move)
        step(0)

    def ai_move(self):
        if self.use_ab:
            _,move=alphabeta.alphabeta(self.board,self.depth,float('-inf'),float('inf'),maximizing_player=(self.current=='X'))
        else:
            _,move=minimax.minimax(self.board,self.depth,maximizing_player=(self.current=='X'))
        self.animate_drop(move)

    def check_end(self):
        if self._check_win(self.current):
            col='Rouge' if self.current=='X' else 'Jaune'
            messagebox.showinfo("Fin de partie",f"Le joueur {self.current} a gagné ({col}) !")
            self.master.quit()
            return True
        if all(self.board[0][c] for c in range(COLS)):
            messagebox.showinfo("Fin de partie","Match nul !")
            self.master.quit()
            return True
        return False

    def switch_player(self):
        self.current='O' if self.current=='X' else 'X'

    def _check_win(self,player):
        for r in range(ROWS):
            for c in range(COLS-3):
                if all(self.board[r][c+i]==player for i in range(4)):return True
        for c in range(COLS):
            for r in range(ROWS-3):
                if all(self.board[r+i][c]==player for i in range(4)):return True
        for r in range(ROWS-3):
            for c in range(COLS-3):
                if all(self.board[r+i][c+i]==player for i in range(4)):return True
        for r in range(3,ROWS):
            for c in range(COLS-3):
                if all(self.board[r-i][c+i]==player for i in range(4)):return True
        return False

if __name__=='__main__':
    root=tk.Tk()
    app=Connect4GUI(root)
    root.mainloop()