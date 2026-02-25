"""
Audio Enhancer GUI
Drag/drop or paste a video file path, then click Start.
Calls enhance_audio_v2.py via the system Python so torch/speechbrain
don't need to be bundled into this EXE.
"""

import sys
import os
import shutil
import subprocess
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False


# ── Locate the processing script and the Python to run it with ────────────────

def find_python() -> str:
    """Return the path to a Python that has torch/speechbrain installed.
    Preference order: same dir as this script/EXE, then PATH."""
    candidates = []

    # If running as a PyInstaller bundle, look for python.exe beside the EXE
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).parent

    for name in ("python.exe", "python3.exe", "python"):
        p = base / name
        if p.exists():
            candidates.append(str(p))

    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            candidates.append(found)

    # Known install path as last fallback
    candidates.append(
        r"C:\Users\willv\AppData\Local\Programs\Python\Python312\python.exe"
    )

    for c in candidates:
        if Path(c).exists():
            return c

    return "python"   # hope it's on PATH


def find_script() -> Path:
    """Return the path to enhance_audio_v2.py next to this script/EXE."""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).parent
    return base / "enhance_audio_v2.py"


PYTHON = find_python()
SCRIPT = find_script()


# ── GUI ────────────────────────────────────────────────────────────────────────

class App(tk.Tk if not HAS_DND else TkinterDnD.Tk):  # type: ignore
    def __init__(self):
        super().__init__()
        self.title("Audio Enhancer")
        self.resizable(False, False)
        self._build_ui()
        self._proc = None

    def _build_ui(self):
        PAD = 14
        BG = "#1e1e2e"
        FG = "#cdd6f4"
        ACCENT = "#89b4fa"
        BTN_BG = "#313244"
        ENTRY_BG = "#181825"
        FONT = ("Segoe UI", 10)
        FONT_SMALL = ("Segoe UI", 9)
        FONT_MONO = ("Consolas", 9)

        self.configure(bg=BG)
        self.option_add("*Font", FONT)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCheckbutton", background=BG, foreground=FG,
                         font=FONT, focuscolor=BG)
        style.map("TCheckbutton", background=[("active", BG)])

        outer = tk.Frame(self, bg=BG, padx=PAD, pady=PAD)
        outer.pack(fill="both", expand=True)

        # ── Title ────────────────────────────────────────────────────────────
        tk.Label(outer, text="Audio Enhancer", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 10))

        # ── Drop / file row ──────────────────────────────────────────────────
        tk.Label(outer, text="Video file", bg=BG, fg=FG,
                 font=FONT_SMALL).pack(anchor="w")

        file_row = tk.Frame(outer, bg=BG)
        file_row.pack(fill="x", pady=(2, 8))

        self._path_var = tk.StringVar()
        self._entry = tk.Entry(
            file_row, textvariable=self._path_var,
            bg=ENTRY_BG, fg=FG, insertbackground=FG,
            relief="flat", font=FONT, width=48,
        )
        self._entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 6))

        tk.Button(
            file_row, text="Browse…",
            bg=BTN_BG, fg=FG, relief="flat",
            activebackground=ACCENT, activeforeground=BG,
            cursor="hand2", command=self._browse,
        ).pack(side="left", ipady=5, ipadx=6)

        # Drop-zone hint
        if HAS_DND:
            self._entry.drop_target_register(DND_FILES)
            self._entry.dnd_bind("<<Drop>>", self._on_drop)
            tk.Label(outer, text="↑  or drag & drop a file onto the field above",
                     bg=BG, fg="#585b70", font=FONT_SMALL).pack(anchor="w", pady=(0, 10))
        else:
            tk.Label(outer, text="Paste a file path into the field above",
                     bg=BG, fg="#585b70", font=FONT_SMALL).pack(anchor="w", pady=(0, 10))

        # ── Options ──────────────────────────────────────────────────────────
        self._export_audio = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            outer,
            text="Also export standalone audio file (.wav)",
            variable=self._export_audio,
        ).pack(anchor="w", pady=(0, 12))

        # ── Start / Cancel button ────────────────────────────────────────────
        self._btn = tk.Button(
            outer, text="Start",
            bg=ACCENT, fg=BG, font=("Segoe UI", 10, "bold"),
            relief="flat", cursor="hand2",
            activebackground="#b4befe", activeforeground=BG,
            command=self._on_start,
        )
        self._btn.pack(fill="x", ipady=8, pady=(0, 10))

        # ── Status log ───────────────────────────────────────────────────────
        tk.Label(outer, text="Output", bg=BG, fg=FG,
                 font=FONT_SMALL).pack(anchor="w")

        log_frame = tk.Frame(outer, bg=BG)
        log_frame.pack(fill="both", expand=True)

        self._log = tk.Text(
            log_frame,
            bg=ENTRY_BG, fg="#a6e3a1",
            font=FONT_MONO,
            relief="flat",
            height=14, width=64,
            state="disabled",
            wrap="word",
        )
        scroll = tk.Scrollbar(log_frame, command=self._log.yview,
                               bg=BG, troughcolor=BG, relief="flat")
        self._log.configure(yscrollcommand=scroll.set)
        self._log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.webm"), ("All files", "*.*")]
        )
        if path:
            self._path_var.set(path)

    def _on_drop(self, event):
        raw = event.data.strip()
        # tkinterdnd2 wraps paths with spaces in {}
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        self._path_var.set(raw)

    def _log_write(self, text: str):
        self._log.configure(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _log_clear(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    def _set_running(self, running: bool):
        if running:
            self._btn.configure(text="Cancel", bg="#f38ba8", fg="#1e1e2e",
                                 command=self._on_cancel)
        else:
            self._btn.configure(text="Start", bg="#89b4fa", fg="#1e1e2e",
                                 command=self._on_start)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _on_start(self):
        path = self._path_var.get().strip()
        if not path:
            self._log_clear()
            self._log_write("Please select or paste a video file path first.\n")
            return
        if not Path(path).exists():
            self._log_clear()
            self._log_write(f"File not found:\n  {path}\n")
            return
        if not SCRIPT.exists():
            self._log_clear()
            self._log_write(
                f"Processing script not found:\n  {SCRIPT}\n\n"
                "Make sure enhance_audio_v2.py is in the same folder as this app.\n"
            )
            return

        cmd = [PYTHON, "-u", str(SCRIPT), "--input", path]
        if self._export_audio.get():
            cmd.append("--export-audio")

        self._log_clear()
        self._log_write(f"Running: {' '.join(cmd)}\n\n")
        self._set_running(True)

        self._proc = None
        t = threading.Thread(target=self._run_process, args=(cmd,), daemon=True)
        t.start()

    def _on_cancel(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._log_write("\n[Cancelled]\n")
        self._set_running(False)

    def _run_process(self, cmd):
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            for line in self._proc.stdout:
                self.after(0, self._log_write, line)
            self._proc.wait()
            rc = self._proc.returncode
            msg = "\nDone!\n" if rc == 0 else f"\nProcess exited with code {rc}\n"
            self.after(0, self._log_write, msg)
        except Exception as e:
            self.after(0, self._log_write, f"\nError: {e}\n")
        finally:
            self.after(0, self._set_running, False)


if __name__ == "__main__":
    app = App()
    app.mainloop()
