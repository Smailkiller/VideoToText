import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import threading
import os
import sys
import datetime
import webbrowser
import whisper
import wave
import json
from vosk import Model, KaldiRecognizer
import shutil
from pydub.utils import which
from pydub import AudioSegment


# Автопуть для ffmpeg/ffprobe если приложение собрано
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
    ffmpeg_path = os.path.join(base_path, "ffmpeg", "ffmpeg.exe")
    ffprobe_path = os.path.join(base_path, "ffmpeg", "ffprobe.exe")
else:
    # Для отладки из IDE
    ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg", "ffmpeg.exe")
    ffprobe_path = os.path.join(os.getcwd(), "ffmpeg", "ffprobe.exe")

AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS

    # Обновляем PATH
    os.environ['PATH'] = os.pathsep.join([
        os.path.join(base_path, 'vosk'),
        os.environ['PATH']
    ])

    # Для корректной работы Vosk DLL
    try:
        os.add_dll_directory(os.path.join(base_path, 'vosk'))
    except (AttributeError, FileNotFoundError):
        pass


# Ensure mel_filters.npz is available
if getattr(sys, 'frozen', False):
    whisper_asset_path = os.path.join(sys._MEIPASS, "whisper", "assets", "mel_filters.npz")
else:
    whisper_asset_path = os.path.join(os.getcwd(), "whisper", "assets", "mel_filters.npz")

os.environ["WHISPER_ASSETS"] = os.path.dirname(whisper_asset_path)


MODELS_INFO = {
    'tiny':   {'size': '75MB',  'speed': 5, 'quality': 1},
    'base':   {'size': '142MB', 'speed': 5, 'quality': 1},
    'small':  {'size': '456MB', 'speed': 4, 'quality': 2},
    'medium': {'size': '1.5GB', 'speed': 3, 'quality': 4},
    'large':  {'size': '2.9GB', 'speed': 1, 'quality': 5},
}
WHISPER_LANGUAGES = {
    "🌐 Auto-detect": None,
    "🇬🇧 English": "en",
    "🇷🇺 Russian": "ru",
    "🇫🇷 French": "fr",
    "🇩🇪 German": "de",
    "🇪🇸 Spanish": "es",
    "🇮🇹 Italian": "it",
    "🇨🇳 Chinese": "zh",
    "🇯🇵 Japanese": "ja",
    "🇰🇷 Korean": "ko",
    "🇺🇦 Ukrainian": "uk",
    "🇵🇱 Polish": "pl",
    "🇧🇷 Portuguese": "pt",
    "🇹🇷 Turkish": "tr"
    # Добавляй по мере необходимости
}

def model_label(name):
    info = MODELS_INFO[name]
    return f"{name.capitalize()} — {info['size']} | Speed {info['speed']}/5 | Quality {info['quality']}/5"

whisper_stop_flag = False
vosk_stop_flag = False

def extract_audio(video_path, wav_path):
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])

def transcribe_audio_whisper(wav_path, model, language=None):
    try:
        transcribe_args = {'verbose': False, "word_timestamps": True}
        if language:
            transcribe_args["language"] = language
        result = model.transcribe(wav_path, **transcribe_args)
        output_lines = []
        for segment in result.get("segments", []):
            start_time = str(datetime.timedelta(seconds=int(segment["start"])))
            line = f"[{start_time}] {segment['text'].strip()}"
            output_lines.append(line)
        return "\n".join(output_lines)
    except Exception as e:
        print(f"⚠️ Ошибка в Whisper: {e}")
        return None



def process_videos_whisper(base_folder, model_key, log, skip_existing, lang_code):
    global whisper_stop_flag
    whisper_stop_flag = False

    model_name = model_key.split(" — ")[0].lower()
    log(f"🚀 Загрузка модели Whisper: {model_name}...")
    model = whisper.load_model(model_name)
    log("✅ Модель загружена, поиск видеофайлов...")

    supported_ext = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpeg", ".mpg")
    video_files = []
    for root, _, files in os.walk(base_folder):
        for f in files:
            if f.lower().endswith(supported_ext):
                video_files.append(os.path.join(root, f))

    log(f"🔎 Найдено файлов: {len(video_files)}")
    if not video_files:
        log("❌ Видео-файлы не найдены.")
        return

    for idx, video_path in enumerate(video_files, 1):
        if whisper_stop_flag:
            log("⛔ Процесс остановлен.")
            return

        log(f"\n🔄 [{idx}/{len(video_files)}] Обработка файла:")
        log(f"    📄 {video_path}")

        base_name = os.path.splitext(video_path)[0]
        wav_path = base_name + ".wav"
        txt_path = base_name + ".txt"

        if skip_existing and os.path.exists(txt_path):
            log(f"    ⏭️ Пропущен (уже есть .txt): {txt_path}")
            continue

        try:
            log("    🎧 Вытаскивание аудио...")
            extract_audio(video_path, wav_path)
            log("    🎧 Аудио успешно извлечено.")
        except Exception as e:
            log(f"    ⚠️ Ошибка при извлечении аудио: {e}")
            continue

        try:
            log("    🖋️ Транскрибация аудио...")
            text = transcribe_audio_whisper(wav_path, model, lang_code)
            if text is None:
                log(f"    ⚠️ Не удалось получить текст для {wav_path}")
                continue
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"# {os.path.basename(video_path)}\n\n")
                f.write(text)
            log(f"    ✅ Текст сохранён в: {txt_path}")
        except Exception as e:
            log(f"    ⚠️ Ошибка транскрипции: {e}")
            continue

    log("\n✅ Все видео обработаны.")


def start_process_whisper(video_path, model_key, log, stop_button, skip_existing, language_ui):
    stop_button.config(state='normal')
    lang_code = WHISPER_LANGUAGES.get(language_ui)
    threading.Thread(target=process_videos_whisper, args=(video_path, model_key, log, skip_existing, lang_code), daemon=True).start()



def stop_process_whisper():
    global whisper_stop_flag
    whisper_stop_flag = True

def transcribe_audio_vosk(wav_path, model, log):
    global vosk_stop_flag
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    total = wf.getnframes()
    read = 0
    last_percent = -1
    while True:
        if vosk_stop_flag:
            log("\n❌ Процесс остановлен пользователем.")
            return None
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)
        read += 4000
        percent = min(100, int(read / total * 100))
        if percent != last_percent:
            log(f"    🟩 Прогресс транскрипции: {percent}%", overwrite=True)
            last_percent = percent
    log("\n    ✅ Распознавание завершено.")
    return json.loads(rec.FinalResult())

def format_transcription(result, pause_threshold=0.8):
    entries = result.get("result", [])
    output = []
    current_sentence = []
    sentence_start_time = None
    last_word_end = 0
    for word in entries:
        if sentence_start_time is None:
            sentence_start_time = word['start']
        if word['start'] - last_word_end > pause_threshold and current_sentence:
            output.append(f"[{sentence_start_time:.1f}] {' '.join(current_sentence)}")
            current_sentence = []
            sentence_start_time = word['start']
        current_sentence.append(word['word'])
        last_word_end = word['end']
    if current_sentence:
        output.append(f"[{sentence_start_time:.1f}] {' '.join(current_sentence)}")
    return '\n'.join(output)

def process_videos_vosk(base_folder, model_path, log):
    global vosk_stop_flag
    vosk_stop_flag = False
    log(f"🚀 Загрузка модели из: {model_path}")
    model = Model(model_path)
    supported_ext = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpeg", ".mpg")
    video_files = []
    for root, _, files in os.walk(base_folder):
        for f in files:
            if f.lower().endswith(supported_ext):
                video_files.append(os.path.join(root, f))
    log(f"🔎 Найдено видеофайлов: {len(video_files)}\n")
    error_log_path = os.path.join(base_folder, "error_log.txt")
    if os.path.exists(error_log_path):
        os.remove(error_log_path)
    for idx, video_path in enumerate(video_files, 1):
        if vosk_stop_flag:
            log("\n❌ Обработка остановлена пользователем.")
            break
        log(f"\n🔄 [{idx}/{len(video_files)}] {video_path}")
        base_name = os.path.splitext(video_path)[0]
        wav_path = base_name + ".wav"
        txt_path = base_name + ".txt"
        try:
            log("    🎧 Извлечение аудио...")
            extract_audio(video_path, wav_path)
        except Exception as e:
            log(f"    ⚠️ Ошибка аудио: {e}")
            with open(error_log_path, "a", encoding="utf-8") as log_f:
                log_f.write(f"[Audio Error] {video_path}: {e}\n")
            continue
        try:
            log("    📝 Транскрибация...")
            result = transcribe_audio_vosk(wav_path, model, log)
            if result is None:
                break
            formatted_text = format_transcription(result)
            with wave.open(wav_path, "rb") as wf:
                duration = round(wf.getnframes() / wf.getframerate(), 1)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"# {os.path.basename(base_name)} | Длительность: {duration} сек\n")
                f.write(formatted_text)
            log(f"    💾 Сохранено: {txt_path}")
        except Exception as e:
            log(f"    ⚠️ Ошибка транскрипции: {e}")
            with open(error_log_path, "a", encoding="utf-8") as log_f:
                log_f.write(f"[Transcription Error] {video_path}: {e}\n")
            continue
    log("\n✅ Обработка завершена.")

def start_process_vosk(video_path, model_path, log, stop_button):
    stop_button.config(state='normal')
    threading.Thread(target=process_videos_vosk, args=(video_path, model_path, log), daemon=True).start()

def stop_process_vosk():
    global vosk_stop_flag
    vosk_stop_flag = True

class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self._stdout = sys.stdout
        self.last_line_tag = "progress_line"

    def write(self, text):
        self._stdout.write(text)

        self.text_widget.configure(state='normal')

        if "\r" in text:
            # Удалить старую строку прогресса
            self.text_widget.mark_set("progress_start", "end-1l linestart")
            self.text_widget.mark_set("progress_end", "end-1l lineend")
            self.text_widget.delete("progress_start", "progress_end")
            # Вставить новую строку
            self.text_widget.insert("end", text.replace("\r", ""))
        else:
            self.text_widget.insert("end", text)

        self.text_widget.see("end")
        self.text_widget.configure(state='disabled')

    def flush(self):
        self._stdout.flush()


def open_vosk_link(event=None):
    webbrowser.open_new("https://alphacephei.com/vosk/models")

def gui_app():
    root = tk.Tk()
    root.title("🎙️ Video Transcriber GUI (Whisper + Vosk)")
    root.geometry("1260x780")
    root.configure(bg="#ecf0f3")

    # --- Основная рамка ---
    main_frame = tk.Frame(root, bg="#ecf0f3")
    main_frame.pack(fill='both', expand=True, padx=8, pady=8)

    # --- Notebook с вкладками ---
    notebook = ttk.Notebook(main_frame)
    notebook.pack(side='left', fill='both', expand=True)

    # --- Справка сбоку ---
    sidebar = tk.Frame(main_frame, bg="#f6f8fa", width=310)
    sidebar.pack(side='right', fill='y', padx=(12, 0))

    tk.Label(sidebar, text="Whisper (OpenAI)", bg="#f6f8fa", fg="#00539b", font=("Segoe UI", 12, "bold")).pack(anchor='nw', padx=12, pady=(14, 2))
    tk.Label(sidebar, text="• Поддерживает почти все языки\n• Модели подгружаются автоматически\n• Качество: отличное (особенно large)\n• Требует больше ресурсов\n",
             justify='left', bg="#f6f8fa", font=("Segoe UI", 9)).pack(anchor='nw', padx=12)

    tk.Label(sidebar, text="Модели Whisper:", bg="#f6f8fa", fg="#444", font=("Segoe UI", 10, "bold")).pack(anchor='nw', padx=12)
    model_table = tk.Text(sidebar, height=6, width=40, bg="#f6f8fa", fg="#333", relief='flat', font=("Consolas", 9))
    model_table.pack(anchor='nw', padx=12)
    model_table.insert(tk.END, "  Name       Size      Speed  Quality\n")
    for k, v in MODELS_INFO.items():
        model_table.insert(tk.END, f"  {k:<9}{v['size']:<9} {v['speed']}/5    {v['quality']}/5\n")
    model_table.configure(state='disabled')

    tk.Label(sidebar, text="Vosk (Kaldi)", bg="#f6f8fa", fg="#5f4d00", font=("Segoe UI", 12, "bold")).pack(anchor='nw', padx=12, pady=(20, 2))
    tk.Label(sidebar, text="• Мгновенная работа без интернета\n• Легко запускать на слабых ПК\n• Требует скачивания модели вручную\n• Качество — сильно зависит от модели\n",
             justify='left', bg="#f6f8fa", font=("Segoe UI", 9)).pack(anchor='nw', padx=12)
    vosk_link = tk.Label(sidebar, text="→ Скачать модели Vosk", fg="#3275c6", bg="#f6f8fa", cursor="hand2", font=("Segoe UI", 10, "underline"))
    vosk_link.pack(anchor='nw', padx=12, pady=(0, 18))
    vosk_link.bind("<Button-1>", open_vosk_link)

    # --- Whisper tab ---
    whisper_tab = tk.Frame(notebook, bg="#ecf0f3")
    notebook.add(whisper_tab, text="Whisper")

    bg_main = "#ecf0f3"
    entry_style = {"bg": "#ffffff", "relief": "flat", "highlightthickness": 1, "highlightbackground": "#d0d5db", "bd": 0, "font": ("Segoe UI", 10)}
    btn_style = {"bg": "#2c3e50", "fg": "#ffffff", "activebackground": "#34495e", "activeforeground": "#ecf0f1",
                 "relief": "flat", "bd": 0, "font": ("Segoe UI", 10, "bold"), "cursor": "hand2", "padx": 10, "pady": 6}

    tk.Label(whisper_tab, text="Папка с видео:", bg=bg_main, font=("Segoe UI", 10, "bold"), anchor='w').pack(anchor='w', padx=15, pady=(15, 0))
    frame_video_w = tk.Frame(whisper_tab, bg=bg_main)
    frame_video_w.pack(fill='x', padx=15)
    entry_video_w = tk.Entry(frame_video_w, **entry_style)
    entry_video_w.pack(side='left', fill='x', expand=True, ipady=6, pady=5)
    tk.Button(frame_video_w, text="Обзор", command=lambda: entry_video_w.delete(0, tk.END) or entry_video_w.insert(0, filedialog.askdirectory()), **btn_style).pack(side='left', padx=5)

    tk.Label(whisper_tab, text="Выберите модель Whisper:", bg=bg_main, font=("Segoe UI", 10, "bold")).pack(anchor='w', padx=15, pady=(10, 0))
    model_var = tk.StringVar(value=model_label("small"))
    model_labels = [model_label(m) for m in MODELS_INFO]
    dropdown = tk.OptionMenu(whisper_tab, model_var, *model_labels)
    dropdown.config(font=("Segoe UI", 10), bg="white", width=36)
    dropdown.pack(fill='x', padx=15, pady=(0, 10))

    skip_existing_var = tk.BooleanVar(value=True)
    skip_checkbox = tk.Checkbutton(whisper_tab, text="Пропускать уже обработанные", variable=skip_existing_var, bg=bg_main, font=("Segoe UI", 9))
    skip_checkbox.pack(anchor='w', padx=20, pady=(0, 10))
    tk.Label(whisper_tab, text="Язык (не обязательно):", bg=bg_main, font=("Segoe UI", 10, "bold")).pack(anchor='w',
                                                                                                         padx=15,
                                                                                                         pady=(0, 0))
    language_var = tk.StringVar()
    language_dropdown = ttk.Combobox(whisper_tab, textvariable=language_var, font=("Segoe UI", 10))
    language_dropdown['values'] = list(WHISPER_LANGUAGES.keys())

    language_dropdown.pack(fill='x', padx=15, pady=(0, 10))
    language_dropdown.set("🌐 Auto-detect")  # по умолчанию пусто

    frame_buttons_w = tk.Frame(whisper_tab, bg=bg_main)
    frame_buttons_w.pack(pady=15)
    tk.Button(frame_buttons_w, text="🗑️ Удалить .wav", command=lambda: delete_wavs(entry_video_w.get(), log_output_w),
              **btn_style).pack(side='left', padx=5)
    tk.Button(frame_buttons_w, text="📄 Собрать TXT", command=lambda: merge_txt(entry_video_w.get(), log_output_w),
              **btn_style).pack(side='left', padx=5)

    btn_start_w = tk.Button(frame_buttons_w, text="▶️ Обработка", command=lambda: start_process_whisper(entry_video_w.get(), model_var.get(), log_output_w, btn_stop_w, skip_existing_var.get(), language_var.get()), **btn_style)
    btn_start_w.pack(side='left', padx=5)
    btn_stop_w = tk.Button(frame_buttons_w, text="⛔ Остановить", command=stop_process_whisper, state='normal', **btn_style)
    btn_stop_w.pack(side='left', padx=5)
    tk.Button(frame_buttons_w, text="🧹 Очистить", command=lambda: clear_output(text_output_w), **btn_style).pack(side='left', padx=5)

    text_output_w = scrolledtext.ScrolledText(whisper_tab, height=28, bg="#ffffff", relief="flat", font=("Consolas", 10), wrap="word", state='disabled')
    text_output_w.pack(fill='both', expand=True, padx=15, pady=(0, 10))

    # --- Vosk tab ---
    vosk_tab = tk.Frame(notebook, bg="#ecf0f3")
    notebook.add(vosk_tab, text="Vosk")

    tk.Label(vosk_tab, text="Папка с видео:", bg=bg_main, font=("Segoe UI", 10, "bold")).pack(anchor='w', padx=15, pady=(15, 0))
    frame_video_v = tk.Frame(vosk_tab, bg=bg_main)
    frame_video_v.pack(fill='x', padx=15)
    entry_video_v = tk.Entry(frame_video_v, **entry_style)
    entry_video_v.pack(side='left', fill='x', expand=True, ipady=6, pady=5)
    tk.Button(frame_video_v, text="Обзор", command=lambda: entry_video_v.delete(0, tk.END) or entry_video_v.insert(0, filedialog.askdirectory()), **btn_style).pack(side='left', padx=5)

    tk.Label(vosk_tab, text="Папка с моделью Vosk:", bg=bg_main, font=("Segoe UI", 10, "bold")).pack(anchor='w', padx=15, pady=(10, 0))
    frame_model_v = tk.Frame(vosk_tab, bg=bg_main)
    frame_model_v.pack(fill='x', padx=15)
    entry_model_v = tk.Entry(frame_model_v, **entry_style)
    entry_model_v.pack(side='left', fill='x', expand=True, ipady=6, pady=5)
    tk.Button(frame_model_v, text="Обзор", command=lambda: entry_model_v.delete(0, tk.END) or entry_model_v.insert(0, filedialog.askdirectory()), **btn_style).pack(side='left', padx=5)

    skip_existing_var_v = tk.BooleanVar(value=True)
    skip_checkbox_v = tk.Checkbutton(vosk_tab, text="Пропускать уже обработанные", variable=skip_existing_var_v,
                                     bg=bg_main, font=("Segoe UI", 9))
    skip_checkbox_v.pack(anchor='w', padx=20, pady=(0, 10))

    frame_buttons_v = tk.Frame(vosk_tab, bg=bg_main)
    frame_buttons_v.pack(pady=15)


    tk.Button(frame_buttons_v, text="🗑️ Удалить .wav", command=lambda: delete_wavs(entry_video_v.get(), log_output_v),
              **btn_style).pack(side='left', padx=5)
    tk.Button(frame_buttons_v, text="📄 Собрать TXT", command=lambda: merge_txt(entry_video_v.get(), log_output_v),
              **btn_style).pack(side='left', padx=5)

    btn_start_v = tk.Button(frame_buttons_v, text="▶️ Обработка", command=lambda: start_process_vosk(entry_video_v.get(), entry_model_v.get(), log_output_v, btn_stop_v, skip_existing_var_v.get())
, **btn_style)
    btn_start_v.pack(side='left', padx=5)
    btn_stop_v = tk.Button(frame_buttons_v, text="⛔ Остановить", command=stop_process_vosk, state='normal', **btn_style)
    btn_stop_v.pack(side='left', padx=5)
    tk.Button(frame_buttons_v, text="🧹 Очистить", command=lambda: clear_output(text_output_v), **btn_style).pack(side='left', padx=5)

    text_output_v = scrolledtext.ScrolledText(vosk_tab, height=28, bg="#ffffff", relief="flat", font=("Consolas", 10), wrap="word", state='disabled')
    text_output_v.pack(fill='both', expand=True, padx=15, pady=(0, 10))

    sys.stdout = StdoutRedirector(text_output_w)
    sys.stderr = StdoutRedirector(text_output_w)

    def merge_txt(base_folder, log):
        summary_path = os.path.join(base_folder, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            for root, _, files in os.walk(base_folder):
                for f in sorted(files):
                    if f.endswith(".txt"):
                        base = os.path.splitext(f)[0]
                        full_video = [base + ext for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]]
                        if any(os.path.exists(os.path.join(root, v)) for v in full_video):
                            txt_path = os.path.join(root, f)
                            with open(txt_path, "r", encoding="utf-8") as tf:
                                summary_file.write(f"\n\n--- {f} ---\n")
                                summary_file.write(tf.read())
        log(f"\n📄 Файл summary.txt сохранён в: {summary_path}")

    def delete_wavs(base_folder, log):
        deleted = 0
        for root, _, files in os.walk(base_folder):
            for f in files:
                if f.lower().endswith(".wav"):
                    wav_path = os.path.join(root, f)
                    video_path = os.path.splitext(wav_path)[0]
                    if any(os.path.exists(video_path + ext) for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]):
                        try:
                            os.remove(wav_path)
                            deleted += 1
                        except Exception as e:
                            log(f"⚠️ Не удалось удалить {wav_path}: {e}")
        log(f"\n🗑️ Удалено {deleted} .wav файлов")

    def log_output_w(msg, overwrite=False):
        if len(msg) > 400:
            return

        text_output_w.configure(state='normal')

        # Если строка похожа на прогресс (содержит % и / и frames/s)
        is_progress = "%" in msg and "/" in msg and "frames/s" in msg

        if is_progress:
            # Удаляем последнюю строку
            text_output_w.delete("end-2l", "end-1l")
            text_output_w.insert("end", msg + "\n")
        else:
            text_output_w.insert("end", msg + "\n")

        text_output_w.see("end")
        text_output_w.configure(state='disabled')

    def log_output_v(msg, overwrite=False):
        if len(msg) < 400:
            text_output_v.configure(state='normal')
            if overwrite:
                text_output_v.delete("end-2l", "end-1l")
            text_output_v.insert(tk.END, msg + "\n")
            text_output_v.see(tk.END)
            text_output_v.configure(state='disabled')

    def clear_output(text_output):
        text_output.configure(state='normal')
        text_output.delete("1.0", tk.END)
        text_output.configure(state='disabled')

    footer = tk.Label(root, text="Разработано TG @Smailkiller", fg="#7f8c8d", bg="#ecf0f3", font=("Segoe UI", 9, "italic"))
    footer.pack(pady=(0, 12))

    root.mainloop()

if __name__ == "__main__":
    gui_app()