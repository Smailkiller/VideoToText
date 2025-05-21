import tkinter as tk
from tkinter import filedialog, scrolledtext, PhotoImage
import threading
import sys
import os
import wave
import json
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

stop_flag = False

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)

def extract_audio(video_path, wav_path):
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])

def transcribe_audio(wav_path, model, log):
    global stop_flag
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    total = wf.getnframes()
    read = 0
    last_percent = -1
    while True:
        if stop_flag:
            log("\n❌ Процесс остановлен пользователем.")
            return None
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)
        read += 4000
        percent = min(100, int(read / total * 100))
        if percent != last_percent:
            log(f"\r    🟩 Прогресс транскрипции: {percent}%", overwrite=True)
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

def process_videos(base_folder, model_path, log):
    global stop_flag
    stop_flag = False
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
        if stop_flag:
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
            result = transcribe_audio(wav_path, model, log)
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

def start_process(video_path, model_path, log, stop_button):
    stop_button.config(state='normal')
    threading.Thread(target=process_videos, args=(video_path, model_path, log), daemon=True).start()

def stop_process():
    global stop_flag
    stop_flag = True

def gui_app():
    root = tk.Tk()
    root.title("🎙️ Video Transcriber GUI")
    root.geometry("740x640")
    root.configure(bg="#ecf0f3")

    # ✅ установка иконки из .png
    icon_path = resource_path("favicon.png")
    if os.path.exists(icon_path):
        icon_img = PhotoImage(file=icon_path)
        root.iconphoto(False, icon_img)

    def choose_video_folder():
        path = filedialog.askdirectory()
        if path:
            entry_video.delete(0, tk.END)
            entry_video.insert(0, path)

    def choose_model_folder():
        path = filedialog.askdirectory()
        if path:
            entry_model.delete(0, tk.END)
            entry_model.insert(0, path)

    def log_output(msg, overwrite=False):
        if overwrite:
            text_output.delete("end-2l", "end-1l")
        text_output.insert(tk.END, msg + "\n")
        text_output.see(tk.END)

    def clear_output():
        text_output.delete("1.0", tk.END)

    # Стили
    bg_main = "#ecf0f3"
    entry_style = {"bg": "#ffffff", "relief": "flat", "highlightthickness": 1, "highlightbackground": "#d0d5db", "bd": 0, "font": ("Segoe UI", 10)}
    btn_style = {"bg": "#2c3e50", "fg": "#ffffff", "activebackground": "#34495e", "activeforeground": "#ecf0f1",
                 "relief": "flat", "bd": 0, "font": ("Segoe UI", 10, "bold"), "cursor": "hand2", "padx": 10, "pady": 6}

    tk.Label(root, text="Папка с видео:", bg=bg_main, font=("Segoe UI", 10, "bold")).pack(anchor='w', padx=15, pady=(15, 0))
    frame_video = tk.Frame(root, bg=bg_main)
    frame_video.pack(fill='x', padx=15)
    entry_video = tk.Entry(frame_video, **entry_style)
    entry_video.pack(side='left', fill='x', expand=True, ipady=6, pady=5)
    tk.Button(frame_video, text="Обзор", command=choose_video_folder, **btn_style).pack(side='left', padx=5)

    tk.Label(root, text="Папка с моделью Vosk:", bg=bg_main, font=("Segoe UI", 10, "bold")).pack(anchor='w', padx=15, pady=(10, 0))
    frame_model = tk.Frame(root, bg=bg_main)
    frame_model.pack(fill='x', padx=15)
    entry_model = tk.Entry(frame_model, **entry_style)
    entry_model.pack(side='left', fill='x', expand=True, ipady=6, pady=5)
    tk.Button(frame_model, text="Обзор", command=choose_model_folder, **btn_style).pack(side='left', padx=5)

    frame_buttons = tk.Frame(root, bg=bg_main)
    frame_buttons.pack(pady=15)
    tk.Button(frame_buttons, text="▶️ Обработка", command=lambda: start_process(entry_video.get(), entry_model.get(), log_output, btn_stop), **btn_style).pack(side='left', padx=5)
    btn_stop = tk.Button(frame_buttons, text="⛔ Остановить", command=stop_process, state='disabled', **btn_style)
    btn_stop.pack(side='left', padx=5)
    tk.Button(frame_buttons, text="🧹 Очистить", command=clear_output, **btn_style).pack(side='left', padx=5)

    text_output = scrolledtext.ScrolledText(root, height=18, bg="#ffffff", relief="flat", font=("Consolas", 10), wrap="word")
    text_output.pack(fill='both', expand=True, padx=15, pady=(0, 10))

    footer = tk.Label(root, text="Разработано TG @Smailkiller", fg="#7f8c8d", bg=bg_main, font=("Segoe UI", 9, "italic"))
    footer.pack(pady=(0, 12))

    root.mainloop()

if __name__ == "__main__":
    gui_app()
