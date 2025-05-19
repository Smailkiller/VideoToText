import os
import wave
import json
import sys
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

# ==========================
# 🔧 ПАРАМЕТРЫ ДЛЯ НАСТРОЙКИ
# ==========================
video_root_path = r"path1"                     # Путь к папке с видеофайлами (возможно с подпапками)
vosk_model_path = r"path2"                     # Путь к модели Vosk
# ==========================

def extract_audio(video_path, wav_path):
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])

def transcribe_audio(wav_path, model):
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    total = wf.getnframes()
    read = 0
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)
        read += 4000
        percent = min(100, int(read / total * 100))
        print(f"\r    🟩 Прогресс транскрипции: {percent}%", end="")
        sys.stdout.flush()
    print("\n    ✅ Распознавание завершено.")
    return json.loads(rec.FinalResult()).get("text", "")

def process_videos(base_folder, model_path):
    print(f"🚀 Загрузка модели Vosk из: {model_path}")
    model = Model(model_path)

    video_files = []
    for root, _, files in os.walk(base_folder):
        for f in files:
            video_files.append(os.path.join(root, f))

    print(f"🔎 Найдено видеофайлов: {len(video_files)}\n")

    error_log_path = os.path.join(base_folder, "error_log.txt")
    if os.path.exists(error_log_path):
        os.remove(error_log_path)

    supported_ext = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpeg", ".mpg")

    for idx, video_path in enumerate(video_files, 1):
        if not video_path.lower().endswith(supported_ext):
            continue

        print(f"🔄 [{idx}/{len(video_files)}] Обработка файла: {video_path}")
        base_name = os.path.splitext(video_path)[0]
        wav_path = base_name + ".wav"
        txt_path = base_name + ".txt"

        try:
            print("    🎧 Извлечение аудио...")
            extract_audio(video_path, wav_path)
        except Exception as e:
            print(f"    ⚠️ Ошибка при извлечении аудио: {e}")
            with open(error_log_path, "a", encoding="utf-8") as log_f:
                log_f.write(f"[Audio Extract Error] {video_path}: {e}\n")
            continue

        try:
            print("    🖋️ Транскрибация...")
            text = transcribe_audio(wav_path, model)

            video_filename = os.path.basename(base_name)
            with wave.open(wav_path, "rb") as wf:
                duration = round(wf.getnframes() / wf.getframerate(), 1)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"# {video_filename} | Длительность: {duration} сек\n")
                f.write(text)

            print(f"    📏 Готово! Сохранён текст: {txt_path}\n")

        except Exception as e:
            print(f"    ⚠️ Ошибка при транскрибации: {e}")
            with open(error_log_path, "a", encoding="utf-8") as log_f:
                log_f.write(f"[Transcription Error] {video_path}: {e}\n")
            continue

    print("✅ Все видео обработаны.")

if __name__ == "__main__":
    process_videos(video_root_path, vosk_model_path)
