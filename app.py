import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

TOKEN = os.getenv("BOT_TOKEN")

# === Flask untuk keep alive di Render ===
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running!"

# === Variabel global untuk pilihan masker user ===
user_choice = {}

# === Start Command ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("😎 Masker Mata", callback_data="mask_eye")],
        [InlineKeyboardButton("🎭 Masker Full Wajah", callback_data="mask_face")]
    ]
    await update.message.reply_text(
        "Pilih masker yang mau dipakai:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# === Simpan pilihan masker user ===
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    choice = query.data
    user_choice[query.from_user.id] = choice
    await query.message.reply_text(f"Kamu pilih: {choice}. Silakan kirim foto wajah kamu!")

# === Proses Foto ===
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    choice = user_choice.get(user_id, "mask_face")

    photo = update.message.photo[-1]
    file = await photo.get_file()
    filepath = "input.jpg"
    await file.download_to_drive(filepath)

    # Load foto user
    img = cv2.imread(filepath)

    # Deteksi wajah (pakai Haar Cascade bawaan OpenCV)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if len(faces) == 0:
        await update.message.reply_text("❌ Tidak ada wajah terdeteksi, coba foto lebih jelas.")
        return

    # Load masker sesuai pilihan
    if choice == "mask_eye":
        mask_path = "assets/mask_eye.png"
    else:
        mask_path = "assets/mask_face.png"

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # Tempel masker ke wajah pertama yang terdeteksi
    for (x, y, w, h) in faces:
        # Resize masker biar pas
        if choice == "mask_eye":
            mask_resized = cv2.resize(mask, (w, int(h/2)))   # hanya bagian mata
            y_offset = y + int(h/4)
        else:
            mask_resized = cv2.resize(mask, (w, h))          # full wajah
            y_offset = y

        # Pisahkan channel (RGBA)
        mask_rgb = mask_resized[:, :, :3]
        mask_alpha = mask_resized[:, :, 3] / 255.0

        # Tempel masker
        roi = img[y_offset:y_offset+mask_resized.shape[0], x:x+mask_resized.shape[1]]

        if roi.shape[0] != mask_resized.shape[0] or roi.shape[1] != mask_resized.shape[1]:
            continue

        for c in range(3):
            roi[:, :, c] = (mask_alpha * mask_rgb[:, :, c] + (1 - mask_alpha) * roi[:, :, c])

        img[y_offset:y_offset+mask_resized.shape[0], x:x+mask_resized.shape[1]] = roi
        break  # hanya wajah pertama

    # Simpan hasil
    output_path = "output.jpg"
    cv2.imwrite(output_path, img)

    # Kirim balik hasil
    await update.message.reply_photo(photo=InputFile(output_path), caption="✅ Masker berhasil ditempel!")

# === Main Bot ===
def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    application.run_polling()

if __name__ == "__main__":
    from threading import Thread
    Thread(target=lambda: app.run(host="0.0.0.0", port=8080)).start()
    main()
