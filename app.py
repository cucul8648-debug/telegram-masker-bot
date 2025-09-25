import os
import cv2
import logging
import numpy as np
from io import BytesIO
from flask import Flask
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from threading import Thread

# ---------- Logging ----------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("BOT_TOKEN")

# ---------- Flask healthcheck ----------
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running!"

# Simpan pilihan masker user
user_choice = {}

# ---------- Command /start ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("😎 Masker Mata", callback_data="mask_eye")],
        [InlineKeyboardButton("🎭 Masker Full Wajah", callback_data="mask_face")]
    ]
    await update.message.reply_text(
        "Pilih masker yang mau dipakai:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# ---------- Pilihan tombol ----------
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_choice[query.from_user.id] = query.data
    await query.message.reply_text(
        f"Kamu pilih: {query.data}. Silakan kirim foto wajah kamu!"
    )

# ---------- Proses foto langsung di memori ----------
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    choice = user_choice.get(user_id, "mask_face")

    # 1. Ambil file foto & download ke memori
    photo = update.message.photo[-1]
    tg_file = await photo.get_file()
    bio = BytesIO()
    await tg_file.download_to_memory(out=bio)
    bio.seek(0)

    # 2. Baca image dengan OpenCV dari buffer
    img_array = np.frombuffer(bio.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # 3. Deteksi wajah
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if len(faces) == 0:
        await update.message.reply_text("❌ Tidak ada wajah terdeteksi, coba foto lebih jelas.")
        return

    # 4. Pilih dan baca masker
    mask_path = "assets/mask_eye.png" if choice == "mask_eye" else "assets/mask_face.png"
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        await update.message.reply_text("❌ File masker tidak ditemukan, hubungi admin.")
        return

    # 5. Tempel masker pada wajah pertama
    for (x, y, w, h) in faces:
        if choice == "mask_eye":
            mask_resized = cv2.resize(mask, (w, int(h/2)))
            y_offset = y + int(h/4)
        else:
            mask_resized = cv2.resize(mask, (w, h))
            y_offset = y

        mask_rgb = mask_resized[:, :, :3]
        mask_alpha = mask_resized[:, :, 3] / 255.0
        roi = img[y_offset:y_offset+mask_resized.shape[0], x:x+mask_resized.shape[1]]

        if roi.shape[0] != mask_resized.shape[0] or roi.shape[1] != mask_resized.shape[1]:
            continue

        for c in range(3):
            roi[:, :, c] = (mask_alpha * mask_rgb[:, :, c] +
                            (1 - mask_alpha) * roi[:, :, c])
        img[y_offset:y_offset+mask_resized.shape[0], x:x+mask_resized.shape[1]] = roi
        break  # cukup satu wajah

    # 6. Encode hasil ke memori & kirim balik
    _, encoded = cv2.imencode(".jpg", img)
    out_buf = BytesIO(encoded.tobytes())
    out_buf.seek(0)

    await update.message.reply_photo(
        photo=InputFile(out_buf, filename="hasil_masker.jpg"),
        caption="✅ Masker berhasil ditempel!"
    )

# ---------- Main ----------
def main():
    if not TOKEN:
        logger.error("BOT_TOKEN environment variable belum di-set!")
        exit(1)

    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    application.run_polling()

if __name__ == "__main__":
    Thread(target=lambda: app.run(host="0.0.0.0", port=8080)).start()
    main()
