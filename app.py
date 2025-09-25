import os
import cv2
import logging
from flask import Flask
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("BOT_TOKEN")

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running!"

# Simpan pilihan masker user
user_choice = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"User {update.message.from_user.id} memulai bot")
    keyboard = [
        [InlineKeyboardButton("😎 Masker Mata", callback_data="mask_eye")],
        [InlineKeyboardButton("🎭 Masker Full Wajah", callback_data="mask_face")]
    ]
    await update.message.reply_text(
        "Pilih masker yang mau dipakai:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    choice = query.data
    user_choice[query.from_user.id] = choice
    logger.info(f"User {query.from_user.id} memilih {choice}")
    await query.message.reply_text(f"Kamu pilih: {choice}. Silakan kirim foto wajah kamu!")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    choice = user_choice.get(user_id, "mask_face")
    logger.info(f"User {user_id} mengirim foto, pilihan masker: {choice}")

    photo = update.message.photo[-1]
    file = await photo.get_file()

    filepath = "input.jpg"
    try:
        await file.download(filepath)  # Ganti download_to_drive ke download
        logger.info(f"File foto berhasil didownload ke {filepath}")
    except Exception as e:
        logger.error(f"Gagal download foto: {e}")
        await update.message.reply_text("❌ Gagal mendownload foto, coba kirim ulang.")
        return

    img = cv2.imread(filepath)
    if img is None:
        logger.error("Gagal baca file foto setelah download")
        await update.message.reply_text("❌ Gagal membaca file foto, coba kirim ulang.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    logger.info(f"Terdeteksi {len(faces)} wajah")

    if len(faces) == 0:
        await update.message.reply_text("❌ Tidak ada wajah terdeteksi, coba foto lebih jelas.")
        return

    if choice == "mask_eye":
        mask_path = "assets/mask_eye.png"
    else:
        mask_path = "assets/mask_face.png"

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        logger.error(f"Gagal baca file masker di {mask_path}")
        await update.message.reply_text("❌ File masker tidak ditemukan, hubungi admin.")
        return

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
            logger.warning("Ukuran ROI tidak sesuai dengan masker, skip wajah ini")
            continue

        for c in range(3):
            roi[:, :, c] = (mask_alpha * mask_rgb[:, :, c] + (1 - mask_alpha) * roi[:, :, c])

        img[y_offset:y_offset+mask_resized.shape[0], x:x+mask_resized.shape[1]] = roi
        break  # Hanya satu wajah saja

    output_path = "output.jpg"
    cv2.imwrite(output_path, img)
    logger.info(f"Hasil foto dengan masker disimpan di {output_path}")

    await update.message.reply_photo(photo=InputFile(output_path), caption="✅ Masker berhasil ditempel!")

def main():
    if not TOKEN:
        logger.error("BOT_TOKEN environment variable belum di-set!")
        exit(1)
    logger.info("Bot mulai berjalan...")
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    application.run_polling()

if __name__ == "__main__":
    from threading import Thread
    Thread(target=lambda: app.run(host="0.0.0.0", port=8080)).start()
    main()
