import json
import os
import random
from telegram import (
    Update, InlineKeyboardMarkup, InlineKeyboardButton, InputFile
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
)

# === KONFIGURASI ===
TOKEN = "8466148433:AAH0yuuC3zetTYRysTkxmfCnc9JTqdwcXpI"

GC_NABRUTT = -1003098333444
DISCUSSION_GROUP = -1003033445498

CHANNEL_MAP = {
    "MENFESS": -1003033445498,
    "DONASI": -1003189592682,
    "MOAN": -1003196180758
}

THREAD_MAP = {
    "MENFESS": 1036,
    "DONASI": 393,
    "MOAN": 1038
}

# Cover default
COVER_CEWE = "https://telegra.ph/file/7a2d41a6ddf9-cover-cewe.jpg"
COVER_COWO = "https://telegra.ph/file/9bb77a6d9d0b-cover-cowo.jpg"

# File counter & ratings
COUNTER_FILE = "counter.json"
RATINGS_FILE = "ratings.json"

def load_counter():
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "r") as f:
            return json.load(f)
    return {"views": 10, "comments": 5}

def save_counter(data):
    with open(COUNTER_FILE, "w") as f:
        json.dump(data, f)

def load_ratings():
    if os.path.exists(RATINGS_FILE):
        with open(RATINGS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_ratings(data):
    with open(RATINGS_FILE, "w") as f:
        json.dump(data, f)

# === Bantu buat Inline Keyboard Rating 1-10 ===
def rating_keyboard(message_id):
    buttons = [[InlineKeyboardButton(str(i), callback_data=f"{message_id}|{i}") for i in range(1, 6)],
               [InlineKeyboardButton(str(i), callback_data=f"{message_id}|{i}") for i in range(6, 11)]]
    return InlineKeyboardMarkup(buttons)

# === Handle Konten User ===
async def handle_content(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caption_user = update.message.caption or update.message.text or "Tanpa keterangan"

    # Gender + emoji
    is_media = bool(update.message.photo or update.message.video)
    is_audio = bool(update.message.voice or update.message.audio)
    is_text = bool(update.message.text and not update.message.caption)

    gender, emoji, cover = "CEWE", "👙", COVER_CEWE
    if "COWO" in caption_user.upper():
        if is_media:
            gender, emoji, cover = "COWO", "🩲", COVER_COWO
        elif is_audio or is_text:
            gender, emoji, cover = "COWO", "🧒", COVER_COWO
    else:
        if is_media:
            gender, emoji, cover = "CEWE", "👙", COVER_CEWE
        elif is_audio or is_text:
            gender, emoji, cover = "CEWE", "🧕", COVER_CEWE

    # Counter dinamis
    counter = load_counter()
    views = counter["views"] + random.randint(2, 8)
    comments = counter["comments"] + random.randint(0, 3)
    counter["views"], counter["comments"] = views, comments
    save_counter(counter)

    # Tentukan kategori
    category = None
    for key in CHANNEL_MAP.keys():
        if key in caption_user.upper():
            category = key
            break
    if not category:
        await update.message.reply_text("❌ Tidak ada kategori cocok (MENFESS / DONASI / MOAN).")
        return

    target_ch = CHANNEL_MAP[category]
    thread_id = THREAD_MAP[category]

    caption_channel = (
        f"📌 <b>𝗚𝗘𝗡𝗗𝗘𝗥:</b> {gender}\n"
        f"{emoji} {caption_user}\n\n"
        f"✨ 𝗜𝗡𝗦𝗧𝗥𝗨𝗞𝗦𝗜 ✨\n"
        f"👉 Tekan tombol rating di bawah\n"
        f"👉 Komentar? Join diskusi (link di bawah)\n\n"
        f"#{gender}\n\n"
        f"👁️ {views}   💬 {comments}   🤖 Bot\n\n"
        f"🌟 RATING TERKINI: -\n"
        f"💬 Diskusi: https://t.me/c/1003033445498"
    )

    msg = None
    kb = rating_keyboard("tmp")  # placeholder, nanti diganti message_id

    if category == "MENFESS" and update.message.text:
        msg = await context.bot.send_message(chat_id=target_ch, text=caption_channel,
                                             parse_mode="HTML", reply_markup=kb)

elif category == "DONASI":
        if update.message.photo:
            msg = await context.bot.send_photo(chat_id=target_ch, photo=update.message.photo[-1].file_id,
                                               caption=caption_channel, parse_mode="HTML",
                                               has_spoiler=True, reply_markup=kb)
        elif update.message.video:
            msg = await context.bot.send_video(chat_id=target_ch, video=update.message.video.file_id,
                                               caption=caption_channel, parse_mode="HTML",
                                               has_spoiler=True, reply_markup=kb)
    elif category == "MOAN":
        if update.message.voice:
            msg = await context.bot.send_voice(chat_id=target_ch, voice=update.message.voice.file_id,
                                               caption=caption_channel, parse_mode="HTML",
                                               reply_markup=kb)
        elif update.message.audio:
            msg = await context.bot.send_audio(chat_id=target_ch, audio=update.message.audio.file_id,
                                               caption=caption_channel, parse_mode="HTML",
                                               reply_markup=kb)

    if not msg:
        await update.message.reply_text(f"❌ Format konten tidak sesuai untuk kategori {category}.")
        return

    # Update keyboard dengan message_id yang benar
    kb = rating_keyboard(msg.message_id)
    await context.bot.edit_message_reply_markup(chat_id=target_ch, message_id=msg.message_id, reply_markup=kb)

    # Kirim info ke GC NABRUTT
    link = f"https://t.me/c/{str(target_ch)[4:]}/{msg.message_id}"
    keyboard_info = InlineKeyboardMarkup([[InlineKeyboardButton("🔞 Lihat Full", url=link)]])
    info_text = f"📢 <b>{category} 18+</b>\n👤 Gender: {gender}\n👉 Klik tombol untuk lihat full di channel"
    await context.bot.send_message(chat_id=GC_NABRUTT, text=info_text, parse_mode="HTML",
                                   message_thread_id=thread_id, reply_markup=keyboard_info)

# === Handle Rating Callback ===
async def handle_rating(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data
    message_id_str, rating_str = data.split("|")
    rating = int(rating_str)
    message_id = str(message_id_str)

    ratings = load_ratings()
    if message_id not in ratings:
        ratings[message_id] = {}
    ratings[message_id][str(user_id)] = rating
    save_ratings(ratings)

    # Hitung rating rata-rata
    scores = list(ratings[message_id].values())
    avg = round(sum(scores) / len(scores), 1)

    # Edit caption di message
    try:
        message = query.message
        caption_lines = message.caption.split("\n")
        for i, line in enumerate(caption_lines):
            if line.startswith("🌟 RATING TERKINI:"):
                caption_lines[i] = f"🌟 RATING TERKINI: {avg}"
        new_caption = "\n".join(caption_lines)
        await context.bot.edit_message_caption(chat_id=message.chat_id, message_id=message.message_id,
                                               caption=new_caption, parse_mode="HTML",
                                               reply_markup=message.reply_markup)
        await query.answer(f"✅ Rating {rating} diterima!")
    except Exception as e:
        await query.answer("⚠️ Terjadi kesalahan.")

# === Command Start ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Kirim konten sesuai kategori:\n"
        "- MENFESS (teks only)\n"
        "- DONASI (foto/video)\n"
        "- MOAN (vn/audio)\n\n"
        "Gunakan caption yang menyertakan kategori (MENFESS/DONASI/MOAN)."
    )

# === Main App ===
def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_content))
    app.add_handler(CallbackQueryHandler(handle_rating))


print("Bot jalan...")
    app.run_polling()

if name == "main":
    main()
