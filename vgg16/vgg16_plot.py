import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP PATH KE FILE CSV
# ==========================================
# Menggunakan path relatif terhadap lokasi skrip ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE1_CSV = os.path.join(SCRIPT_DIR, 'vgg16_stage1.csv')
STAGE2_CSV = os.path.join(SCRIPT_DIR, 'vgg16_stage2.csv')

# ==========================================
# 2. MEMUAT DAN MENGGABUNGKAN DATA HISTORY
# ==========================================
print("Memuat data training history...")
df_stage1 = pd.read_csv(STAGE1_CSV)
df_stage2 = pd.read_csv(STAGE2_CSV)

# Jumlah epoch pada stage 1 (titik awal fine-tuning)
stage1_epochs = len(df_stage1)

# Menggabungkan kedua stage menjadi satu DataFrame kontinu
# Reset epoch stage2 agar melanjutkan dari stage1
df_stage2 = df_stage2.copy()
df_stage2['epoch'] = df_stage2['epoch'] + stage1_epochs

df_combined = pd.concat([df_stage1, df_stage2], ignore_index=True)
total_epochs = len(df_combined)

print(f"Stage 1: {stage1_epochs} epoch (Feature Extraction)")
print(f"Stage 2: {len(df_stage2)} epoch (Fine-tuning)")
print(f"Total  : {total_epochs} epoch")

# ==========================================
# 3. PLOT TRAINING & VALIDATION LOSS
# ==========================================
epochs = df_combined['epoch'].values

fig1, ax1 = plt.subplots(figsize=(10, 6))

# Train Loss — garis solid biru tebal
ax1.plot(epochs, df_combined['loss'], 
         color='navy', linewidth=2.5, linestyle='-', 
         label='Train Loss')

# Validation Loss — garis dashed oranye tebal
ax1.plot(epochs, df_combined['val_loss'], 
         color='darkorange', linewidth=2.5, linestyle='--', 
         label='Val Loss')

# Garis vertikal penanda Stage 2 dimulai
ax1.axvline(x=stage1_epochs, color='gray', linestyle=':', linewidth=1.5,
            label='Stage 2 (Fine-tuning) Begins')

ax1.set_title('VGG16 Training & Validation Loss', fontsize=16)
ax1.set_xlabel('Epochs', fontsize=13)
ax1.set_ylabel('Loss', fontsize=13)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=11)
fig1.tight_layout()

# Simpan grafik loss
loss_path = os.path.join(SCRIPT_DIR, 'vgg16_loss_plot.png')
fig1.savefig(loss_path, dpi=300, bbox_inches='tight')
print(f"\nGrafik Loss disimpan di: {loss_path}")

# ==========================================
# 4. PLOT TRAINING & VALIDATION ACCURACY
# ==========================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Train Accuracy — garis solid hijau tua tebal
ax2.plot(epochs, df_combined['accuracy'], 
         color='darkgreen', linewidth=2.5, linestyle='-', 
         label='Train Accuracy')

# Validation Accuracy — garis dashed merah tebal
ax2.plot(epochs, df_combined['val_accuracy'], 
         color='red', linewidth=2.5, linestyle='--', 
         label='Val Accuracy')

# Garis vertikal penanda Stage 2 dimulai
ax2.axvline(x=stage1_epochs, color='gray', linestyle=':', linewidth=1.5,
            label='Stage 2 (Fine-tuning) Begins')

ax2.set_title('VGG16 Training & Validation Accuracy', fontsize=16)
ax2.set_xlabel('Epochs', fontsize=13)
ax2.set_ylabel('Accuracy', fontsize=13)
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=11)
fig2.tight_layout()

# Simpan grafik accuracy
acc_path = os.path.join(SCRIPT_DIR, 'vgg16_accuracy_plot.png')
fig2.savefig(acc_path, dpi=300, bbox_inches='tight')
print(f"Grafik Accuracy disimpan di: {acc_path}")

# Tampilkan kedua grafik
plt.show()

print("\nSelesai! Kedua grafik telah disimpan.")
