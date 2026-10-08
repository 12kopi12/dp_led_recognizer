import cv2
import sys

# Otevření kamery (CAP_DSHOW je na Windows klíčový pro správné čtení/zápis vlastností)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Chyba: Kameru se nepodařilo otevřít.")
    sys.exit()

# Vypnutí automatik, aby manuální hodnoty zůstaly aktivní
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)

# Načtení aktuálních hodnot z kamery
cur_exp = int(cap.get(cv2.CAP_PROP_EXPOSURE))
cur_bright = int(cap.get(cv2.CAP_PROP_BRIGHTNESS))
cur_contrast = int(cap.get(cv2.CAP_PROP_CONTRAST))
cur_sat = int(cap.get(cv2.CAP_PROP_SATURATION))
cur_sharp = int(cap.get(cv2.CAP_PROP_SHARPNESS))
cur_wb = int(cap.get(cv2.CAP_PROP_WB_TEMPERATURE))

print("--- Načtené výchozí hodnoty ---")
print(f"Expozice: {cur_exp} (trackbar offset +13)")
print(f"Jas: {cur_bright}")
print(f"Kontrast: {cur_contrast}")
print(f"Sytost: {cur_sat}")
print(f"Ostrost: {cur_sharp}")
print(f"Vyvážení bílé: {cur_wb}")

# Callback funkce pro posuvníky
# Expozice v DSHOW bývá v logaritmických krocích cca -13 až 0 (offset 13 umožňuje rozsah 0..13)
def set_exposure(val):
    cap.set(cv2.CAP_PROP_EXPOSURE, val - 13)

def set_brightness(val):
    cap.set(cv2.CAP_PROP_BRIGHTNESS, val)

def set_contrast(val):
    cap.set(cv2.CAP_PROP_CONTRAST, val)

def set_saturation(val):
    cap.set(cv2.CAP_PROP_SATURATION, val)

def set_sharpness(val):
    cap.set(cv2.CAP_PROP_SHARPNESS, val)

def set_wb(val):
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, val)

# Inicializace okna a trackbarů
window_name = "Genius WideCam F100 V2"
cv2.namedWindow(window_name)

# Nastavení výchozí pozice trackbaru podle zjištěného stavu (ošetřeno proti -1 při chybě čtení)
init_exp_pos = max(0, min(13, cur_exp + 13)) if cur_exp != -1 else 6
init_bright = max(0, min(255, cur_bright)) if cur_bright != -1 else 128
init_contrast = max(0, min(255, cur_contrast)) if cur_contrast != -1 else 32
init_sat = max(0, min(255, cur_sat)) if cur_sat != -1 else 64
init_sharp = max(0, min(255, cur_sharp)) if cur_sharp != -1 else 2
init_wb = max(2800, min(6500, cur_wb)) if cur_wb != -1 else 4500

cv2.createTrackbar("Expozice (-13 az 0)", window_name, init_exp_pos, 13, set_exposure)
cv2.createTrackbar("Jas (0-255)", window_name, init_bright, 255, set_brightness)
cv2.createTrackbar("Kontrast (0-255)", window_name, init_contrast, 255, set_contrast)
cv2.createTrackbar("Sytost (0-255)", window_name, init_sat, 255, set_saturation)
cv2.createTrackbar("Ostrost (0-255)", window_name, init_sharp, 255, set_sharpness)
cv2.createTrackbar("Bílá (Kelvin)", window_name, init_wb, 6500, set_wb)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nelze načíst snímek z kamery.")
        break

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()