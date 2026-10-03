import cv2

# Nastavení kamery (index 0 je obvykle výchozí USB kamera)
# Pro Windows může být nutné přidat cv2.CAP_DSHOW: cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0)

# Vypnutí automatické expozice (hodnoty se liší dle OS, 0.25 je časté pro manuální režim)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

# Funkce volané při změně posuvníku
def set_exposure(val):
    # Expozice mívá často záporné hodnoty (např. -10 až 0), upravte dle ovladače
    cap.set(cv2.CAP_PROP_EXPOSURE, val - 10)

def set_brightness(val):
    cap.set(cv2.CAP_PROP_BRIGHTNESS, val)

def set_contrast(val):
    cap.set(cv2.CAP_PROP_CONTRAST, val)

cv2.namedWindow("Genius Cam")

# Vytvoření posuvníků (Název, Okno, Výchozí hodnota, Max hodnota, Callback)
cv2.createTrackbar("Expozice", "Genius Cam", 5, 20, set_exposure)
cv2.createTrackbar("Jas", "Genius Cam", 50, 100, set_brightness)
cv2.createTrackbar("Kontrast", "Genius Cam", 50, 100, set_contrast)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nelze načíst obraz z kamery.")
        break
        
    cv2.imshow("Genius Cam", frame)
    
    # Klávesou 'q' program ukončíš
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()