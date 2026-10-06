import cv2
import numpy as np
import os
import sys
import glob
import argparse

# ==========================================
# 1. DETEKCE A ROZPOZNÁNÍ BARVY V HSV PROSTORU
# ==========================================

def classify_hsv_region(hsv_img, mask):
    """
    Načte obraz v HSV prostoru a vyhodnotí dominantní barvu uvnitř vyhraněného kruhového regionu (masky).
    Rozpoznávané barvy: Červená, Zelená, Oranžová.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return None
        
    # Výběr pixelů uvnitř kruhové masky
    hsv_pixels = hsv_img[mask == 255]
    if len(hsv_pixels) == 0:
        return None
        
    h_vals = hsv_pixels[:, 0]
    s_vals = hsv_pixels[:, 1]
    v_vals = hsv_pixels[:, 2]
    
    # Filtrování šumu pozadí (vyžadujeme alespoň minimální saturaci a jas)
    valid_mask = (s_vals >= 30) & (v_vals >= 30)
    valid_h = h_vals[valid_mask]
    valid_s = s_vals[valid_mask]
    valid_v = v_vals[valid_mask]
    
    total_valid = len(valid_h)
    if total_valid < 5:
        avg_h = float(np.mean(h_vals))
        avg_s = float(np.mean(s_vals))
        avg_v = float(np.mean(v_vals))
        return {
            "color": "NEZŘETELNÁ / VYPNUTÁ",
            "confidence": 0.0,
            "display_bgr": (128, 128, 128),
            "hsv_mean": [round(avg_h, 1), round(avg_s, 1), round(avg_v, 1)],
            "percentages": {"red": 0.0, "orange": 0.0, "green": 0.0}
        }
        
    # Definice HSV odstínů Hue (0..179 v OpenCV)
    # 1. Červená (Červená prochází přes 0° a 180°): H <= 8 nebo H >= 160
    red_mask = (valid_h <= 8) | (valid_h >= 160)
    
    # 2. Oranžová: 9 <= H <= 30
    orange_mask = (valid_h >= 9) & (valid_h <= 30)
    
    # 3. Zelená: 32 <= H <= 88
    green_mask = (valid_h >= 32) & (valid_h <= 88)
    
    red_count = int(np.count_nonzero(red_mask))
    orange_count = int(np.count_nonzero(orange_mask))
    green_count = int(np.count_nonzero(green_mask))
    
    pct_red = red_count / float(total_valid)
    pct_orange = orange_count / float(total_valid)
    pct_green = green_count / float(total_valid)
    
    # Určení dominantní barvy s nejvyšším počtem pixelů
    counts = {
        "ČERVENÁ": (red_count, pct_red),
        "ORANŽOVÁ": (orange_count, pct_orange),
        "ZELENÁ": (green_count, pct_green)
    }
    
    best_color = max(counts.keys(), key=lambda k: counts[k][0])
    best_pct = counts[best_color][1]
    
    # Mapa BGR barvy pro vizualizaci
    color_bgr_map = {
        "ČERVENÁ": (0, 0, 255),
        "ORANŽOVÁ": (0, 140, 255),
        "ZELENÁ": (0, 255, 0)
    }
    
    avg_h = float(np.mean(valid_h))
    avg_s = float(np.mean(valid_s))
    avg_v = float(np.mean(valid_v))
    
    return {
        "color": best_color,
        "confidence": round(best_pct * 100.0, 1),
        "display_bgr": color_bgr_map.get(best_color, (255, 255, 255)),
        "hsv_mean": [round(avg_h, 1), round(avg_s, 1), round(avg_v, 1)],
        "percentages": {
            "red": round(pct_red * 100.0, 1),
            "orange": round(pct_orange * 100.0, 1),
            "green": round(pct_green * 100.0, 1)
        }
    }

# ==========================================
# 2. INTERAKTIVNÍ GUI PRO VYZNAČENÍ OBLASTI
# ==========================================

class ColorRecognizerGUI:
    def __init__(self, img_files, max_disp_w=1280, max_disp_h=800):
        self.img_files = img_files
        self.current_idx = 0
        self.max_disp_w = max_disp_w
        self.max_disp_h = max_disp_h
        
        self.raw_img = None
        self.hsv_img = None
        self.h, self.w = 0, 0
        self.scale = 1.0
        self.disp_w, self.disp_h = 0, 0
        
        self.drawing = False
        self.start_pt = None  # (x_orig, y_orig)
        self.current_pt = None # (x_orig, y_orig)
        
        self.circle_region = None
        self.recognition_result = None
        
        self.load_image(self.current_idx)

    def load_image(self, idx):
        self.current_idx = idx
        img_path = self.img_files[self.current_idx]
        self.raw_img = cv2.imread(img_path)
        if self.raw_img is None:
            raise FileNotFoundError(f"Nelze načíst obrázek: {img_path}")
            
        self.h, self.w = self.raw_img.shape[:2]
        
        # Převedení načteného obrázku do prostoru HSV
        self.hsv_img = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2HSV)
        
        # Přizpůsobení obrazovce
        self.scale = min(self.max_disp_w / float(self.w), self.max_disp_h / float(self.h), 1.0)
        self.disp_w = int(round(self.w * self.scale))
        self.disp_h = int(round(self.h * self.scale))
        
        self.circle_region = None
        self.recognition_result = None
        self.drawing = False

    def mouse_callback(self, event, x_disp, y_disp, flags, param):
        x_orig = int(round(x_disp / self.scale))
        y_orig = int(round(y_disp / self.scale))
        
        x_orig = max(0, min(self.w - 1, x_orig))
        y_orig = max(0, min(self.h - 1, y_orig))
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x_orig, y_orig)
            self.current_pt = (x_orig, y_orig)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_pt = (x_orig, y_orig)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.current_pt = (x_orig, y_orig)
                self.finalize_circle_selection()

    def finalize_circle_selection(self):
        if self.start_pt is None or self.current_pt is None:
            return
            
        x1, y1 = self.start_pt
        x2, y2 = self.current_pt
        
        center = (int(x1), int(y1))
        radius = int(np.hypot(x2 - x1, y2 - y1))
        if radius < 3:
            radius = int(25 / self.scale)
            
        self.circle_region = {"center": center, "radius": radius}
        
        # Vytvoření kruhové masky v originálním rozlišení
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        # Vyhodnocení barvy v HSV
        self.recognition_result = classify_hsv_region(self.hsv_img, mask)
        
        if self.recognition_result:
            res = self.recognition_result
            img_name = os.path.basename(self.img_files[self.current_idx])
            print(f"\n[ROZPOZNÁNÍ BARVY] Soubor: {img_name} | Střed: {center}, Poloměr: {radius} px")
            print(f"   -> Výsledek: {res['color']} (Jistota: {res['confidence']}%)")
            print(f"   -> Průměrné HSV: H={res['hsv_mean'][0]}, S={res['hsv_mean'][1]}, V={res['hsv_mean'][2]}")
            print(f"   -> Zastoupení: Červená={res['percentages']['red']}%, Oranžová={res['percentages']['orange']}%, Zelená={res['percentages']['green']}%")

    def render(self):
        canvas = self.raw_img.copy()
        
        line_thick = max(2, int(round(2 / self.scale)))
        font_scale = max(0.55, 0.6 / self.scale)
        text_thick = max(1, int(round(1.5 / self.scale)))
        
        # 1. Náhled právě taženého kruhu
        if self.drawing and self.start_pt and self.current_pt:
            x1, y1 = self.start_pt
            x2, y2 = self.current_pt
            r = int(np.hypot(x2 - x1, y2 - y1))
            cv2.circle(canvas, (x1, y1), r, (255, 255, 255), line_thick, cv2.LINE_AA)
            cv2.drawMarker(canvas, (x1, y1), (255, 255, 255), cv2.MARKER_CROSS, int(16/self.scale), line_thick)

        # 2. Vykreslení uloženého kruhového regionu a výsledku rozpoznání
        if self.circle_region:
            c = self.circle_region["center"]
            r = self.circle_region["radius"]
            
            bgr_col = (0, 255, 255)
            if self.recognition_result:
                bgr_col = self.recognition_result["display_bgr"]
                
            cv2.circle(canvas, c, r, bgr_col, line_thick, cv2.LINE_AA)
            cv2.drawMarker(canvas, c, bgr_col, cv2.MARKER_CROSS, int(20/self.scale), line_thick)

        # 3. Změna velikosti plného obrázku na rozlišení okna
        disp_canvas = cv2.resize(canvas, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)

        # 4. Horní stavová lišta
        img_name = os.path.basename(self.img_files[self.current_idx])
        overlay = disp_canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.disp_w, 105), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, disp_canvas, 0.2, 0, disp_canvas)
        
        header_str = f"Snímek [{self.current_idx+1}/{len(self.img_files)}]: {img_name} ({self.w}x{self.h} px)"
        cv2.putText(disp_canvas, header_str, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        controls_str = "Tažením myši vyznačte kruhový region  |  [MEZERNÍK/N] Další  |  [P] Předchozí  |  [Z] Smazat  |  [Q] Konec"
        cv2.putText(disp_canvas, controls_str, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 200), 1)

        # 5. Panel s výsledkem rozpoznání barvy
        if self.recognition_result:
            res = self.recognition_result
            col_name = res["color"]
            conf = res["confidence"]
            bgr_col = res["display_bgr"]
            
            # Barevný vzorek (obdélníček)
            cv2.rectangle(disp_canvas, (15, 62), (45, 92), bgr_col, -1)
            cv2.rectangle(disp_canvas, (15, 62), (45, 92), (255, 255, 255), 1)
            
            res_str = f"ROZPOZNANÁ BARVA: {col_name} (Jistota: {conf}%)"
            cv2.putText(disp_canvas, res_str, (55, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, bgr_col, 2)
            
            details_str = f"HSV průměr: [H={res['hsv_mean'][0]}, S={res['hsv_mean'][1]}, V={res['hsv_mean'][2]}]  |  Čer: {res['percentages']['red']}%  Ora: {res['percentages']['orange']}%  Zel: {res['percentages']['green']}%"
            cv2.putText(disp_canvas, details_str, (55, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        else:
            cv2.putText(disp_canvas, "Vyznačte myší kruhový region pro rozpoznání barvy (Červená / Zelená / Oranžová)...",
                        (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)

        return disp_canvas

    def run(self):
        window_name = "Rozpoznani barvy v HSV prostoru (real_img)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.disp_w, self.disp_h)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\n=======================================================")
        print("     ROZPOZNÁVÁNÍ BARVY (HSV PROSTOR - real_img)       ")
        print("=======================================================")
        print(" * Tažením myší vyznačte kruhový region na obrázku.")
        print(" * Program v HSV prostoru rozezná barvu: ČERVENÁ, ZELENÁ, ORANŽOVÁ.")
        print(" * Klávesy [MEZERNÍK] / [N]: Další obrázek ve složce real_img")
        print(" * Klávesy [P]: Předchozí obrázek")
        print(" * Klávesa [Z]: Vymazání vyznačené oblasti")
        print(" * Klávesy [ESC] / [Q]: Ukončení aplikace\n")
        
        while True:
            disp_frame = self.render()
            cv2.imshow(window_name, disp_frame)
            key = cv2.waitKey(30) & 0xFF
            
            if key in (ord(' '), ord('n'), ord('N'), 83):
                next_idx = (self.current_idx + 1) % len(self.img_files)
                self.load_image(next_idx)
                cv2.resizeWindow(window_name, self.disp_w, self.disp_h)
            elif key in (ord('p'), ord('P'), 81):
                prev_idx = (self.current_idx - 1) % len(self.img_files)
                self.load_image(prev_idx)
                cv2.resizeWindow(window_name, self.disp_w, self.disp_h)
            elif key in (ord('z'), ord('Z'), ord('c'), ord('C')):
                self.circle_region = None
                self.recognition_result = None
            elif key in (ord('s'), ord('S')):
                os.makedirs("real_img_output", exist_ok=True)
                img_name = os.path.basename(self.img_files[self.current_idx])
                out_path = os.path.join("real_img_output", f"color_{img_name}")
                cv2.imwrite(out_path, disp_frame)
                print(f"--> Uložen výstupní obrázek do: {out_path}")
            elif key in (27, ord('q'), ord('Q')):
                break
                
        cv2.destroyAllWindows()

# ==========================================
# 3. KONTROLA SLOŽKY S OBRÁZKY
# ==========================================

def get_real_img_files(folder_path="real_img"):
    os.makedirs(folder_path, exist_ok=True)
    files = sorted(glob.glob(os.path.join(folder_path, "*.*")))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    return files

# ==========================================
# 4. SPUŠTĚNÍ PROGRAMU
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Rozpoznávání barev (Červená, Zelená, Oranžová) v HSV prostoru")
    parser.add_argument("--dir", type=str, default="real_img", help="Složka s obrázky pro rozpoznání (výchozí: real_img)")
    args = parser.parse_args()
    
    img_files = get_real_img_files(args.dir)
    
    if not img_files:
        print(f"[CHYBA] Složka '{args.dir}' neobsahuje žádné obrázky (.png, .jpg, .jpeg, .bmp).")
        print(f"Vložte obrázky do složky '{args.dir}' a spusťte program znovu.")
        return
        
    print(f"Nalezeno {len(img_files)} obrázků ve složce '{args.dir}'. Spouštím aplikaci...")
    gui = ColorRecognizerGUI(img_files)
    gui.run()

if __name__ == "__main__":
    main()
