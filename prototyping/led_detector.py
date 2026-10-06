import cv2
import numpy as np
import json
import os
import sys
import glob
import argparse

# ==========================================
# 1. ARUCO & HOMOGRAPHY HELPERS
# ==========================================

ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

def detect_aruco_markers(img):
    """
    Detekuje ArUco markery v obrázku.
    Vrací slovník { marker_id: pole_4_rohů (shape: 4x2 float32) }
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    
    if hasattr(cv2.aruco, 'ArucoDetector'):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)
        
    detected = {}
    if ids is not None:
        for idx, m_id in enumerate(ids.flatten()):
            detected[int(m_id)] = corners[idx][0].astype(np.float32) # 4x2 float array
    return detected

def compute_direct_homography(ref_markers_dict, target_markers_dict):
    """
    Spočítá přímou Homografii H_ref2target z korespondujících rohových bodů markerů
    mezi referenčním obrázkem a cílovým obrázkem.
    Využívá všechny dostupné rohy společně detekovaných markerů.
    """
    ref_pts = []
    target_pts = []
    
    # Najít společné ID markerů
    common_ids = sorted(set(ref_markers_dict.keys()) & set(target_markers_dict.keys()))
    
    for m_id in common_ids:
        ref_pts.extend(ref_markers_dict[m_id])
        target_pts.extend(target_markers_dict[m_id])
        
    if len(ref_pts) < 4:
        return None, len(common_ids)
        
    ref_pts = np.array(ref_pts, dtype=np.float32)
    target_pts = np.array(target_pts, dtype=np.float32)
    
    H_ref2target, mask = cv2.findHomography(ref_pts, target_pts, cv2.RANSAC, 5.0)
    return H_ref2target, len(common_ids)

def transform_point(pt, H):
    """Transformuje 2D bod (x, y) pomocí matice homografie H."""
    if H is None:
        return None
    vec = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
    res = H @ vec
    if res[2] != 0:
        return (float(res[0] / res[2]), float(res[1] / res[2]))
    return None

def transform_points(pts, H):
    """Transformuje pole 2D bodů pomocí matice homografie H."""
    if H is None or len(pts) == 0:
        return None
    pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts_arr, H)
    return dst.reshape(-1, 2)

# ==========================================
# 2. INTERAKTIVNÍ ANOTAČNÍ GUI (MALOVÁNÍ)
# ==========================================

class LEDAnnotator:
    def __init__(self, image_path, max_disp_w=1280, max_disp_h=800):
        self.image_path = image_path
        self.raw_img = cv2.imread(image_path)
        if self.raw_img is None:
            raise FileNotFoundError(f"Obrázek nenalezen: {image_path}")
            
        self.h, self.w = self.raw_img.shape[:2]
        
        # Měřítko zobrazení pro přizpůsobení obrazovce
        self.scale = min(max_disp_w / float(self.w), max_disp_h / float(self.h), 1.0)
        self.disp_w = int(round(self.w * self.scale))
        self.disp_h = int(round(self.h * self.scale))
        
        self.detected_markers = detect_aruco_markers(self.raw_img)
        
        # Stav nástrojů (souřadnice v původním plném rozlišení)
        self.active_led = 1  # 1 nebo 2
        self.active_shape = "circle"  # "circle" nebo "rectangle"
        self.drawing = False
        self.start_pt = None  # (x_orig, y_orig)
        self.current_pt = None # (x_orig, y_orig)
        
        # Uložené anotace v plném rozlišení
        self.annotations = {1: None, 2: None}
        
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
                self.save_current_drawn_shape()
                
    def save_current_drawn_shape(self):
        if self.start_pt is None or self.current_pt is None:
            return
            
        x1, y1 = self.start_pt
        x2, y2 = self.current_pt
        
        if self.active_shape == "circle":
            center = (int(x1), int(y1))
            radius = int(np.hypot(x2 - x1, y2 - y1))
            if radius < 3:
                radius = int(20 / self.scale)
            self.annotations[self.active_led] = {
                "shape_type": "circle",
                "center": center,
                "radius": radius
            }
        elif self.active_shape == "rectangle":
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            w = xmax - xmin
            h = ymax - ymin
            if w < 5 or h < 5:
                w, h = int(40 / self.scale), int(40 / self.scale)
                xmin, ymin = max(0, x1 - w//2), max(0, y1 - h//2)
            
            # Uložení 4 rohů obdélníku pro přesnou perspektivní transformaci
            corners = [
                [int(xmin), int(ymin)],
                [int(xmin + w), int(ymin)],
                [int(xmin + w), int(ymin + h)],
                [int(xmin), int(ymin + h)]
            ]
            
            self.annotations[self.active_led] = {
                "shape_type": "rectangle",
                "bbox": [int(xmin), int(ymin), int(w), int(h)],
                "center": (int(xmin + w / 2), int(ymin + h / 2)),
                "corners": corners
            }
            
    def extract_color_profile(self, ann):
        """Vytáhne barevný profil z označené oblasti (ROI)."""
        cx, cy = ann["center"]
        if ann["shape_type"] == "circle":
            r = ann["radius"]
            xmin, ymin = max(0, cx - r), max(0, cy - r)
            xmax, ymax = min(self.w, cx + r), min(self.h, cy + r)
        else:
            bx, by, bw, bh = ann["bbox"]
            xmin, ymin = max(0, bx), max(0, by)
            xmax, ymax = min(self.w, bx + bw), min(self.h, by + bh)
            
        roi = self.raw_img[ymin:ymax, xmin:xmax]
        if roi.size == 0:
            return {"dominant_channel": "red"}
            
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        b, g, r_ch = cv2.split(roi)
        
        avg_b, avg_g, avg_r = float(np.mean(b)), float(np.mean(g)), float(np.mean(r_ch))
        avg_h = float(np.mean(hsv_roi[:, :, 0]))
        avg_s = float(np.mean(hsv_roi[:, :, 1]))
        avg_v = float(np.mean(hsv_roi[:, :, 2]))
        
        if avg_r > avg_g and avg_r > avg_b:
            dom = "red"
        elif avg_g > avg_r and avg_g > avg_b:
            dom = "green"
        elif avg_b > avg_r and avg_b > avg_g:
            dom = "blue"
        else:
            dom = "white"
            
        return {
            "dominant_channel": dom,
            "mean_bgr": [round(avg_b, 1), round(avg_g, 1), round(avg_r, 1)],
            "mean_hsv": [round(avg_h, 1), round(avg_s, 1), round(avg_v, 1)]
        }

    def render(self):
        canvas = self.raw_img.copy()
        
        line_thick = max(2, int(round(2 / self.scale)))
        font_scale = max(0.55, 0.6 / self.scale)
        text_thick = max(1, int(round(1.5 / self.scale)))
        
        # 1. Vykreslení detekovaných ArUco markerů
        for m_id, corners in self.detected_markers.items():
            pts = corners.astype(int)
            cv2.polylines(canvas, [pts], True, (0, 255, 0), line_thick)
            c = pts.mean(axis=0).astype(int)
            cv2.putText(canvas, f"ID:{m_id}", (c[0]-int(25/self.scale), c[1]+int(8/self.scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), text_thick)
                        
        if len(self.detected_markers) >= 2:
            m_centers = [self.detected_markers[i].mean(axis=0).astype(int) 
                         for i in sorted(self.detected_markers.keys())]
            for i in range(len(m_centers)):
                cv2.line(canvas, tuple(m_centers[i]), tuple(m_centers[(i+1)%len(m_centers)]),
                         (100, 200, 100), max(1, line_thick//2))

        # 2. Vykreslení uložených anotací
        colors = {1: (0, 0, 255), 2: (0, 255, 255)} # Červená / Žlutá
        for led_id, ann in self.annotations.items():
            if ann is not None:
                col = colors[led_id]
                if ann["shape_type"] == "circle":
                    cv2.circle(canvas, ann["center"], ann["radius"], col, line_thick)
                    cv2.drawMarker(canvas, ann["center"], col, cv2.MARKER_CROSS, int(20/self.scale), line_thick)
                    cv2.putText(canvas, f"LED {led_id} (Kruh)", (ann["center"][0]+ann["radius"]+int(10/self.scale), ann["center"][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, text_thick)
                elif ann["shape_type"] == "rectangle":
                    x, y, w, h = ann["bbox"]
                    cv2.rectangle(canvas, (x, y), (x+w, y+h), col, line_thick)
                    cv2.drawMarker(canvas, ann["center"], col, cv2.MARKER_CROSS, int(20/self.scale), line_thick)
                    cv2.putText(canvas, f"LED {led_id} (Obdelnik)", (x + w + int(10/self.scale), y + int(25/self.scale)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, text_thick)

        # 3. Náhled právě taženého tvaru
        if self.drawing and self.start_pt and self.current_pt:
            col = colors[self.active_led]
            x1, y1 = self.start_pt
            x2, y2 = self.current_pt
            if self.active_shape == "circle":
                r = int(np.hypot(x2 - x1, y2 - y1))
                cv2.circle(canvas, (x1, y1), r, col, line_thick, cv2.LINE_AA)
            elif self.active_shape == "rectangle":
                cv2.rectangle(canvas, (x1, y1), (x2, y2), col, line_thick, cv2.LINE_AA)

        # 4. Resize pro obrazovku
        disp_canvas = cv2.resize(canvas, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)

        # 5. Ovládací lišta
        overlay = disp_canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.disp_w, 65), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, disp_canvas, 0.2, 0, disp_canvas)
        
        mode_str = f"Aktivni LED: [{self.active_led}] ({'Cervena' if self.active_led==1 else 'Zluta'})  |  Tvar: [{self.active_shape.upper()}]  |  Rozliseni: {self.w}x{self.h}"
        cv2.putText(disp_canvas, mode_str, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        controls_str = "[1/2] Prepnout LED  |  [C] Kruznice  |  [R] Obdelnik  |  [Z] Smazat  |  [ENTER/S] Ulozit JSON"
        cv2.putText(disp_canvas, controls_str, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 230, 200), 1)
        
        return disp_canvas

    def run(self):
        window_name = "Oznaceni LED diod (Malovani)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.disp_w, self.disp_h)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\n=======================================================")
        print(f"    ANOTACE LED DIOD (Rozlišení: {self.w}x{self.h} px)")
        print(f"    Přizpůsobeno obrazovce: {self.disp_w}x{self.disp_h} px (Měřítko: {self.scale:.2f})")
        print("=======================================================")
        print(" * Klávesy [1] / [2]: Přepínání mezi LED 1 a LED 2")
        print(" * Klávesy [C] / [R]: Přepínání tvaru (C = Kružnice, R = Obdélník)")
        print(" * Tažení myší: Nakreslení LED diody v obrázku")
        print(" * Klávesa [Z]: Smazání anotace pro aktivní LED")
        print(" * Klávesa [ENTER] / [S]: Uložení konfigurace do JSON a pokračování\n")
        
        while True:
            disp_frame = self.render()
            cv2.imshow(window_name, disp_frame)
            key = cv2.waitKey(20) & 0xFF
            
            if key == ord('1'):
                self.active_led = 1
            elif key == ord('2'):
                self.active_led = 2
            elif key in (ord('c'), ord('C')):
                self.active_shape = "circle"
            elif key in (ord('r'), ord('R')):
                self.active_shape = "rectangle"
            elif key in (ord('z'), ord('Z')):
                self.annotations[self.active_led] = None
            elif key in (13, ord('s'), ord('S')):
                if self.annotations[1] is None or self.annotations[2] is None:
                    print("[UPOZORNĚNÍ] Označte prosím OBĚ LED diody (LED 1 i LED 2) před uložením!")
                else:
                    cv2.destroyWindow(window_name)
                    return self.export_json_config()
            elif key in (27, ord('q'), ord('Q')):
                cv2.destroyWindow(window_name)
                print("Aplikace byla ukončena uživatelem.")
                sys.exit(0)

    def export_json_config(self):
        """Uloží marker souřadnice z referenčního obrázku a přesné 2D vztahy k LED."""
        if not self.detected_markers:
            raise ValueError("K výpočtu relace je potřeba detekovat alespoň jeden ArUco marker!")
            
        ref_markers_serializable = {
            str(m_id): corners.tolist() for m_id, corners in self.detected_markers.items()
        }
        
        config = {
            "reference_image": os.path.basename(self.image_path),
            "image_dimensions": {"width": self.w, "height": self.h},
            "ref_markers": ref_markers_serializable,
            "leds": []
        }
        
        for led_id in [1, 2]:
            ann = self.annotations[led_id]
            if ann is None:
                continue
                
            cx_pix, cy_pix = ann["center"]
            color_prof = self.extract_color_profile(ann)
            
            # Relativní odchylky v pixelech k centrům markerů v referenčním snímku
            rel_markers = {}
            for m_id, m_corners in self.detected_markers.items():
                m_c_pix = m_corners.mean(axis=0)
                dx = cx_pix - m_c_pix[0]
                dy = cy_pix - m_c_pix[1]
                dist = float(np.hypot(dx, dy))
                rel_markers[str(m_id)] = {
                    "dx": round(float(dx), 2),
                    "dy": round(float(dy), 2),
                    "dist": round(dist, 2)
                }

            led_entry = {
                "id": led_id,
                "name": f"LED {led_id}",
                "shape_type": ann["shape_type"],
                "ref_center": [cx_pix, cy_pix],
                "color_profile": color_prof,
                "relative_to_markers": rel_markers
            }
            
            if ann["shape_type"] == "circle":
                led_entry["ref_radius"] = ann["radius"]
            elif ann["shape_type"] == "rectangle":
                led_entry["ref_bbox"] = ann["bbox"]
                led_entry["ref_corners"] = ann["corners"]
                
            config["leds"].append(led_entry)
            
        return config

# ==========================================
# 3. DETEKCE A LOKALIZACE V NOVÝCH OBRÁZKÁCH
# ==========================================

def detect_and_refine_led_in_image(img, config):
    """
    1. Detekuje ArUco markery v novém obrázku.
    2. Vypočítá přímou Homografii z referenčního obrázku do nového obrázku.
    3. Transformuje geometrickou polohu i přesný tvar LED (kruh nebo perspektivní obdélník).
    4. Provedení optické detekce svitu LED v lokálním ROI.
    """
    target_markers = detect_aruco_markers(img)
    
    # Načtení referenčních markerů z JSONu
    ref_markers = {}
    for m_id_str, corners_list in config["ref_markers"].items():
        ref_markers[int(m_id_str)] = np.array(corners_list, dtype=np.float32)
        
    H_ref2target, common_marker_count = compute_direct_homography(ref_markers, target_markers)
    
    results = []
    if H_ref2target is None:
        return results, target_markers, H_ref2target
        
    for led_cfg in config["leds"]:
        shape_type = led_cfg["shape_type"]
        ref_cx, ref_cy = led_cfg["ref_center"]
        color_prof = led_cfg.get("color_profile", {})
        dom_color = color_prof.get("dominant_channel", "red")
        
        # 1. Geometrická projekce středu a tvaru z referenčního snímku
        proj_center_tuple = transform_point((ref_cx, ref_cy), H_ref2target)
        if proj_center_tuple is None:
            continue
            
        px, py = int(round(proj_center_tuple[0])), int(round(proj_center_tuple[1]))
        
        proj_shape_geom = {}
        if shape_type == "circle":
            r_ref = led_cfg["ref_radius"]
            p_top = transform_point((ref_cx, ref_cy - r_ref), H_ref2target)
            p_right = transform_point((ref_cx + r_ref, ref_cy), H_ref2target)
            if p_top and p_right:
                r1 = np.hypot(p_top[0] - proj_center_tuple[0], p_top[1] - proj_center_tuple[1])
                r2 = np.hypot(p_right[0] - proj_center_tuple[0], p_right[1] - proj_center_tuple[1])
                proj_radius = int(round((r1 + r2) / 2.0))
            else:
                proj_radius = 15
            proj_shape_geom = {"radius": max(5, proj_radius)}
            
        elif shape_type == "rectangle":
            ref_corners = np.array(led_cfg["ref_corners"], dtype=np.float32)
            proj_corners = transform_points(ref_corners, H_ref2target)
            proj_shape_geom = {"corners": proj_corners} # 4-bodový perspektivní čtyřúhelník
            
        # 2. Optická detekce v lokálním ROI
        # Dynamický poloměr hledání ROI podle velikosti obrázku a projekce LED
        search_radius = max(35, int(min(img.shape[:2]) * 0.06))
        x_min = max(0, px - search_radius)
        y_min = max(0, py - search_radius)
        x_max = min(img.shape[1], px + search_radius)
        y_max = min(img.shape[0], py + search_radius)
        
        roi_img = img[y_min:y_max, x_min:x_max]
        refined_center = (px, py)
        is_active = False
        confidence = 0.0
        fitted_shape = None
        
        if roi_img.size > 0:
            b_roi, g_roi, r_roi = cv2.split(roi_img)
            
            if dom_color == "red":
                contrast_map = r_roi.astype(float) - (g_roi.astype(float) + b_roi.astype(float)) / 2.0
            elif dom_color == "green":
                contrast_map = g_roi.astype(float) - (r_roi.astype(float) + b_roi.astype(float)) / 2.0
            elif dom_color == "blue":
                contrast_map = b_roi.astype(float) - (r_roi.astype(float) + g_roi.astype(float)) / 2.0
            else:
                contrast_map = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY).astype(float)
                
            max_c = np.max(contrast_map)
            
            if max_c > 35:
                thresh_val = max(25.0, 0.55 * max_c)
                _, mask = cv2.threshold(contrast_map.astype(np.uint8), int(thresh_val), 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) >= 4:
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            cx_roi = M["m10"] / M["m00"]
                            cy_roi = M["m01"] / M["m00"]
                            refined_center = (int(x_min + cx_roi), int(y_min + cy_roi))
                            is_active = True
                            confidence = min(1.0, float(max_c) / 200.0)
                            
                            if shape_type == "circle":
                                (rx, ry), rad = cv2.minEnclosingCircle(c)
                                fitted_shape = {"radius": int(rad)}
                            elif shape_type == "rectangle":
                                bx, by, bw, bh = cv2.boundingRect(c)
                                fitted_shape = {"bbox": [x_min + bx, y_min + by, bw, bh]}

        results.append({
            "id": led_cfg["id"],
            "name": led_cfg["name"],
            "shape_type": shape_type,
            "projected_center": (px, py),
            "projected_shape_geom": proj_shape_geom,
            "refined_center": refined_center,
            "is_active": is_active,
            "confidence": round(confidence, 2),
            "fitted_shape": fitted_shape
        })
        
    return results, target_markers, H_ref2target

def draw_detection_overlay(img, results, detected_markers, H_ref2target, img_name=""):
    canvas = img.copy()
    h, w = canvas.shape[:2]
    
    scale = min(1280.0 / float(w), 800.0 / float(h), 1.0)
    line_thick = max(2, int(round(2 / scale)))
    font_scale = max(0.55, 0.6 / scale)
    text_thick = max(1, int(round(1.5 / scale)))
    
    # 1. Zvýraznění detekovaných ArUco markerů
    for m_id, corners in detected_markers.items():
        pts = corners.astype(int)
        cv2.polylines(canvas, [pts], True, (0, 255, 0), line_thick)
        c = pts.mean(axis=0).astype(int)
        cv2.putText(canvas, f"ID:{m_id}", (c[0]-int(20/scale), c[1]+int(8/scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), text_thick)
                    
    if H_ref2target is None:
        cv2.putText(canvas, "CHYBA: Nedostatek markerů pro homografii!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return cv2.resize(canvas, (int(w*scale), int(h*scale)))

    # 2. Vykreslení poloh LED diod SE ZACHOVÁNÍM TVARU (KRUH VS OBDÉLNÍK)
    for res in results:
        led_id = res["id"]
        shape_type = res["shape_type"]
        px, py = res["projected_center"]
        rx, ry = res["refined_center"]
        is_active = res["is_active"]
        geom_shape = res["projected_shape_geom"]
        
        color_active = (0, 255, 0) if led_id == 1 else (0, 255, 255) # Světle zelená / Žlutá
        color_proj = (255, 140, 0) # Oranžová pro projekci
        
        # A) Krok 1: Vykreslení geometrické předpovědi z markerů (Oranžově)
        if shape_type == "circle":
            r_geom = geom_shape.get("radius", int(15/scale))
            cv2.circle(canvas, (px, py), r_geom, color_proj, max(1, line_thick//2), cv2.LINE_AA)
            cv2.drawMarker(canvas, (px, py), color_proj, cv2.MARKER_TILTED_CROSS, int(14/scale), 1)
        elif shape_type == "rectangle":
            corners_geom = geom_shape.get("corners")
            if corners_geom is not None:
                pts = corners_geom.astype(int)
                cv2.polylines(canvas, [pts], True, color_proj, max(1, line_thick//2), cv2.LINE_AA)
            else:
                cv2.rectangle(canvas, (px-int(20/scale), py-int(20/scale)), (px+int(20/scale), py+int(20/scale)), color_proj, max(1, line_thick//2))
            cv2.drawMarker(canvas, (px, py), color_proj, cv2.MARKER_TILTED_CROSS, int(14/scale), 1)

        # B) Krok 2: Vykreslení opticky potvrzené nebo geometrické polohy
        if is_active:
            cv2.drawMarker(canvas, (rx, ry), color_active, cv2.MARKER_CROSS, int(22/scale), line_thick)
            
            if shape_type == "circle":
                rad = res["fitted_shape"]["radius"] if res["fitted_shape"] else r_geom
                cv2.circle(canvas, (rx, ry), max(int(10/scale), rad), color_active, line_thick)
                status_txt = f"{res['name']} (SVITI - KRUH) [{rx}, {ry}]"
            elif shape_type == "rectangle":
                if res["fitted_shape"]:
                    bx, by, bw, bh = res["fitted_shape"]["bbox"]
                    cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh), color_active, line_thick)
                else:
                    if corners_geom is not None:
                        cv2.polylines(canvas, [corners_geom.astype(int)], True, color_active, line_thick)
                    else:
                        cv2.rectangle(canvas, (rx-int(20/scale), ry-int(20/scale)), (rx+int(20/scale), ry+int(20/scale)), color_active, line_thick)
                status_txt = f"{res['name']} (SVITI - OBDELNIK) [{rx}, {ry}]"
                
            cv2.putText(canvas, status_txt, (rx + int(20/scale), ry + int(5/scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_active, text_thick)
        else:
            # VYPNUTÁ LED: Zachovat přesný tvar (obdélník nebo kruh)
            if shape_type == "circle":
                cv2.circle(canvas, (px, py), r_geom, color_proj, line_thick, cv2.LINE_AA)
                cv2.drawMarker(canvas, (px, py), color_proj, cv2.MARKER_CROSS, int(16/scale), line_thick)
                status_txt = f"{res['name']} (VYPNUTO - KRUH) [{px}, {py}]"
            elif shape_type == "rectangle":
                if corners_geom is not None:
                    cv2.polylines(canvas, [corners_geom.astype(int)], True, color_proj, line_thick, cv2.LINE_AA)
                else:
                    cv2.rectangle(canvas, (px-int(20/scale), py-int(20/scale)), (px+int(20/scale), py+int(20/scale)), color_proj, line_thick)
                cv2.drawMarker(canvas, (px, py), color_proj, cv2.MARKER_CROSS, int(16/scale), line_thick)
                status_txt = f"{res['name']} (VYPNUTO - OBDELNIK) [{px}, {py}]"
                
            cv2.putText(canvas, status_txt, (px + int(20/scale), py + int(5/scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_proj, text_thick)

    disp_canvas = cv2.resize(canvas, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    panel = disp_canvas.copy()
    cv2.rectangle(panel, (0, 0), (disp_canvas.shape[1], 50), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.8, disp_canvas, 0.2, 0, disp_canvas)
    
    cv2.putText(disp_canvas, f"Snimek: {img_name} ({w}x{h} px)", (15, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(disp_canvas, "[MEZERNIK/N] Dalsi  |  [P] Predchozi  |  [S] Ulozit snimek  |  [A] Ulozit vsechny  |  [Q] Konec",
                (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 200), 1)
                
    return disp_canvas

def run_image_sequence_detection(img_dir, config_path, output_dir="output"):
    if not os.path.exists(config_path):
        print(f"[CHYBA] Konfigurační soubor '{config_path}' neexistuje.")
        print("Spusťte nejprve anotaci pomocí: python led_detector.py --mode annotate")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    img_files = [f for f in img_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not img_files:
        print(f"[UPOZORNĚNÍ] Ve složce '{img_dir}' nebyly nalezeny žádné obrázky.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=======================================================")
    print(f"   DETEKCE POLOHY LED DIOD (Nalezeno {len(img_files)} obrázků)")
    print(f"=======================================================")
    
    current_idx = 0
    window_name = "Detekce polohy LED diod"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    all_results_export = {}

    while True:
        img_path = img_files[current_idx]
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Nelze načíst obrázek: {img_path}")
            current_idx = (current_idx + 1) % len(img_files)
            continue
            
        results, target_markers, H_ref2target = detect_and_refine_led_in_image(img, config)
        overlay = draw_detection_overlay(img, results, target_markers, H_ref2target, f"{img_name} ({current_idx+1}/{len(img_files)})")
        
        h_disp, w_disp = overlay.shape[:2]
        cv2.resizeWindow(window_name, w_disp, h_disp)
        
        print(f"[{current_idx+1}/{len(img_files)}] {img_name}: ArUco markery = {list(target_markers.keys())}")
        for r in results:
            state = f"SVÍTÍ ({r['shape_type'].upper()})" if r["is_active"] else f"GEOMETRICKÁ PROJEKCE ({r['shape_type'].upper()})"
            print(f"   -> {r['name']}: Poloha={r['refined_center']}, Tvar={r['shape_type']}, Stav={state}")

        all_results_export[img_name] = {
            "markers_detected": list(target_markers.keys()),
            "leds": [
                {
                    "name": r["name"],
                    "shape_type": r["shape_type"],
                    "projected": list(r["projected_center"]),
                    "refined": list(r["refined_center"]),
                    "is_active": r["is_active"],
                    "confidence": r["confidence"]
                } for r in results
            ]
        }
        
        cv2.imshow(window_name, overlay)
        key = cv2.waitKey(0) & 0xFF
        
        if key in (ord(' '), ord('n'), ord('N'), 83):
            current_idx = (current_idx + 1) % len(img_files)
        elif key in (ord('p'), ord('P'), 81):
            current_idx = (current_idx - 1) % len(img_files)
        elif key in (ord('s'), ord('S')):
            out_file = os.path.join(output_dir, f"det_{img_name}")
            cv2.imwrite(out_file, overlay)
            print(f"--> Uložen snímek: {out_file}")
        elif key in (ord('a'), ord('A')):
            print("Zpracovávám a ukládám všechny snímky...")
            for idx, path in enumerate(img_files):
                fname = os.path.basename(path)
                test_img = cv2.imread(path)
                if test_img is not None:
                    res, mks, H = detect_and_refine_led_in_image(test_img, config)
                    ov = draw_detection_overlay(test_img, res, mks, H, fname)
                    cv2.imwrite(os.path.join(output_dir, f"det_{fname}"), ov)
            
            summary_path = os.path.join(output_dir, "detection_summary.json")
            with open(summary_path, "w", encoding="utf-8") as sf:
                json.dump(all_results_export, sf, indent=2, ensure_ascii=False)
            print(f"--> Všechny obrázky a výstupní JSON uloženy do složky '{output_dir}'!")
        elif key in (27, ord('q'), ord('Q')):
            break
            
    cv2.destroyAllWindows()

# ==========================================
# 4. HLAVNÍ SPUŠTĚNÍ A ARGUMENTY
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Detekce polohy LED diod pomocí ArUco markerů")
    parser.add_argument("--mode", type=str, choices=["annotate", "detect", "auto"], default="auto",
                        help="Režim: 'annotate' pro označení LED, 'detect' pro detekci ve složce img, 'auto' pro automatické spuštění")
    parser.add_argument("--ref", type=str, default="reference.png", help="Cesta k referenčnímu obrázku")
    parser.add_argument("--config", type=str, default=".\img\led_config.json", help="Cesta k JSON souboru s konfigurací")
    parser.add_argument("--img_dir", type=str, default="img", help="Složka s testovacími obrázky")
    parser.add_argument("--generate", action="store_true", help="Vygenerovat syntetická testovací data (reference.png + img/*)")
    
    args = parser.parse_args()
    
    if args.generate:
        print("Generování syntetických testovacích dat...")
        import prototyping.generate_test_data as generate_test_data
        board = generate_test_data.create_synthetic_board()
        ref_img = generate_test_data.draw_leds(board, True, True)
        cv2.imwrite("reference.png", ref_img)
        os.makedirs("img", exist_ok=True)
        test_params = [
            (0.2, 0.1, 0.95, 10, 10, True, True),
            (-0.3, 0.2, 0.85, 40, 20, True, False),
            (0.4, -0.2, 1.05, -20, 0, False, True),
            (-0.2, -0.3, 0.75, 80, 50, True, True),
            (0.1, 0.4, 0.9, 0, -30, True, True),
        ]
        for idx, (ax, ay, sc, dx, dy, l1, l2) in enumerate(test_params, 1):
            scene = generate_test_data.draw_leds(board, led1_on=l1, led2_on=l2)
            warped = generate_test_data.apply_perspective(scene, ax, ay, sc, dx, dy)
            cv2.imwrite(f"img/image_{idx:02d}.png", warped)
        print("Syntetická data připravena!")

    if args.mode == "annotate":
        annotator = LEDAnnotator(args.ref)
        config = annotator.run()
        with open(args.config, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"\n[ÚSPĚCH] Konfigurace LED byla úspěšně uložena do: {args.config}")
        
    elif args.mode == "detect":
        run_image_sequence_detection(args.img_dir, args.config)
        
    elif args.mode == "auto":
        if not os.path.exists(args.config):
            if not os.path.exists(args.ref):
                print(f"[INFO] Referenční obrázek '{args.ref}' nenalezen, generuji testovací data...")
                import prototyping.generate_test_data as generate_test_data
                board = generate_test_data.create_synthetic_board()
                ref_img = generate_test_data.draw_leds(board, True, True)
                cv2.imwrite("reference.png", ref_img)
                os.makedirs("img", exist_ok=True)
                test_params = [
                    (0.2, 0.1, 0.95, 10, 10, True, True),
                    (-0.3, 0.2, 0.85, 40, 20, True, False),
                    (0.4, -0.2, 1.05, -20, 0, False, True),
                    (-0.2, -0.3, 0.75, 80, 50, True, True),
                    (0.1, 0.4, 0.9, 0, -30, True, True),
                ]
                for idx, (ax, ay, sc, dx, dy, l1, l2) in enumerate(test_params, 1):
                    scene = generate_test_data.draw_leds(board, led1_on=l1, led2_on=l2)
                    warped = generate_test_data.apply_perspective(scene, ax, ay, sc, dx, dy)
                    cv2.imwrite(f"img/image_{idx:02d}.png", warped)
            
            print(f"Spouštím anotaci na referenčním obrázku '{args.ref}'...")
            annotator = LEDAnnotator(args.ref)
            config = annotator.run()
            with open(args.config, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"[ÚSPĚCH] Konfigurace uložena do: {args.config}")
            
        run_image_sequence_detection(args.img_dir, args.config)

if __name__ == "__main__":
    main()
