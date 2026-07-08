import os
import sys
import urllib.request

def download_yolov6_weights():
    # Die offiziellen Meituan YOLOv6s Gewichte (v0.3.0 ist die stabilste Version für den Hub-Load)
    url = "https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6s.pt"
    filename = "yolov6s.pt"
    
    print(f"[INFO] Starte Download von {filename}...")
    print(f"[INFO] Quelle: {url}")
    
    # Ein einfacher Fortschritts-Melder fürs Slurm-Log, ohne den Output zuzuspammen
    def progress_callback(blocks_transferred, block_size, total_size):
        if total_size > 0:
            percent = int(blocks_transferred * block_size * 100 / total_size)
            # Nur alle 10% loggen, damit das Slurm-Log lesbar bleibt
            if percent % 10 == 0:
                sys.stdout.write(f"\r[DOWNLOAD-STATUS] {min(percent, 100)}% geladen...")
                sys.stdout.flush()

    try:
        # Download starten
        urllib.request.urlretrieve(url, filename, reporthook=progress_callback)
        print(f"\n[ERFOLG] '{filename}' wurde erfolgreich heruntergeladen!")
        print(f"[INFO] Speicherort: {os.path.abspath(filename)}")
        
    except Exception as e:
        print(f"\n[FEHLER] Download fehlgeschlagen: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_yolov6_weights()