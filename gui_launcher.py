import sys
import threading
import time
import webbrowser
from pathlib import Path
import streamlit.web.cli as stcli

def open_browser(port=8501):
    # Wait for the local server to start before opening browser
    time.sleep(2.5)
    webbrowser.open(f"http://localhost:{port}")

def main():
    base_dir = Path(__file__).resolve().parent
    app_script = base_dir / "app.py"

    print("=" * 65)
    print("STARTING SPACE RADIATION MISSION CONTROL DASHBOARD")
    print("=" * 65)
    print(f"[*] Serving interface: {app_script}")
    print("[*] Opening http://localhost:8501 in your default browser...")

    # Spawn background thread to open browser automatically
    threading.Thread(target=open_browser, daemon=True).start()

    # Pass Streamlit startup flags
    sys.argv = [
        "streamlit",
        "run",
        str(app_script),
        "--global.developmentMode=false",
        "--server.headless=true",
        "--server.port=8501"
    ]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()