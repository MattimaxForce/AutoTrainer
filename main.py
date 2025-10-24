import json
import sys
from trainer_core import avvia_training
from ui_qt import avvia_ui_qt
from ui_web import avvia_ui_web

def main():
    with open("config.json", "r") as f:
        config = json.load(f)
    # Command line overrides: --web to force web UI, --qt to force PyQt UI
    arg_web = any(a in ("--web", "-w") for a in sys.argv[1:])
    arg_qt = any(a in ("--qt", "-q") for a in sys.argv[1:])

    if arg_web and arg_qt:
        # conflicting args; prefer --qt
        arg_web = False

    if arg_qt:
        avvia_ui_qt()
        return

    if arg_web:
        host = config.get("web_host", "127.0.0.1")
        port = int(config.get("web_port", 7860))
        avvia_ui_web(host, port)
        return

    # fallback to config.json
    # Backwards-compatible: if config.json has "ui_mode": "true" -> open Qt; otherwise use web
    usa_qt = str(config.get("ui_mode", "false")).lower() == "true"
    if usa_qt:
        avvia_ui_qt()
    else:
        host = config.get("web_host", "127.0.0.1")
        port = int(config.get("web_port", 7860))
        avvia_ui_web(host, port)

if __name__ == "__main__":
    main()