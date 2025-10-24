from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import threading
import asyncio
import webbrowser
from trainer_core import avvia_training


app = FastAPI()

# Serve static files under /static to avoid shadowing websocket and API routes
app.mount("/static", StaticFiles(directory="HTML"), name="static")

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("HTML/favicon.ico")


@app.get("/")
async def index():
    # serve the main HTML page
    return FileResponse("HTML/index.html")

# Set of connected websocket clients
clients = set()

# Training state on the app
app.state.training = False


@app.on_event("startup")
async def startup_event():
    # capture the running loop for thread-safe scheduling
    app.state.loop = asyncio.get_event_loop()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        # keep the websocket open; frontend doesn't send messages except maybe pings
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)
    except Exception:
        clients.discard(ws)


async def broadcast(message: str):
    # send message to all connected clients; drop disconnected ones
    to_remove = []
    for ws in list(clients):
        try:
            await ws.send_text(message)
        except Exception:
            to_remove.append(ws)
    for ws in to_remove:
        clients.discard(ws)


def strip_ansi(text: str) -> str:
    """Rimuove i codici ANSI di colorazione dal testo."""
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def invia_log(msg: str):
    """Called from background training thread. Schedule broadcast to websocket clients."""
    try:
        # Rimuovi i codici ANSI solo per l'interfaccia web
        clean_msg = strip_ansi(msg)
        
        loop = getattr(app.state, "loop", None)
        if loop and loop.is_running():
            # schedule the broadcast coroutine safely in the event loop
            asyncio.run_coroutine_threadsafe(broadcast(clean_msg), loop)
        # else: no running loop yet; drop message silently
    except Exception:
        # best-effort: ignore scheduling errors
        pass


@app.post("/start")
async def start(request: Request):
    payload = await request.json()
    model = payload.get("model") or payload.get("modelName")
    dataset = payload.get("dataset") or payload.get("datasetPath")
    epochs = int(payload.get("epochs", 1))
    batch_size = int(payload.get("batch_size", payload.get("batchSize", 4)))
    precision = payload.get("precision")

    if not model or not dataset:
        return JSONResponse({"error": "model and dataset are required"}, status_code=400)

    # prevent concurrent training sessions
    if getattr(app.state, "training", False):
        return JSONResponse({"error": "Training already in progress"}, status_code=409)

    def _target():
        try:
            app.state.training = True
            invia_log(f"[SYSTEM] Avvio training: model={model}, dataset={dataset}, epochs={epochs}, batch={batch_size}")
            res = avvia_training(model, dataset, epochs=epochs, batch_size=batch_size, precision_override=precision, log_callback=invia_log)
            # avvia_training returns True on successful completion, False on abort
            if res is False:
                invia_log("TRAINING_ABORTED")
        except Exception as e:
            invia_log(f"ERROR: {e}")
        finally:
            app.state.training = False
            invia_log("TRAINING_THREAD_ENDED")

    threading.Thread(target=_target, daemon=True).start()
    return JSONResponse({"message": "Training avviato", "model": model, "dataset": dataset})


def avvia_ui_web(host, port, open_browser: bool = True, start_in_thread: bool = False):
    """Start the web UI server.

    If start_in_thread is True the server is started in a daemon thread and the function returns immediately.
    If open_browser is True the default browser is opened (short delay) after the server starts.
    """
    def _maybe_open():
        if not open_browser:
            return
        url = f"http://{host}:{port}"
        try:
            webbrowser.open(url)
        except Exception:
            pass

    if start_in_thread:
        # start server in background thread
        def _run():
            threading.Timer(1.0, _maybe_open).start()
            uvicorn.run(app, host=host, port=port)

        threading.Thread(target=_run, daemon=True).start()
        return

    # blocking run
    try:
        threading.Timer(1.0, _maybe_open).start()
    except Exception:
        pass
    uvicorn.run(app, host=host, port=port)
