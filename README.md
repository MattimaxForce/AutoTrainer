AutoTrainer — istruzioni rapide

Questo progetto fornisce un sistema "Autotrain" che può essere eseguito in due modalità:

- GUI desktop con PyQt5
- Web UI servita localmente tramite FastAPI (con WebSocket per i log in tempo reale)

Requisiti

- Python 3.8+
- Dipendenze elencate in `requirements.txt` (nota: installare `torch` può richiedere tempo e spazio)

Installazione (PowerShell)

```powershell
python -m pip install -r requirements.txt
```

Avvio

Modifica `config.json` per scegliere la modalità:
- `"ui_mode": "true"` -> apre la GUI PyQt5
- `"ui_mode": "false"` -> avvia il server web (host e port configurabili)

Esegui:

```powershell
python main.py
```

Uso Web UI

- Apri il browser su `http://<web_host>:<web_port>` (es. `http://127.0.0.1:7860`)
- Inserisci `Model Name` (es. `distilbert-base-uncased` o `username/model`) e `Dataset Path` (es. `imdb` o `username/dataset`)
- Clicca `Start Training`. La pagina mostrerà i log e la barra di avanzamento via WebSocket.

Uso PyQt GUI

- Avvia con `"ui_mode": "true"` e lancia `python main.py`.
- Seleziona modello/dataset, imposta epochs e batch, e premi `Start Training`.
- I log appariranno nella finestra dell'app.

Note e limitazioni

- Il progetto invia messaggi PROGRESS tramite WebSocket per aggiornare la UI; per ora c'è una simulazione di progresso che fornisce feedback UI prima che `Trainer.train()` termini.
- Assicurati di avere accesso a internet per scaricare modelli/dataset da HuggingFace.
- Per un'integrazione più precisa della percentuale di avanzamento, si può aggiungere un `TrainerCallback` che invii lo stato reale del training — posso implementarlo su richiesta.

Supporto

Se desideri che:
- implementi il progresso reale basato su `Trainer` (callback),
- aggiunga autenticazione HuggingFace (token),
- migliori la UI (validazioni, dropdown dinamici),

dimmi quale priorità e procedo.