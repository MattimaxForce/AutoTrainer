import time
import os
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from tqdm.auto import tqdm
from colorama import Fore, Style, init as colorama_init
from transformers import TrainerCallback

colorama_init(autoreset=True)


def rileva_precisione():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return device, "bf16"
    if device == "cuda":
        return device, "fp16"
    return device, "fp32"


def avvia_training(model_id, dataset_id, epochs=1, batch_size=4, precision_override=None, log_callback=None):
    """Avvia il training usando i parametri passati.

    model_id: string (es. 'distilbert-base-uncased' o 'username/model')
    dataset_id: string (es. 'imdb' o 'username/dataset')
    epochs, batch_size: numerici
    precision_override: 'fp16'|'bf16'|'fp32' or None per usare rilevamento automatico
    log_callback: funzione che riceve stringhe di log (usata da UI)
    """

    # container to hold results dir path once created so log() can write to file
    results = {"dir": None}

    def log(msg, level="info"):
        if level == "info":
            pref = Fore.CYAN + "[INFO]" + Style.RESET_ALL
        elif level == "warn":
            pref = Fore.YELLOW + "[WARN]" + Style.RESET_ALL
        elif level == "error":
            pref = Fore.RED + "[ERROR]" + Style.RESET_ALL
        else:
            pref = "[LOG]"
        out = f"{pref} {msg}"
        if log_callback:
            try:
                log_callback(out)
            except Exception:
                # best-effort: ignore callback errors
                pass
        else:
            print(out)
        # also try to append to a logfile in results dir if available
        try:
            d = results.get("dir")
            if d:
                with open(os.path.join(d, "training.log"), "a", encoding="utf-8") as fh:
                    # write plain text (no color codes)
                    fh.write(f"{msg}\n")
        except Exception:
            pass

    device, detected_precision = rileva_precisione()

    # decide final precision, honoring override but falling back if unsupported
    def choose_precision(preferred, device, detected):
        if preferred is None:
            return detected
        p = str(preferred).lower()
        if p == "bf16":
            if device == "cuda" and torch.cuda.is_bf16_supported():
                return "bf16"
            if device == "cuda":
                # bf16 not supported: try fp16
                return "fp16"
            return "fp32"
        if p == "fp16":
            if device == "cuda":
                return "fp16"
            return "fp32"
        return "fp32"

    precision = choose_precision(precision_override, device, detected_precision)
    log(f"Device rilevato: {device}, precisione suggerita: {detected_precision}. Precisione richiesta: {precision_override}. Usata: {precision}")

    log(f"Caricamento tokenizer e dataset: model={model_id}, dataset={dataset_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        log(f"Errore caricamento tokenizer: {e}", level="error")
        return

    try:
        # load_dataset may download data; log progress
        log("Scaricamento/lettura dataset da HuggingFace...")
        data = load_dataset(dataset_id)
    except Exception as e:
        log(f"Errore caricamento dataset: {e}", level="error")
        # signal abort
        log("TRAINING_ABORTED: errore caricamento dataset", level="error")
        return False

    # determine split
    split = "train" if "train" in data else list(data.keys())[0]

    # try common column names first
    common_text_names = ["text", "sentence", "review", "content", "article", "document", "text_a", "text1", "prompt"]
    common_label_names = ["label", "labels", "target", "label_ids", "label_id", "category", "response"]

    # Get first row to check structure
    try:
        first_row = data[split][0]
        # Check if data is nested under the split key
        if isinstance(first_row, dict) and split in first_row and isinstance(first_row[split], dict):
            actual_data = first_row[split]
            col_names = list(actual_data.keys())
            log(f"Rilevata struttura dati nidificata sotto chiave '{split}'")
        else:
            col_names = data[split].column_names
        log(f"Colonne disponibili dataset: {col_names}")
    except Exception as e:
        log(f"Errore nell'analisi della struttura del dataset: {e}", level="error")
        return False

    text_col = None
    label_col = None

    for name in common_text_names:
        if name in col_names:
            text_col = name
            break

    for name in common_label_names:
        if name in col_names:
            label_col = name
            break

    # heuristic scanning over first N rows
    if text_col is None or label_col is None:
        sample_count = min(100, len(data[split]))
        str_counts = {c: 0 for c in col_names}
        num_unique = {}
        for i in range(sample_count):
            row = data[split][i]
            for c in col_names:
                v = row[c]
                if isinstance(v, str):
                    str_counts[c] += 1
                # handle lists of strings
                elif isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
                    str_counts[c] += 1
                # count uniques for numeric-like fields
                try:
                    if c not in num_unique:
                        num_unique[c] = set()
                    # try to add scalar values
                    if isinstance(v, (int, float, bool)):
                        num_unique[c].add(v)
                except Exception:
                    pass

        # choose text column as the one with highest str count
        if text_col is None:
            sorted_by_str = sorted(str_counts.items(), key=lambda x: x[1], reverse=True)
            if sorted_by_str and sorted_by_str[0][1] > 0:
                text_col = sorted_by_str[0][0]

        # choose label column as numeric column with small unique set
        if label_col is None:
            candidates = []
            for c, s in num_unique.items():
                try:
                    uniq = len(s)
                except Exception:
                    uniq = 999999
                candidates.append((c, uniq))
            candidates = sorted(candidates, key=lambda x: x[1])
            if candidates and candidates[0][1] < max(50, sample_count // 2):
                label_col = candidates[0][0]

    log(f"Rilevate colonne candidate: text='{text_col}', label='{label_col}'")
    if text_col is None or label_col is None:
        log("Impossibile identificare automaticamente colonne testo/label.", level="error")
        log("Esempi prima righe del dataset per diagnostica:")
        try:
            for i in range(min(5, len(data[split]))):
                log(str(data[split][i]))
        except Exception:
            pass
        log("TRAINING_ABORTED: colonne testo/label non identificate", level="error")
        return False

    # Pre-process the dataset to handle nested structure
    if isinstance(data[split][0], dict) and split in data[split][0]:
        log("Riorganizzazione del dataset per gestire la struttura nidificata...")
        # Estrai i dati dalla struttura nidificata
        flattened_data = []
        for item in data[split]:
            if split in item and isinstance(item[split], dict):
                flattened_data.append(item[split])
        # Crea un nuovo dataset con i dati appiattiti
        from datasets import Dataset
        ds = Dataset.from_list(flattened_data)
    else:
        ds = data[split]

    # For text generation tasks, we need a different model class
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
    
    # Determine if we should use a causal LM or a seq2seq model
    # For chat/dialogue, typically use causal LM
    use_causal_lm = True  # Per dialoghi/chat meglio usare casual LM
    
    log("Preparazione dataset per text generation...")
    
    def tokenize(examples):
        # Process input texts
        inputs = examples[text_col]
        input_tokens = tokenizer(inputs, truncation=True, padding='max_length', max_length=128)
        
        # Process response texts
        responses = examples[label_col]
        response_tokens = tokenizer(responses, truncation=True, padding='max_length', max_length=256)
        
        # For causal LM, concatenate input and response with a separator
        if use_causal_lm:
            # Concatenate input and output with a separator token
            combined_inputs = [f"{inp} [SEP] {resp}" for inp, resp in zip(inputs, responses)]
            tokens = tokenizer(combined_inputs, truncation=True, padding='max_length', max_length=384)
            # Create attention mask that allows seeing everything
            tokens["labels"] = tokens["input_ids"].copy()
        else:
            # For seq2seq, keep them separate
            tokens = {
                "input_ids": input_tokens["input_ids"],
                "attention_mask": input_tokens["attention_mask"],
                "labels": response_tokens["input_ids"]
            }
        
        return tokens

    ds = ds.map(tokenize, batched=True)
    
    # Replace classification model with appropriate text generation model
    try:
        log("Scaricamento modello pretrained da HuggingFace per text generation...")
        if use_causal_lm:
            model = AutoModelForCausalLM.from_pretrained(model_id)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    except Exception as e:
        log(f"Errore caricamento modello: {e}", level="error")
        return False

    # normalizziamo le label e contiamo
    log(f"Dataset preparato: {len(ds)} esempi")

    # prepare results directory
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_root = os.path.join(os.getcwd(), "Results")
        os.makedirs(results_root, exist_ok=True)
        results_dir = os.path.join(results_root, f"run_{ts}")
        os.makedirs(results_dir, exist_ok=True)
        # expose results_dir to log() through mutable container
        results["dir"] = results_dir
        log(f"Risultati saranno salvati in: {results_dir}")
    except Exception:
        results_dir = os.path.join(os.getcwd(), "Results")
        results["dir"] = results_dir

    args = TrainingArguments(
        output_dir=results_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        fp16=(precision == "fp16"),
        bf16=(precision == "bf16"),
        logging_steps=50,
        save_strategy="epoch",
        report_to=[],
    )
    # Stima dei passi totali per produrre una percentuale anche quando max_steps non è impostato
    try:
        import math
        dataset_len = len(ds)
        grad_acc = getattr(args, "gradient_accumulation_steps", 1)
        # world_size not considered here (single-node). Stima: steps per epoch = ceil(dataset_len / batch_size / grad_acc)
        steps_per_epoch = math.ceil(dataset_len / max(1, batch_size) / max(1, grad_acc))
        estimated_total_steps = max(1, steps_per_epoch * max(1, epochs))
    except Exception:
        estimated_total_steps = None

    # Callback per inviare progresso reale al front-end via log_callback
    class ProgressCallback(TrainerCallback):
        def __init__(self, send_log, estimated_total=None):
            self.send_log = send_log
            self.estimated_total = estimated_total

        def on_step_end(self, args, state, control, **kwargs):
            try:
                if getattr(state, "max_steps", None) and state.max_steps > 0:
                    pct = int(state.global_step / max(1, state.max_steps) * 100)
                    self.send_log(f"PROGRESS:{pct}")
                elif self.estimated_total:
                    pct = int(state.global_step / max(1, self.estimated_total) * 100)
                    pct = min(100, max(0, pct))
                    self.send_log(f"PROGRESS:{pct}")
                else:
                    self.send_log(f"STEP:{state.global_step}")
            except Exception:
                pass

        def on_train_end(self, args, state, control, **kwargs):
            try:
                self.send_log("TRAINING_COMPLETED")
            except Exception:
                pass

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        callbacks=[ProgressCallback(lambda m: log_callback(m) if log_callback else None, estimated_total=estimated_total_steps)],
    )

    log("Avvio procedura di training...")
    try:
        trainer.train()
        log("Training completato.", level="info")
    except Exception as e:
        log(f"Errore durante training: {e}", level="error")
        if log_callback:
            log_callback(f"ERROR: {e}")
        return False

    # final save and report
    try:
        log(f"Salvataggio finale dei risultati in {results_dir}")
    except Exception:
        pass

    return True
