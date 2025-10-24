// Shared functionality across pages
document.addEventListener('DOMContentLoaded', () => {
    // Initialize any shared components or functionality
    console.log('HF AutoTrain Wizard initialized');
    
    // Make sure all checkboxes and radios are properly initialized
    document.querySelectorAll('input[type="checkbox"], input[type="radio"]').forEach(input => {
        input.addEventListener('change', function() {
            // Handle any checkbox/radio changes
        });
    });
});

// Form validation function
function validateTrainingForm() {
    const modelName = document.getElementById('model-name').value.trim();
    const datasetPath = document.getElementById('dataset-path').value.trim();
    
    if (!modelName || !datasetPath) {
        alert('Please fill in both Model Name and Dataset Path');
        return false;
    }
    
    return true;
}

// API call function (placeholder - implement your actual API call)
async function startTraining(config) {
    try {
        // Call backend /start endpoint
        const response = await fetch('/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(config)
        });
        const data = await response.json();
        console.log('Training started:', data);
        return data;
    } catch (error) {
        console.error('Error starting training:', error);
        throw error;
    }
}

// Hook train button to call backend and open websocket for logs
document.getElementById('train-button').addEventListener('click', async function (e) {
    e.preventDefault();
    if (!validateTrainingForm()) return;

    const modelName = document.getElementById('model-name').value.trim();
    const datasetPath = document.getElementById('dataset-path').value.trim();
    const precision = document.querySelector('input[name="precision"]:checked')?.value || null;
    const epochs = parseInt(document.getElementById('epochs').value || '1');
    const batch_size = parseInt(document.getElementById('batch-size').value || '4');
    const grad_acc = parseInt(document.getElementById('grad-acc').value || '1');

    const config = {
        model: modelName,
        dataset: datasetPath,
        epochs: epochs,
        batch_size: batch_size,
        gradient_accumulation_steps: grad_acc,
        precision: precision
    };

    try {
        await startTraining(config);
    } catch (err) {
        alert('Errore avvio training: ' + err);
        return;
    }

    // disable button to avoid multiple starts
    const btn = document.getElementById('train-button');
    btn.disabled = true;

    // open websocket to receive logs
    const protocol = (location.protocol === 'https:') ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws`;
    const ws = new WebSocket(wsUrl);

    ws.addEventListener('open', () => {
        console.log('WebSocket connesso');
    });

    ws.addEventListener('message', (ev) => {
        const msg = ev.data;
        if (msg.startsWith('PROGRESS:')) {
            const v = parseInt(msg.split(':')[1]);
            document.getElementById('progress-bar').style.width = v + '%';
        } else if (msg === 'TRAINING_COMPLETED' || msg === 'TRAINING_THREAD_ENDED') {
            document.getElementById('log-output').innerText += '\n[TRAINING COMPLETED]\n';
            document.getElementById('progress-bar').style.width = '100%';
            // re-enable button
            btn.disabled = false;
        } else if (msg === 'TRAINING_ABORTED' || msg.startsWith('TRAINING_ABORTED')) {
            document.getElementById('log-output').innerText += '\n[TRAINING ABORTED] Check logs.\n';
            // re-enable button
            btn.disabled = false;
        } else {
            const out = document.getElementById('log-output');
            out.innerText += msg + '\n';
            out.scrollTop = out.scrollHeight;
        }
    });

    ws.addEventListener('close', () => console.log('WebSocket chiuso'));
    ws.addEventListener('error', (e) => console.error('WebSocket error', e));
});