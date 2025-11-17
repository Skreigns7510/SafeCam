const alertDiv = document.getElementById('alert');
const alertAudio = document.getElementById('alert-audio');
let soundEnabled = true;

// enable sound button isn't used here; user must click page for autoplay to work
document.addEventListener('click', () => { if (alertAudio) alertAudio.pause(); }, { once: true });

async function pollStatus() {
    try {
        const resp = await fetch('/status', { cache: 'no-store' });
        if (!resp.ok) { setTimeout(pollStatus, 800); return; }
        const data = await resp.json();
        if (data.person) {
            if (alertDiv) alertDiv.style.display = 'block';
            if (alertAudio && soundEnabled && alertAudio.paused) {
                alertAudio.play().catch(() => { /* autoplay blocked */ });
            }
            if (document.getElementById('last')) document.getElementById('last').textContent = `Last: ${data.last_capture || 'just now'}`;
        } else {
            if (alertDiv) alertDiv.style.display = 'none';
            if (alertAudio && !alertAudio.paused) {
                alertAudio.pause();
                alertAudio.currentTime = 0;
            }
        }
    } catch (e) {}
    setTimeout(pollStatus, 500);
}
pollStatus();