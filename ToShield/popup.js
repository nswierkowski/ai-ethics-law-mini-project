const statusDot    = document.getElementById('statusDot');
const statusLabel  = document.getElementById('statusLabel');
const statusDetail = document.getElementById('statusDetail');
const progressWrap = document.getElementById('progressWrap');
const progressFill = document.getElementById('progressFill');
const progressLabel= document.getElementById('progressLabel');
const btnAnalyze   = document.getElementById('btnAnalyze');
const btnIcon      = document.getElementById('btnIcon');
const btnText      = document.getElementById('btnText');
const spinner      = document.getElementById('spinner');
const results      = document.getElementById('results');
const unfairCount  = document.getElementById('unfairCount');
const totalCount   = document.getElementById('totalCount');

function setModelStatus(state) {
  statusDot.classList.remove('ready', 'loading', 'error');

  switch (state) {
    case 'ready':
      statusDot.classList.add('ready');
      statusLabel.textContent  = 'Model Ready';
      statusDetail.textContent = 'Local DistilBERT · ONNX INT8 quantised';
      btnAnalyze.disabled      = false;
      progressWrap.classList.remove('visible');
      break;

    case 'loading':
      statusDot.classList.add('loading');
      statusLabel.textContent  = 'Loading Model…';
      statusDetail.textContent = 'Downloading ONNX weights into memory';
      btnAnalyze.disabled      = true;
      progressWrap.classList.add('visible');
      break;

    case 'error':
      statusDot.classList.add('error');
      statusLabel.textContent  = 'Model Failed to Load';
      statusDetail.textContent = 'Check the extension console for details';
      btnAnalyze.disabled      = true;
      progressWrap.classList.remove('visible');
      break;

    default:
      statusLabel.textContent  = 'Model Not Loaded';
      statusDetail.textContent = 'Click Analyze to load and run';
      btnAnalyze.disabled      = false;
  }
}

function setAnalyzing(active) {
  if (active) {
    spinner.style.display   = 'block';
    btnIcon.style.display   = 'none';
    btnText.textContent     = 'Analyzing…';
    btnAnalyze.disabled     = true;
  } else {
    spinner.style.display   = 'none';
    btnIcon.style.display   = '';
    btnText.textContent     = 'Analyze This Page';
    btnAnalyze.disabled     = false;
  }
}

function showResults(unfair, total) {
  unfairCount.textContent = unfair;
  totalCount.textContent  = total;
  results.classList.add('visible');

  const unfairCard = document.getElementById('resultUnfair');
  unfairCard.className = `result-item ${unfair > 0 ? 'danger' : 'safe'}`;
}

function setProgress(pct) {
  progressFill.style.width   = `${pct}%`;
  progressLabel.textContent  = `Loading model…  ${pct}%`;
}

async function init() {
  chrome.runtime.sendMessage({ type: 'CHECK_MODEL_STATUS' }, response => {
    if (chrome.runtime.lastError) {
      setModelStatus('unknown');
      return;
    }
    setModelStatus(response?.ready ? 'ready' : 'unknown');
  });

  chrome.storage.local.get(['modelReady', 'modelLoadProgress'], data => {
    if (data.modelReady) {
      setModelStatus('ready');
    } else if (data.modelLoadProgress > 0 && data.modelLoadProgress < 100) {
      setModelStatus('loading');
      setProgress(data.modelLoadProgress);
    }
  });

  const pollInterval = setInterval(() => {
    chrome.storage.local.get(['modelReady', 'modelLoadProgress'], data => {
      if (data.modelReady) {
        setModelStatus('ready');
        clearInterval(pollInterval);
      } else if (data.modelLoadProgress > 0) {
        setProgress(data.modelLoadProgress);
      }
    });
  }, 500);

  window.addEventListener('unload', () => clearInterval(pollInterval));
}

init();

btnAnalyze.addEventListener('click', async () => {
  setAnalyzing(true);

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab?.id) {
      throw new Error('No active tab found.');
    }

    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files:  ['content.js'],
    });

    chrome.tabs.sendMessage(
      tab.id,
      { type: 'START_ANALYSIS' },
      response => {
        if (chrome.runtime.lastError) {
          console.warn('[ToS Shield popup] content script not ready:', chrome.runtime.lastError.message);
        }
      }
    );

    chrome.runtime.sendMessage({ type: 'PRELOAD_MODEL' });

    setTimeout(() => setAnalyzing(false), 2000);

  } catch (err) {
    console.error('[ToS Shield popup] Analyze error:', err);
    setAnalyzing(false);
  }
});

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'ANALYSIS_COMPLETE') {
    setAnalyzing(false);
    showResults(message.unfairCount, message.totalParagraphs);
  }
});
