if (window.__tosShieldActive) {
  console.log('[ToS Shield] Already active on this page, skipping.');
} else {
  window.__tosShieldActive = true;

  const MAX_CHUNK_CHARS    = 500;
  const CHUNK_DELAY_MS     = 150;
  const MIN_PARAGRAPH_CHARS = 40;

  function scrapePageParagraphs() {
    return Array.from(document.querySelectorAll('p'))
      .map(el => ({ element: el, text: el.innerText?.trim() ?? '' }))
      .filter(({ text }) => text.length >= MIN_PARAGRAPH_CHARS);
  }

  function setScanningHighlight(element, active) {
    if (element.dataset.tosShieldFlagged) return;
    if (active) {
      element.style.outline         = '2px dashed #3b82f6';
      element.style.backgroundColor = 'rgba(59, 130, 246, 0.05)';
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } else {
      element.style.outline         = '';
      element.style.backgroundColor = '';
    }
  }

  function splitIntoChunks(text) {
    if (text.length <= MAX_CHUNK_CHARS) return [text];

    const chunks    = [];
    const sentences = text.match(/[^.!?]+[.!?]+/g) ?? [text];
    let current     = '';

    for (const sentence of sentences) {
      if ((current + sentence).length > MAX_CHUNK_CHARS && current.length > 0) {
        chunks.push(current.trim());
        current = sentence;
      } else {
        current += sentence;
      }
    }
    if (current.trim()) chunks.push(current.trim());
    return chunks;
  }

  function highlightElement(element) {
    if (element.dataset.tosShieldFlagged) return;
    element.dataset.tosShieldFlagged  = 'true';
    element.style.backgroundColor     = 'rgba(220, 38, 38, 0.08)';
    element.style.borderLeft          = '3px solid rgba(220, 38, 38, 0.7)';
    element.style.paddingLeft         = '10px';
    element.style.borderRadius        = '0 4px 4px 0';
    element.style.transition          = 'background-color 0.3s ease';
  }

  function injectWarningBox(element, label, score, explanation) {
    if (element.previousSibling?.dataset?.tosShieldWarning) return;

    const host = document.createElement('div');
    host.dataset.tosShieldWarning = 'true';
    host.style.display            = 'block';

    const shadow = host.attachShadow({ mode: 'closed' });

    shadow.innerHTML = `
      <style>
        :host { display: block; }
        .warning {
          display: flex; align-items: flex-start; gap: 10px;
          margin: 8px 0 4px; padding: 10px 14px;
          background: rgba(220, 38, 38, 0.06);
          border: 1px solid rgba(220, 38, 38, 0.35);
          border-radius: 6px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          font-size: 13px; line-height: 1.45; color: #1a1a1a;
          animation: slideIn 0.25s ease-out;
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(-6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .icon  { flex-shrink: 0; font-size: 16px; margin-top: 1px; }
        .body  { flex: 1; min-width: 0; }
        .header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
        .badge {
          display: inline-block; padding: 2px 8px;
          background: rgba(220, 38, 38, 0.12);
          border: 1px solid rgba(220, 38, 38, 0.3);
          border-radius: 100px; font-size: 11px; font-weight: 600;
          letter-spacing: 0.03em; color: #b91c1c; text-transform: uppercase;
        }
        .confidence { font-size: 11px; color: #888; }
        .explanation { color: #333; font-size: 13px; }
        .dismiss {
          flex-shrink: 0; background: none; border: none; cursor: pointer;
          color: #aaa; font-size: 16px; padding: 0; line-height: 1; transition: color 0.15s;
        }
        .dismiss:hover { color: #555; }
      </style>
      <div class="warning" role="alert">
        <span class="icon">⚠️</span>
        <div class="body">
          <div class="header">
            <span class="badge">${escapeHtml(label)}</span>
            <span class="confidence">${Math.round(score * 100)}% confidence</span>
          </div>
          <div class="explanation">${escapeHtml(explanation ?? 'This clause may be unfair.')}</div>
        </div>
        <button class="dismiss" title="Dismiss">×</button>
      </div>
    `;

    shadow.querySelector('.dismiss').addEventListener('click', () => host.remove());
    element.parentNode.insertBefore(host, element);
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  let statusOverlay = null;

  function showStatus(message, type = 'info') {
    if (!statusOverlay) {
      statusOverlay = document.createElement('div');
      Object.assign(statusOverlay.style, {
        position: 'fixed', bottom: '20px', right: '20px',
        zIndex: '2147483647',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        fontSize: '13px', padding: '10px 16px', borderRadius: '8px',
        boxShadow: '0 4px 20px rgba(0,0,0,0.18)',
        transition: 'opacity 0.3s ease', maxWidth: '300px', lineHeight: '1.4',
      });
      document.body.appendChild(statusOverlay);
    }

    const colors = {
      info:    { bg: '#1e293b', text: '#e2e8f0', border: '#334155' },
      success: { bg: '#14532d', text: '#bbf7d0', border: '#166534' },
      error:   { bg: '#450a0a', text: '#fca5a5', border: '#7f1d1d' },
    };
    const c = colors[type] ?? colors.info;

    statusOverlay.style.backgroundColor = c.bg;
    statusOverlay.style.color           = c.text;
    statusOverlay.style.border          = `1px solid ${c.border}`;
    statusOverlay.style.opacity         = '1';
    statusOverlay.textContent           = message;
  }

  function hideStatus(delay = 3000) {
    if (!statusOverlay) return;
    setTimeout(() => { if (statusOverlay) statusOverlay.style.opacity = '0'; }, delay);
  }

  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

  function classifyChunk(text, chunkIndex) {
    return new Promise((resolve, reject) => {
      chrome.runtime.sendMessage(
        { type: 'CLASSIFY_CHUNK', text, chunkIndex },
        response => {
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
          } else if (!response?.success) {
            reject(new Error(response?.error ?? 'Unknown classification error'));
          } else {
            resolve(response);
          }
        }
      );
    });
  }

  async function analyzePage() {
    let paragraphs = [];
    const selection    = window.getSelection();
    const selectedText = selection.toString().trim();

    if (selectedText.length >= MIN_PARAGRAPH_CHARS) {
      let container = selection.getRangeAt(0).commonAncestorContainer;
      let element   = container.nodeType === Node.ELEMENT_NODE ? container : container.parentElement;
      paragraphs    = [{ element, text: selectedText }];
    } else {
      paragraphs = scrapePageParagraphs();
    }

    if (paragraphs.length === 0) {
      showStatus('🛡️ No readable paragraphs found on this page.', 'info');
      hideStatus(3000);
      return;
    }

    let unfairCount = 0;
    let chunkIndex  = 0;
    let processed   = 0;

    for (const { element, text } of paragraphs) {
      const chunks         = splitIntoChunks(text);
      let paragraphFlagged = false;

      setScanningHighlight(element, true);
      const snippet = text.length > 40 ? text.substring(0, 40) + '...' : text;
      showStatus(`🔍 Scanning: "${snippet}"`);

      for (const chunk of chunks) {
        try {
          const result = await classifyChunk(chunk, chunkIndex++);

          if (!result.isFair && !paragraphFlagged) {
            paragraphFlagged = true;
            unfairCount++;
            highlightElement(element);
            injectWarningBox(element, result.label, result.score, result.explanation);
          }
        } catch (err) {
          console.warn(`[ToS Shield] chunk ${chunkIndex} error:`, err.message);
        }

        await sleep(CHUNK_DELAY_MS);
      }

      setScanningHighlight(element, false);
      processed++;
    }

    if (unfairCount === 0) {
      showStatus('✅ Analysis complete — no unfair clauses detected.', 'success');
    } else {
      showStatus(
        `⚠️ Analysis complete — ${unfairCount} potentially unfair clause${unfairCount > 1 ? 's' : ''} highlighted.`,
        'error'
      );
    }
    hideStatus(5000);

    chrome.runtime.sendMessage({
      type: 'ANALYSIS_COMPLETE',
      unfairCount,
      totalParagraphs: paragraphs.length,
    }).catch(() => {});
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === 'START_ANALYSIS') {
      analyzePage()
        .then(() => sendResponse({ success: true }))
        .catch(err => sendResponse({ success: false, error: err.message }));
      return true;
    }
  });
}