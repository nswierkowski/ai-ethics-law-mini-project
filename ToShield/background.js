import { pipeline, env } from './lib/transformers.min.js';

env.allowLocalModels  = true;
env.allowRemoteModels = false;
env.backends.onnx.wasm.numThreads = 1;
env.localModelPath = chrome.runtime.getURL('/');
env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL('lib/');

const LABEL_MAP = {
  '0': 'Arbitration',
  '1': 'Content Removal',
  '2': 'Copyright / IP',
  '3': 'Jurisdiction',
  '4': 'Governing Law',
  '5': 'Limitation of Liability',
  '6': 'Unilateral Termination',
  '7': 'Broad Data Use',
  '8': 'Privacy Change',
  '9': 'OK / Fair',
  'LABEL_0': 'Arbitration',
  'LABEL_1': 'Content Removal',
  'LABEL_2': 'Copyright / IP',
  'LABEL_3': 'Jurisdiction',
  'LABEL_4': 'Governing Law',
  'LABEL_5': 'Limitation of Liability',
  'LABEL_6': 'Unilateral Termination',
  'LABEL_7': 'Broad Data Use',
  'LABEL_8': 'Privacy Change',
  'LABEL_9': 'OK / Fair',
};

function resolveLabel(raw) {
  return LABEL_MAP[raw] ?? LABEL_MAP[String(raw)] ?? raw;
}

const LABEL_DESCRIPTIONS = {
  'Arbitration':             "forces users into private arbitration and waives the right to sue in court",
  'Content Removal':         "allows the company to delete user content or accounts without notice or reason",
  'Copyright / IP':          "transfers or grants broad intellectual property rights away from the user",
  'Jurisdiction':            "forces legal disputes to be heard in a court chosen by the company, far from the user",
  'Governing Law':           "applies the laws of a specific state or country chosen unilaterally by the company",
  'Limitation of Liability': "severely limits the company's legal responsibility if something goes wrong",
  'Unilateral Termination':  "lets the company end the agreement at any time for any reason without warning",
  'Broad Data Use':          "allows the company to collect and use personal data in broad or vague ways",
  'Privacy Change':          "lets the company change the privacy policy at any time without notifying users",
};

let classifierPipeline = null;
let isLoading = false;

async function getClassifier() {
  if (classifierPipeline) return classifierPipeline;
  if (isLoading) {
    await new Promise(resolve => {
      const check = setInterval(() => {
        if (!isLoading) { clearInterval(check); resolve(); }
      }, 100);
    });
    return classifierPipeline;
  }

  isLoading = true;

  try {
    classifierPipeline = await pipeline(
      'text-classification',
      'model',
      {
        quantized: true,
        progress_callback: (p) => {
          if (p.status === 'progress') {
            chrome.storage.local.set({ modelLoadProgress: Math.round(p.progress) });
          }
        },
      }
    );
    chrome.storage.local.set({ modelReady: true, modelLoadProgress: 100 });
  } catch (err) {
    chrome.storage.local.set({ modelReady: false });
    throw err;
  } finally {
    isLoading = false;
  }

  return classifierPipeline;
}

async function explainWithGeminiNano(clauseText, label) {
  const description = LABEL_DESCRIPTIONS[label] ?? 'may be unfair to users';

  if (typeof ai === 'undefined' || !ai.languageModel) {
    return `This clause ${description}.`;
  }

  try {
    const capabilities = await ai.languageModel.capabilities();
    if (capabilities.available === 'no') {
      return `This clause ${description}.`;
    }

    const session = await ai.languageModel.create({
      systemPrompt:
        'You are a legal expert who explains Terms of Service clauses to ' +
        'ordinary people in plain English. Be direct and concise. ' +
        'Always write exactly ONE sentence. Start with "This clause".',
    });

    const prompt =
      `The following Terms of Service clause has been flagged as potentially ` +
      `unfair because it ${description}.\n\n` +
      `Clause: "${clauseText.slice(0, 400)}"\n\n` +
      `Explain in one sentence why an ordinary person should be concerned.`;

    const explanation = await session.prompt(prompt);
    await session.destroy();

    return explanation.trim().replace(/\.?$/, '.');
  } catch (err) {
    return `This clause ${description}.`;
  }
}

async function classifyChunk(text) {
  const classifier = await getClassifier();
  const [result] = await classifier(text, { topk: 1 });

  const label  = resolveLabel(result.label);
  const isFair = label === 'OK / Fair';

  return { label, score: result.score, isFair };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === 'CLASSIFY_CHUNK') {
    (async () => {
      try {
        const { label, score, isFair } = await classifyChunk(message.text);

        let explanation = null;
        if (!isFair) {
          explanation = await explainWithGeminiNano(message.text, label);
        }

        sendResponse({ success: true, chunkIndex: message.chunkIndex, label, score, isFair, explanation });
      } catch (err) {
        sendResponse({ success: false, error: err.message, chunkIndex: message.chunkIndex });
      }
    })();
    return true;
  }

  if (message.type === 'CHECK_MODEL_STATUS') {
    sendResponse({ ready: classifierPipeline !== null });
    return false;
  }

  if (message.type === 'PRELOAD_MODEL') {
    getClassifier()
      .then(() => sendResponse({ success: true }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    return true;
  }
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({ modelReady: false, modelLoadProgress: 0 });
});