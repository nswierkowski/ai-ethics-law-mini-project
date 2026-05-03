# Dokumentacja procesu

Ten plik dokumentuje **jak** pracowałem/am nad mini-projektem — jakie narzędzia AI wykorzystałem, jakie prompty pisałem, jakie decyzje podjąłem i co nie zadziałało.

> **PROCESS.md jest tak samo ważny jak kod.** Prowadzący ocenia świadome korzystanie z narzędzi AI — to jest kurs o aspektach AI.

---

## Narzędzia AI

[Lista narzędzi AI użytych w projekcie]

| Narzędzie | Do czego używałem |
|-----------|-------------------|
| Claude Sonnet 4.6 | Generowanie szkieletu notebooków, wydzielanie fragmentu kodów z notebooków do folderu src |
| Gemini Pro | Pomoc w debuggingu, burza mózgów, planowanie projektu |

## Prompty

> Nie wklejaj outputu z AI — tylko prompty, które wpisywałeś/aś.

### "Generowanie kodu do pierwszego notebooka"

```
Role: You are an expert Full-Stack ML Engineer specializing in Chrome Extension development (Manifest V3) and On-Device AI.
Project Goal: Build a "ToS Shield" Chrome extension that identifies unfair clauses in Terms of Service locally.
The Tech Stack:

Model: DistilBERT fine-tuned on lex_glue (unfair_tos subset).
Optimization: Exported to ONNX and 8-bit quantized (shrunk from 260MB to ~50MB) using Hugging Face Optimum.
Inference: Running inside a Chrome Extension via Transformers.js.
Explainer: Using Chrome’s built-in LanguageModel API (Gemini Nano) to explain detected unfair clauses.
UI: Content script that scrapes text and an injected sidebar for explanations.
Key Constraints to Remember:

Security: Manifest V3 blocks remote code. All ONNX and WASM files (from ort-wasm.wasm) must be bundled locally and declared in web_accessible_resources.
Imbalance: The training data is 90% "Fair." We must downsample the Fair class in Python during Step 1.
Privacy: Everything must run 100% on-device (No external API calls). Current Objective: I want you start with analysing the dataset lex_glue (unfair_tos subset) in jupyter notebook format. Check data distribution, class inbalance, prepare EDA. Write code in format to import needed components from src directory (you can add new components just make sure to write the path to file).
```

**Kontekst:** 
Chciałem uzyskać ogólny plan działania, a także kod do analizy danych. Zależało mi na tym, aby kod był modularny i łatwy do przeniesienia do folderu src, stąd prośba o importowanie komponentów z tego katalogu.
Daje on też potrzebny kontekst dotyczący projektu, aby AI mogło lepiej dostosować swoje odpowiedzi do specyfiki zadania.

### "Generowanie kodu do drugiego notebooka"

```
Create notebooks/02_finetune_distilbert.ipynb where you finetune distilbert on this problem. Compare few approaches:

* finetune without handling imbalance
* finetune with  handling imbalance
* train on all output class
* train on just fair/unfair
Study best models on which they mostly take attention to. What was most important in input for their prediction? You can use any explanaibility tools like SHAP, Integrated Gradients, Attention Rollout. Write code in format to import needed components from src directory.
Save models at end.
```

**Kontekst:** 
Chciałem uzyskać kod do trenowania modelu, szczególnie z uwzględnieniem problemu niezbalansowania klas. Dodatkowo zależało mi na tym, aby AI zaproponowało różne metody analizy ważności tokenów, co jest kluczowe dla zrozumienia, dlaczego model podejmuje takie a nie inne decyzje. Ponownie prosiłem o modularny kod, który można łatwo przenieść do folderu src.

### "Generowanie kodu do trzeciego notebooka"

```
Next Steps → notebooks/03_onnx_export.ipynb

Export best model ({best_mc_name} -> it is baseline) to ONNX
Apply 8-bit dynamic quantization with HuggingFace Optimum
Validate ONNX model output vs PyTorch output (≤ 1e-3 tolerance)
Bundle with Transformers.js-compatible config for the Chrome Extension The notebook do not has to be long. Just reading model, onnx'ing it, bundle with transforemers.js and save it
```

**Kontekst:** 
Ponownie, chciałem uzyskać kod do eksportowania modelu do formatu ONNX, z zastosowaniem optymalizacji 8-bitowej, a także kod do walidacji, aby upewnić się, że model działa poprawnie po konwersji. Dodatkowo zależało mi na tym, aby AI przygotowało model w sposób kompatybilny z Transformers.js, co jest kluczowe dla jego późniejszego wykorzystania w Chrome Extension. Chciałem, aby notebook był zwięzły i skupiony na tych konkretnych zadaniach -> korzystałem z darmowej wersji Claude, która ma limit tokenów, więc zależało mi na tym, aby kod był jak najbardziej efektywny.

### "Generowanie kodu do wtyczki"

```
Act as an expert Full-Stack Machine Learning Engineer and Chrome Extension Developer. 
I need you to write the complete code for a Manifest V3 Chrome Extension. This extension reads Terms of Service pages, uses a local AI model to flag unfair legal clauses, and uses Chrome's built-in AI to explain why the clause is bad.
Here is the exact architecture and constraints you must follow strictly:
### 1. The File Structure
Assume I have the following directory structure:
/local-tos-analyzer/
  ├── manifest.json
  ├── background.js
  ├── content.js
  ├── popup.html
  ├── popup.js
  ├── /model/ 
  │   ├── model.onnx (My custom fine-tuned DistilBERT)
  │   ├── tokenizer.json
  │   └── config.json
  └── /lib/
      ├── transformers.min.js (Local copy)
      ├── ort-wasm-simd.wasm (Local copy)
      └── ort-wasm-simd-threaded.wasm (Local copy)
### 2. manifest.json (Strict CSP Rules)
* Must be Manifest V3.
* You MUST declare the `/model/` and `/lib/` directories under `web_accessible_resources`. If you don't do this, Chrome's Content Security Policy will block the WebAssembly files and the ONNX model from loading locally.
* Include permissions for `activeTab`, `scripting`, and whatever is necessary to communicate between content scripts and the background service worker.
### 3. background.js (The ML Worker)
* Import Transformers.js locally from `/lib/transformers.min.js`.
* Configure the environment to ONLY load local files. Set `env.allowLocalModels = true;` and `env.allowRemoteModels = false;`. Point `env.localModelPath` to the `/model/` folder and the WASM paths to `/lib/`.
* Write an async function to load the local `text-classification` pipeline.
* Write a function that uses Chrome's native Prompt API (`ai.languageModel.create()`) to generate a 1-sentence plain-English summary of a text snippet if the classification model flags it as unfair.
* Set up a message listener to receive text chunks from `content.js`, run them through the ONNX model, and return the classification (and explanation if unfair) back to the content script.
### 4. content.js (The Scraper & UI)
* Write a script that gracefully scrapes the text paragraphs (`<p>` tags) of the active webpage.
* Implement a basic "sliding window" or chunking mechanism so it doesn't send 500 paragraphs to `background.js` all at once and freeze the browser.
* When `background.js` returns an "Unfair" result, dynamically highlight that specific paragraph on the page in a light red background and inject a small warning box above it displaying the AI explanation.
### 5. popup.html & popup.js
* Create a simple, modern, dark-mode friendly popup interface. 
* It should have a large "Analyze Page" button to manually trigger the `content.js` script, and a status indicator showing if the Local AI model is loaded and ready.
Please provide the complete, ready-to-use code for `manifest.json`, `background.js`, `content.js`, `popup.html`, and `popup.js`. Ensure the code is heavily commented so I can understand the ML data flow.
```

**Kontekst:** 
Tym razem prompt jest znacznie dłuższy, ale też zadanie wymagało większego wpływu na zachowanie modelu. Chciałem, aby AI wygenerowało kompletny kod do wtyczki Chrome, z zachowaniem wszystkich wymogów Manifest V3 i specyficznych funkcjonalności związanych z lokalnym modelem AI. 

### "Debugging"

```
Okay I quantized model:
import onnxruntime as ort

# ── Load both ONNX sessions ────────────────────────────────────────────────────
sess_fp32 = ort.InferenceSession(
    onnx_fp32_path, providers=["CPUExecutionProvider"]
)
sess_int8 = ort.InferenceSession(
    onnx_int8_path, providers=["CPUExecutionProvider"]
)

# ── Test sentences covering all label types ────────────────────────────────────
TEST_SENTENCES = [
    "Any dispute shall be resolved through binding arbitration.",
    "We may remove any content at our sole discretion without notice.",
    "By uploading content you grant us a worldwide royalty-free license.",
    "Disputes must be brought exclusively in the courts of Delaware.",
    "This agreement is governed by the laws of California.",
    "We are not liable for any indirect or consequential damages.",
    "We may terminate your account at any time for any reason.",
    "Your personal data may be shared with third-party partners.",
    "We may update this privacy policy at any time without notifying you.",
    "You retain full ownership of all content you create on our platform.",
    "We will notify you 30 days before making any material changes.",
    "You may cancel your subscription at any time with no penalty.",
    "We collect only the minimum data necessary to provide the service.",
    "All user disputes will be handled through binding class arbitration.",
    "We disclaim all warranties express or implied including merchantability.",
    "Your data is encrypted in transit and at rest using AES-256.",
    "We may share your location data with advertising networks.",
    "Continued use of the service constitutes acceptance of new terms.",
    "You waive all rights to participate in any class action lawsuit.",
    "You can request deletion of your personal data at any time.",
]

# ── Run all three models ───────────────────────────────────────────────────────
def run_pytorch(texts):
    enc = tokenizer(texts, return_tensors="pt", truncation=True,
                    max_length=128, padding="max_length").to(DEVICE)

    if "token_type_ids" in enc:
      del enc["token_type_ids"]

    with torch.no_grad():
        return model(**enc).logits.cpu().numpy()

def run_onnx(session, texts):
    enc = tokenizer(texts, return_tensors="np", truncation=True,
                    max_length=128, padding="max_length")
    inputs = {
        "input_ids":      enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    return session.run(None, inputs)[0]

logits_pt    = run_pytorch(TEST_SENTENCES)
logits_fp32  = run_onnx(sess_fp32, TEST_SENTENCES)
logits_int8  = run_onnx(sess_int8, TEST_SENTENCES)

# ── Compute differences ────────────────────────────────────────────────────────
diff_fp32 = np.abs(logits_pt   - logits_fp32)
diff_int8 = np.abs(logits_pt   - logits_int8)

TOLERANCE = 1e-3
fp32_pass  = diff_fp32.max() <= TOLERANCE
int8_pass  = diff_int8.max() <= TOLERANCE * 100   # INT8 gets ×100 tolerance

print("Validation results (20 sentences):")
print(f"  FP32 ONNX  — max |Δ| = {diff_fp32.max():.2e}  {'✅ PASS' if fp32_pass  else '❌ FAIL'} (tol={TOLERANCE:.0e})")
print(f"  INT8 ONNX  — max |Δ| = {diff_int8.max():.2e}  {'✅ PASS' if int8_pass  else '❌ FAIL'} (tol={TOLERANCE*100:.0e})")

# ── Prediction agreement ───────────────────────────────────────────────────────
preds_pt   = logits_pt.argmax(axis=1)
preds_fp32 = logits_fp32.argmax(axis=1)
preds_int8 = logits_int8.argmax(axis=1)

agree_fp32 = (preds_pt == preds_fp32).mean() * 100
agree_int8 = (preds_pt == preds_int8).mean() * 100

print(f"\n  FP32 prediction agreement : {agree_fp32:.0f}%")
print(f"  INT8 prediction agreement : {agree_int8:.0f}%")
print()
for i, text in enumerate(TEST_SENTENCES):
    pt_lbl   = cfg.label_names[preds_pt[i]]
    int8_lbl = cfg.label_names[preds_int8[i]]
    match    = "✓" if preds_pt[i] == preds_int8[i] else "✗"
    print(f"  {match} PT={pt_lbl:<25s} INT8={int8_lbl:<25s}  {text[:55]}")

Validation results (20 sentences):
  FP32 ONNX  — max |Δ| = 5.25e-06  ✅ PASS (tol=1e-03)
  INT8 ONNX  — max |Δ| = 4.39e+00  ❌ FAIL (tol=1e-01)

  FP32 prediction agreement : 100%
  INT8 prediction agreement : 55%

  ✗ PT=Broad Data Use            INT8=OK / Fair                  Any dispute shall be resolved through binding arbitrati
  ✗ PT=Jurisdiction              INT8=OK / Fair                  We may remove any content at our sole discretion withou
  ✓ PT=OK / Fair                 INT8=OK / Fair                  By uploading content you grant us a worldwide royalty-f
  ✗ PT=Unilateral Termination    INT8=OK / Fair                  Disputes must be brought exclusively in the courts of D
  ✗ PT=Limitation of Liability   INT8=OK / Fair                  This agreement is governed by the laws of California.
  ✗ PT=Arbitration               INT8=OK / Fair                  We are not liable for any indirect or consequential dam
  ✗ PT=Content Removal           INT8=OK / Fair                  We may terminate your account at any time for any reaso
  ✓ PT=OK / Fair                 INT8=OK / Fair                  Your personal data may be shared with third-party partn
  ✗ PT=Copyright/IP              INT8=OK / Fair                  We may update this privacy policy at any time without n
  ✓ PT=OK / Fair                 INT8=OK / Fair                  You retain full ownership of all content you create on 
  ✓ PT=OK / Fair                 INT8=OK / Fair                  We will notify you 30 days before making any material c
  ✓ PT=OK / Fair                 INT8=OK / Fair                  You may cancel your subscription at any time with no pe
  ✓ PT=OK / Fair                 INT8=OK / Fair                  We collect only the minimum data necessary to provide t
  ✗ PT=Broad Data Use            INT8=OK / Fair                  All user disputes will be handled through binding class
  ✓ PT=OK / Fair                 INT8=OK / Fair                  We disclaim all warranties express or implied including
  ✓ PT=OK / Fair                 INT8=OK / Fair                  Your data is encrypted in transit and at rest using AES
  ✓ PT=OK / Fair                 INT8=OK / Fair                  We may share your location data with advertising networ
  ✗ PT=Governing Law             INT8=OK / Fair                  Continued use of the service constitutes acceptance of 
  ✓ PT=OK / Fair                 INT8=OK / Fair                  You waive all rights to participate in any class action
  ✓ PT=OK / Fair                 INT8=OK / Fair                  You can request deletion of your personal data at any t

I can do something to improve INT8 performance?
```

**Kontekst:** 
Liczna liczba promptów wykorzystanych do debuggingu nie została zawarta w dokumentacji, ponieważ były one bardziej szczegółowe i dotyczyły konkretnych błędów, które napotkałem podczas implementacji. Ich treść to 90% kodu, a 10% pytania o to, jak rozwiązać dany problem, więc nie wnoszą one wiele do zrozumienia ogólnego procesu pracy nad projektem.

## Decyzje

Kluczowe decyzje podjęte w trakcie pracy

1. **Sposób obsługi dysbalansu danych** — Zbiór danych zawiera 90% przykładów "Fair". Zdecydowałem się na downsampling klasy "Fair" podczas trenowania modelu, aby uniknąć problemu zdominowania przez tę klasę i poprawić zdolność modelu do wykrywania "Unfair". Aby wzmocnić klasy mniejszościowe, rozpatrzyłem, także zmianę funkcji straty na inną.
Ostatecznie użyłem obu podejść. Ponieważ nie byłem pewien, które z nich będzie lepsze, zdecydowałem się wytrenować wszystkie kombinacje modeli oraz dodać model traktujący problem jako binarny (Fair/Unfair) oraz wieloklasowy (Fair, Arbitration, etc.). Ostatecznie model binarny z downsamplingiem klasy "Fair" okazał się najlepszy, ale uznałem, że zachowanie etykiet wieloklasowych jest wartościowe, ponieważ pozwala użytkownikowi lepiej zrozumieć, jaki rodzaj niesprawiedliwości występuje w danym fragmencie umowy. 
W wynikach wieloetykietowych paradoksalnie najlepiej wypadł model baseline'owy, który nie miał żadnej specjalnej obsługi dysbalansu, ale jego jakość była i tak gorsza niż modelu binarnego z downsamplingiem.
Mimo to, z decydowałem się użyć modelu multiklasowego ze specjalną funkcją straty, ponieważ zająć drugie miejsce w rankingu, posiadając lepsze wyniki w metryce Recall dla klas mniejszościowych, co jest kluczowe w tym zadaniu, ponieważ chcemy unikać fałszywych negatywów (nie wykrycie niesprawiedliwego zapisu). 

2. **Wybór modelu** — Wiedziałem, że projekt docelową ma być wtyczką do przeglądarki, więc model musi być lekki i szybki. Zdecydowałem się na DistilBERT, ponieważ jest to mniejsza i szybsza wersja BERT-a, która nadal oferuje dobrą wydajność na zadaniach klasyfikacji tekstu. Alternatywnie rozważałem użycie Legalbert, który jest modelem specjalizowanym w języku prawniczym, ale obawiałem się, że może być zbyt duży i wolny do uruchamiania lokalnie w przeglądarce. Ostatecznie DistilBERT okazał się dobrym kompromisem między wydajnością a rozmiarem.
3. **Spadek jakości modelu po kwantyzacji** — Po zastosowaniu 8-bitowej kwantyzacji, model zaczął popełniać znacznie więcej błędów, szczególnie w klasach mniejszościowych. W finalnym projecie wciąż modelu INT8, nawet pomimo jego gorszej jakości, traktując zasobożerność, jako kluczowy parametr. W finalnym projecie pozostawiłem alternatywny model FP32, który można by wykorzystać w przyszłości, gdyby zasoby pozwalały na jego uruchomienie po stronie klienta.
## Co nie zadziałało

[Ślepe uliczki, błędy, nieudane podejścia — to jest wartościowa część dokumentacji]

1. **[Problem]** — [Co poszło nie tak? Jak to naprawiłem / obszedłem?]
2. **[Problem]** — [...]

## Iteracje

[Jak projekt ewoluował? Krótki opis kolejnych wersji / podejść]

1. **v1** — Pierwotnie planowałem użyć dwa modele w projekcie. Mały model typu encoder (Distilbert) oraz wykorzystać API Google Chrome'a, który oferuje, jako eksperymentalną opcję dostęp do małego modelu Gemini Nano, dostępnego dla wtyczek, aby tłumaczył jak dany fragment jest powiązany z etykietą dostarczoną od pierwszego modelu. To był też, główny powód robienia wtyczki konkretnie pod jedną przeglądarkę. Okazało się że opcja ta jest w tym momencie blokowana w większośći krajów europejskich z uwagi na problemy natury prawnej. Wtyczki powinny być małe i nie przeciążające przeglądarki, stąd ostatecznie pozostał jedynie model typu encoder. 
Niemniej kod do łączenia się z gemini nano jest zapisany w projekcie, ale nie używany. Potencjalnie coś co można dodać w przyszłych iteracjach.
2. **v2** — Pierwotnie wtyczka sama miała znajdować na stronie fragmenty tekstu oraz oceniać krok po kroku. Okazało się to nie możliwę z uwagi na zróżnicowaną budowę stron - często wczytywane były fragmenty niezwiązane z tekstem umowy. Stąd zostałem zmuszony, dodać opcję, która wykrywa zaznaczony fragment tekstu i to go przenosi do modelu. Jest to mniej wygodnę niż automatyczne odczytanie, ale daje większą kontrolę nad tym co model analizuje.
3. **vN** — Finalna wersja wtyczki zawiera tylko jeden model typu encoder, zminimalizowany z użyciem ONNX'a, tak aby mógłbyć wykorzystywany po stronie klienta. Wtyczka oznacza tekst stosowną etykietą oraz opisem znaczenia tej etykiety. Pozwala to użytkownikowi zoorientować się, co jest niepokojącego z wybranym fragmentem.
