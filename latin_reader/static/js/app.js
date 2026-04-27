/**
 * Latin Reader – Frontend Logic
 * Rewritten for the latin-reader architecture with /api/* endpoints.
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- DOM References ---
    const latinInput = document.getElementById("latin-input");
    const parseBtn = document.getElementById("parse-btn");
    const charCountValue = document.getElementById("char-count-value");
    const errorBanner = document.getElementById("error-banner");
    const errorMessage = document.getElementById("error-message");
    const errorClose = document.getElementById("error-close");
    const resultsSection = document.getElementById("results-section");
    const treeContent = document.getElementById("tree-content");
    const tokenTableBody = document.getElementById("token-table-body");
    const zoomInBtn = document.getElementById("zoom-in-btn");
    const zoomOutBtn = document.getElementById("zoom-out-btn");
    const zoomResetBtn = document.getElementById("zoom-reset-btn");
    const exportPdfBtn = document.getElementById("export-pdf-btn");
    const exportDocxBtn = document.getElementById("export-docx-btn");
    const depTooltip = document.getElementById("dep-tooltip");

    // --- Dependency Label Definitions (Universal Dependencies) ---
    const DEP_DEFINITIONS = {
        // Core arguments
        ROOT:       "Root — the head of the sentence; the main predicate.",
        nsubj:      "Nominal Subject — the noun or pronoun performing the verb's action.",
        "nsubj:pass": "Passive Nominal Subject — subject of a passive verb.",
        "nsubj:outer": "Outer Nominal Subject — subject of an outer clause in raising/control.",
        obj:        "Direct Object — the noun directly affected by the verb (accusative).",
        iobj:       "Indirect Object — the secondary object, often the recipient (dative).",
        csubj:      "Clausal Subject — a clause serving as the subject of the verb.",
        "csubj:pass": "Passive Clausal Subject — clausal subject of a passive verb.",
        ccomp:      "Clausal Complement — a clause acting as the object (with its own subject).",
        xcomp:      "Open Clausal Complement — a predicative complement sharing its subject with the main verb (e.g. infinitive).",

        // Non-core dependents
        obl:        "Oblique Nominal — a noun in an oblique case (ablative, dative, etc.) modifying the verb; often an adverbial use.",
        "obl:agent": "Oblique Agent — the agent in a passive construction (a/ab + ablative).",
        "obl:arg":  "Oblique Argument — an oblique nominal that is a core argument of the verb.",
        vocative:   "Vocative — a noun of direct address (vocative case).",
        expl:       "Expletive — a pronoun with no referential meaning, filling a syntactic slot.",
        dislocated: "Dislocated — a fronted or postposed element set apart from the clause.",

        // Nominal dependents
        nmod:       "Nominal Modifier — a noun modifying another noun, often in the genitive.",
        appos:      "Appositional Modifier — a noun placed beside another as explanation or renaming.",
        nummod:     "Numeric Modifier — a numeral modifying a noun.",
        amod:       "Adjectival Modifier — an adjective modifying a noun.",
        acl:        "Adnominal Clause — a clause modifying a noun (e.g. relative clause, participle).",
        "acl:relcl": "Relative Clause — a clause introduced by a relative pronoun modifying a noun.",
        det:        "Determiner — a determiner or demonstrative pronoun modifying a noun.",

        // Modifiers of clauses
        advmod:     "Adverbial Modifier — an adverb modifying a verb, adjective, or other adverb.",
        "advmod:emph": "Emphatic Adverbial Modifier — an emphatic adverb (e.g. quidem, ipse).",
        "advmod:lmod": "Locative Adverbial Modifier — an adverb indicating location.",
        advcl:      "Adverbial Clause — a subordinate clause modifying the verb (e.g. cum/ut clauses).",
        discourse:  "Discourse Element — an interjection or particle with discourse function.",

        // Function words
        aux:        "Auxiliary — an auxiliary verb (e.g. forms of 'esse' in periphrastic constructions).",
        "aux:pass": "Passive Auxiliary — auxiliary forming a passive periphrastic (esse + perfect participle).",
        cop:        "Copula — the linking verb 'esse' when used to equate subject and predicate.",
        mark:       "Marker — a subordinating conjunction introducing a dependent clause (ut, cum, quod, etc.).",
        case:       "Case Marker — a preposition or postposition governing a noun.",
        cc:         "Coordinating Conjunction — a conjunction joining parallel elements (et, -que, sed, etc.).",
        "cc:preconj": "Preconjunction — a correlative conjunction appearing before the first conjunct.",

        // Coordination & multi-word
        conj:       "Conjunct — an element joined to another by a coordinating conjunction.",
        fixed:      "Fixed Expression — part of a fixed multi-word expression (e.g. 'quam ob rem').",
        flat:       "Flat — part of a multi-word name or expression with no internal head.",
        "flat:name": "Flat Name — part of a multi-word proper name.",
        "flat:foreign": "Foreign Flat — part of a foreign-language expression kept as-is.",
        compound:   "Compound — part of a compound word or construction.",
        list:       "List — an item in a list-like structure.",
        parataxis:  "Parataxis — a clause placed side-by-side without an explicit connective.",

        // Special / technical
        orphan:     "Orphan — a dependent promoted after head ellipsis.",
        goeswith:   "Goes With — part of a word erroneously split in the source text.",
        reparandum: "Reparandum — a disfluency that is overridden or corrected.",
        punct:      "Punctuation — a punctuation mark attached to its clause.",
        dep:        "Unspecified Dependency — the relation could not be determined precisely.",
    };

    let currentZoom = 1;
    const ZOOM_STEP = 0.15;
    const ZOOM_MIN = 0.3;
    const ZOOM_MAX = 3;

    // --- Treebank Data ---
    const authorSelect = document.getElementById("author-select");
    const loadTreebankBtn = document.getElementById("load-treebank-btn");
    const treebankControls = document.getElementById("treebank-controls");
    let treebankData = null;
    let currentGoldTokens = null;

    fetch("/static/data/perseus_sentences.json")
        .then(res => res.json())
        .then(data => {
            treebankData = data;
            const authors = Object.keys(data).sort();
            authors.forEach(author => {
                const opt = document.createElement("option");
                opt.value = author;
                opt.textContent = `${author} (${data[author].length} sentences)`;
                authorSelect.appendChild(opt);
            });
            treebankControls.style.display = "flex";
        })
        .catch(err => console.error("Failed to load Perseus treebank", err));

    authorSelect.addEventListener("change", () => {
        loadTreebankBtn.disabled = !authorSelect.value;
    });

    loadTreebankBtn.addEventListener("click", () => {
        const author = authorSelect.value;
        if (author && treebankData && treebankData[author]) {
            const sentences = treebankData[author];
            const randomIndex = Math.floor(Math.random() * sentences.length);
            const sentenceObj = sentences[randomIndex];

            latinInput.value = sentenceObj.text;
            currentGoldTokens = sentenceObj.tokens;

            updateCharCount();
            latinInput.focus();
            parseText(); // Auto-parse the gold standard!
        }
    });

    // --- Character Count & Input Editing ---
    function updateCharCount() {
        charCountValue.textContent = latinInput.value.length;
    }
    latinInput.addEventListener("input", () => {
        updateCharCount();
        currentGoldTokens = null; // Clear gold override if user edits text
    });

    // --- Error Handling ---
    function showError(msg) {
        errorMessage.textContent = msg;
        errorBanner.style.display = "flex";
    }

    function hideError() {
        errorBanner.style.display = "none";
    }

    errorClose.addEventListener("click", hideError);

    // --- Zoom Controls ---
    function applyZoom() {
        treeContent.style.transform = `scale(${currentZoom})`;
    }

    zoomInBtn.addEventListener("click", () => {
        currentZoom = Math.min(currentZoom + ZOOM_STEP, ZOOM_MAX);
        applyZoom();
    });

    zoomOutBtn.addEventListener("click", () => {
        currentZoom = Math.max(currentZoom - ZOOM_STEP, ZOOM_MIN);
        applyZoom();
    });

    zoomResetBtn.addEventListener("click", () => {
        currentZoom = 1;
        applyZoom();
    });

    // --- POS Badge Class ---
    function posClass(pos) {
        const map = {
            NOUN: "noun",
            PROPN: "noun",
            VERB: "verb",
            AUX: "verb",
            ADJ: "adj",
            ADV: "adv",
            ADP: "prep",
            CCONJ: "conj",
            SCONJ: "conj",
            PRON: "pron",
            DET: "det",
        };
        return map[pos] || "other";
    }

    // --- Parse Request ---
    async function parseText() {
        const text = latinInput.value.trim();
        if (!text) {
            showError("Please enter some Latin text to parse.");
            return;
        }

        hideError();
        parseBtn.classList.add("loading");

        try {
            const payload = { text };
            if (currentGoldTokens) {
                payload.gold_tokens = currentGoldTokens;
            }

            const response = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data = await response.json();

            if (!response.ok) {
                showError(data.error || "An unexpected error occurred.");
                return;
            }

            // Phase 4: Render Sentence Map SVG instead of DisplaCy
            treeContent.innerHTML = data.chunk_svg ? data.chunk_svg : data.html;
            currentZoom = 1;
            applyZoom();
            if(!data.chunk_svg) attachSvgDepTooltips();

            // Show post-processor change indicator
            const existingBadge = document.getElementById("change-info");
            if (existingBadge) existingBadge.remove();
            if (data.change_count && data.change_count > 0) {
                const badge = document.createElement("div");
                badge.id = "change-info";
                badge.style.cssText = "padding: 8px 16px; background: rgba(110,231,183,0.08); border: 1px solid rgba(110,231,183,0.15); border-radius: 10px; font-size: 0.82rem; color: #6ee7b7; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;";
                const changeTexts = data.changes.map(c => `${c.token}: ${c.old} → ${c.new}`).join(", ");
                badge.innerHTML = `<span style="font-weight:600;">✨ ${data.change_count} correction${data.change_count !== 1 ? 's' : ''} applied</span><span style="color: var(--text-muted); font-size: 0.78rem;">${changeTexts}</span>`;
                treeContent.parentElement.insertBefore(badge, treeContent);
            }

            // Render token table
            tokenTableBody.innerHTML = "";
            data.tokens.forEach((tok, idx) => {
                const tokNum = idx + 1;
                const headId = tok.head_id || tokNum;
                const headDisplay = `${headId} (${escapeHtml(tok.head)})`;
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td style="color: var(--text-muted); font-size: 0.82rem;">${tokNum}</td>
                    <td class="token-text">${escapeHtml(tok.text)}</td>
                    <td class="token-lemma">${escapeHtml(tok.lemma)}</td>
                    <td><span class="pos-badge ${posClass(tok.pos)}">${escapeHtml(tok.pos)}</span></td>
                    <td>${escapeHtml(tok.tag)}</td>
                    <td class="dep-label" data-dep="${escapeHtml(tok.dep)}">${escapeHtml(tok.dep)}</td>
                    <td>${headDisplay}</td>
                    <td class="morph-text" title="${escapeHtml(tok.morph)}">${escapeHtml(tok.morph)}</td>
                `;
                tokenTableBody.appendChild(row);
            });

            // Show results
            resultsSection.style.display = "flex";
            resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
        } catch (err) {
            showError("Network error. Is the server running?");
            console.error(err);
        } finally {
            parseBtn.classList.remove("loading");
        }
    }

    // --- Event Listeners ---
    parseBtn.addEventListener("click", parseText);

    // --- Export Helpers ---
    async function exportFile(endpoint, filename, mimeType, btn) {
        const text = latinInput.value.trim();
        if (!text) {
            showError("No text available to export.");
            return;
        }

        btn.classList.add("loading");
        try {
            const response = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) {
                let errMsg = "Export failed.";
                try { errMsg = (await response.json()).error || errMsg; } catch {}
                throw new Error(errMsg);
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (err) {
            showError("Export error: " + err.message);
            console.error(err);
        } finally {
            btn.classList.remove("loading");
        }
    }

    exportPdfBtn.addEventListener("click", () =>
        exportFile("/api/export/pdf", "sentence_map.pdf", "application/pdf", exportPdfBtn)
    );
    exportDocxBtn.addEventListener("click", () =>
        exportFile("/api/export/docx", "sentence_map.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document", exportDocxBtn)
    );

    // Ctrl/Cmd + Enter to parse
    latinInput.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            parseText();
        }
    });

    // --- Dependency Tooltip System ---
    function showDepTooltip(label, x, y) {
        const key = label.trim();
        const definition = DEP_DEFINITIONS[key];
        if (!definition) return;

        depTooltip.innerHTML = `<strong>${escapeHtml(key)}</strong><br>${escapeHtml(definition.replace(/^[^—]+— /, ''))}`;
        depTooltip.style.display = "block";

        // Position after making visible so we can measure
        requestAnimationFrame(() => {
            const rect = depTooltip.getBoundingClientRect();
            let left = x - rect.width / 2;
            let top = y - rect.height - 12;

            // Keep within viewport
            if (left < 8) left = 8;
            if (left + rect.width > window.innerWidth - 8) left = window.innerWidth - rect.width - 8;
            if (top < 8) top = y + 20; // flip below if no room above

            depTooltip.style.left = left + "px";
            depTooltip.style.top = top + "px";
        });
    }

    function hideDepTooltip() {
        depTooltip.style.display = "none";
    }

    // Attach tooltips to displaCy SVG label elements after render
    function attachSvgDepTooltips() {
        const labels = treeContent.querySelectorAll("text.displacy-label");
        labels.forEach((label) => {
            label.style.cursor = "help";
            label.addEventListener("mouseenter", (e) => {
                const text = label.textContent;
                const bbox = label.getBoundingClientRect();
                showDepTooltip(text, bbox.left + bbox.width / 2, bbox.top);
            });
            label.addEventListener("mouseleave", hideDepTooltip);
        });
    }

    // Attach tooltips to dep cells in the token table (via event delegation)
    tokenTableBody.addEventListener("mouseenter", (e) => {
        const cell = e.target.closest(".dep-label[data-dep]");
        if (!cell) return;
        const dep = cell.dataset.dep;
        const rect = cell.getBoundingClientRect();
        showDepTooltip(dep, rect.left + rect.width / 2, rect.top);
    }, true);

    tokenTableBody.addEventListener("mouseleave", (e) => {
        const cell = e.target.closest(".dep-label[data-dep]");
        if (cell) hideDepTooltip();
    }, true);

    // --- Glossary Panel System ---
    const glossaryPanel = document.getElementById("glossary-panel");
    const glossaryClose = document.getElementById("glossary-close");
    const glossaryContent = document.getElementById("glossary-content");
    let glossaryDebounceTimer = null;
    let currentLemma = null;

    glossaryClose.addEventListener("click", () => {
        glossaryPanel.classList.remove("active");
        document.body.classList.remove("glossary-open");
    });

    async function fetchDefinition(lemma, pos="") {
        if (!lemma || lemma === currentLemma) return;
        
        currentLemma = lemma;
        glossaryPanel.classList.add("active");
        document.body.classList.add("glossary-open");
        
        glossaryContent.innerHTML = `<div style="text-align:center; padding: 20px;"><div class="spinner" style="border-color: rgba(167, 139, 250, 0.3); border-top-color: var(--accent-purple); display: inline-block;"></div></div>`;
        
        try {
            const res = await fetch(`/api/define?lemma=${encodeURIComponent(lemma)}&pos=${encodeURIComponent(pos)}`);
            const data = await res.json();
            
            if (data.error) throw new Error(data.error);
            
            if (!data.matches || data.matches.length === 0) {
                glossaryContent.innerHTML = `
                    <div style="text-align:center; color: var(--text-muted); padding: 20px;">
                        <em>No exact dictionary entry found for '${escapeHtml(lemma)}'</em>
                    </div>`;
                return;
            }
            
            let html = "";
            let first_entry = "";
            
            data.matches.forEach((match, index) => {
                if (index === 0) first_entry = match.entry;
                html += `
                    <div class="glossary-item">
                        <div class="glossary-stems">${escapeHtml(match.entry)}</div>
                        <div class="glossary-meta">${escapeHtml(match.morph_human)}</div>
                        <div class="glossary-def">${escapeHtml(match.definition)}</div>
                    </div>
                `;
            });
            
            glossaryContent.innerHTML = html;
            // Update the panel header to the first lemma matched
            const headerTitle = glossaryPanel.querySelector('.glossary-header h3');
            if (headerTitle && first_entry) {
                headerTitle.textContent = first_entry.split(',')[0];
            }
        } catch (err) {
            console.error(err);
            glossaryContent.innerHTML = `<div style="color: var(--accent-rose); padding: 20px;">Error loading meaning.</div>`;
            currentLemma = null;
        }
    }

    function attachGlossaryTooltips() {
        const tokens = treeContent.querySelectorAll(".hoverable-token");
        tokens.forEach((token) => {
            token.addEventListener("click", (e) => {
                const lemma = token.getAttribute("data-lemma");
                const pos = token.getAttribute("data-pos") || "";
                
                // Clear active states on all tokens
                tokens.forEach(t => t.classList.remove("active-token"));
                token.classList.add("active-token");
                
                // Debounce definition fetch slightly to prevent spamming
                clearTimeout(glossaryDebounceTimer);
                glossaryDebounceTimer = setTimeout(() => {
                    fetchDefinition(lemma, pos);
                }, 150);
            });
        });
    }

    // Capture parseText function to add attachGlossaryTooltips onto success
    const _originalParseText = window.parseText;
    // We already have parseText logic in DOM, modify the observer instead
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.addedNodes.length) {
                attachSvgDepTooltips();
                attachGlossaryTooltips();
            }
        });
    });
    observer.observe(treeContent, { childList: true });

    // --- Utility ---
    function escapeHtml(str) {
        if (!str) return "";
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }
});
