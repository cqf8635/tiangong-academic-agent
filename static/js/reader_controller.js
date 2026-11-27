/**
 * 经卷精读 · 多模态原文阅读控制器 (独立模块)
 * 支持：PDF 高 DPI 原貌展示（含扫描件/图表/公式）、鼠标拉框高清截图、Word/PPT/EPUB/MD 划词、多模态伴读与一键转入主研讨
 */
(function initReaderController() {
  const tabDialogue = document.getElementById('tab-nav-dialogue');
  const tabRadar = document.getElementById('tab-nav-radar');
  const tabReader = document.getElementById('tab-nav-reader');
  const layoutDialogue = document.getElementById('main-layout');
  const layoutRadar = document.getElementById('radar-layout');
  const layoutReader = document.getElementById('reader-layout');

  if (!tabReader || !layoutReader) return;

  const docListEl = document.getElementById('reader-doc-list');
  const tocListEl = document.getElementById('reader-toc-list');
  const btnUploadDoc = document.getElementById('btn-reader-upload');
  const uploadInput = document.getElementById('reader-upload-input');

  const btnPrev = document.getElementById('btn-reader-prev');
  const btnNext = document.getElementById('btn-reader-next');
  const pageInput = document.getElementById('reader-page-input');
  const pageTotalEl = document.getElementById('reader-page-total');
  const btnZoomOut = document.getElementById('btn-reader-zoom-out');
  const btnZoomIn = document.getElementById('btn-reader-zoom-in');
  const zoomLabelEl = document.getElementById('reader-zoom-label');
  const btnCropToggle = document.getElementById('btn-reader-crop-toggle');
  const btnToggleLeft = document.getElementById('btn-reader-toggle-left');
  const btnToggleRight = document.getElementById('btn-reader-toggle-right');
  const btnSearchToggle = document.getElementById('btn-reader-search-toggle');
  const searchBarEl = document.getElementById('reader-search-bar');
  const searchInputEl = document.getElementById('reader-search-input');
  const btnSearchRun = document.getElementById('btn-reader-search-run');
  const searchStatusEl = document.getElementById('reader-search-status');
  const btnSearchClose = document.getElementById('btn-reader-search-close');
  const searchResultsEl = document.getElementById('reader-search-results');
  const btnTextToggle = document.getElementById('btn-reader-text-toggle');
  const textDrawerEl = document.getElementById('reader-text-drawer');
  const textDrawerTitleEl = document.getElementById('reader-text-drawer-title');
  const textDrawerBodyEl = document.getElementById('reader-text-drawer-body');
  const btnCopyPage = document.getElementById('btn-reader-copy-page');
  const btnCloseTextDrawer = document.getElementById('btn-reader-close-text');
  const viewportEl = document.getElementById('reader-viewport');
  const activeDocTitleEl = document.getElementById('reader-active-doc-title');

  const ctxBoxEl = document.getElementById('reader-ai-context');
  const ctxLabelEl = document.getElementById('reader-ai-context-label');
  const ctxImgEl = document.getElementById('reader-crop-preview-img');
  const ctxQuoteEl = document.getElementById('reader-quote-preview');
  const btnClearCtx = document.getElementById('btn-reader-clear-ctx');
  const btnSendToMain = document.getElementById('btn-reader-send-main');

  const aiFeedEl = document.getElementById('reader-ai-feed');
  const stateChapterEl = document.getElementById('reader-state-chapter');
  const btnStateDrawer = document.getElementById('btn-reader-state-drawer');
  const stateDrawerEl = document.getElementById('reader-state-drawer');
  const btnStateClose = document.getElementById('btn-reader-state-close');
  const stateDrawerListEl = document.getElementById('reader-state-drawer-list');
  const aiInputEl = document.getElementById('reader-ai-input');
  const btnAiSend = document.getElementById('btn-reader-ai-send');
  const btnAiClear = document.getElementById('btn-reader-ai-clear');
  const modeSelectEl = document.getElementById('reader-model-mode');
  const btnAttachImg = document.getElementById('btn-reader-attach-img');
  const imgFileInput = document.getElementById('reader-img-input');
  const btnReaderMic = document.getElementById('btn-reader-mic');

  const state = {
    docs: [],
    filename: null,
    mode: null, // 'pdf' | 'sections'
    page: 1,
    pageCount: 1,
    pageAspect: 1.414,
    zoom: 1.0,
    cropMode: false,
    selectedText: '',
    cropImageData: '',
    history: [],
    toc: [],
    loadedOnce: false,
    pageLinesMap: new Map(),
    pdfObserver: null,
    pageAbort: null,
  };

  function escapeHtml(s) {
    return String(s || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function renderBasicMarkdown(md) {
    let src = String(md || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    const tokens = [];
    const putToken = (html) => {
      const id = `@@RTOK_${tokens.length}@@`;
      tokens.push(html);
      return id;
    };

    // 1. 先提取代码块与数学公式（在 escapeHtml 之前，避免 < > & 破坏 LaTeX）
    src = src.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_, lang, code) =>
      putToken(`<pre class="reader-code-block"><code>${escapeHtml(code.replace(/\n$/, ''))}</code></pre>`)
    );

    if (window.katex) {
      src = src.replace(/\$\$([\s\S]*?)\$\$/g, (m, f) => {
        try {
          return `\n${putToken(`<div class="reader-math-display">${window.katex.renderToString(f.trim(), { displayMode: true, throwOnError: false, output: 'html' })}</div>`)}\n`;
        } catch (_) {
          return putToken(`<pre class="reader-code-block">${escapeHtml(m)}</pre>`);
        }
      });
      src = src.replace(/\\\[([\s\S]*?)\\\]/g, (m, f) => {
        try {
          return `\n${putToken(`<div class="reader-math-display">${window.katex.renderToString(f.trim(), { displayMode: true, throwOnError: false, output: 'html' })}</div>`)}\n`;
        } catch (_) {
          return putToken(`<pre class="reader-code-block">${escapeHtml(m)}</pre>`);
        }
      });
      src = src.replace(/(?<!\\)\$([^\$\n]+?)(?<!\\)\$/g, (m, f) => {
        try {
          return putToken(window.katex.renderToString(f.trim(), { displayMode: false, throwOnError: false, output: 'html' }));
        } catch (_) {
          return escapeHtml(m);
        }
      });
      src = src.replace(/\\\((.+?)\\\)/g, (m, f) => {
        try {
          return putToken(window.katex.renderToString(f.trim(), { displayMode: false, throwOnError: false, output: 'html' }));
        } catch (_) {
          return escapeHtml(m);
        }
      });
    }

    const formatInline = (text) => {
      let s = escapeHtml(text);
      s = s.replace(/`([^`\n]+)`/g, '<code class="inline-code">$1</code>');
      s = s.replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>');
      s = s.replace(/(^|[^*])\*([^*\n]+?)\*(?!\*)/g, '$1<em>$2</em>');
      return s;
    };

    const lines = src.split('\n');
    const out = [];
    let i = 0;

    while (i < lines.length) {
      const raw = lines[i];
      const trimmed = raw.trim();

      if (!trimmed) {
        i += 1;
        continue;
      }
      if (/^@@RTOK_\d+@@$/.test(trimmed)) {
        out.push(trimmed);
        i += 1;
        continue;
      }
      if (/^[-*_]{3,}$/.test(trimmed)) {
        out.push('<hr/>');
        i += 1;
        continue;
      }
      const hMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
      if (hMatch) {
        const lvl = hMatch[1].length;
        out.push(`<h${lvl}>${formatInline(hMatch[2])}</h${lvl}>`);
        i += 1;
        continue;
      }
      if (trimmed.startsWith('>')) {
        const bqLines = [];
        while (i < lines.length && lines[i].trim().startsWith('>')) {
          bqLines.push(formatInline(lines[i].trim().replace(/^>\s?/, '')));
          i += 1;
        }
        out.push(`<blockquote>${bqLines.join('<br/>')}</blockquote>`);
        continue;
      }
      // Markdown 表格
      if (trimmed.includes('|') && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1].trim())) {
        const parseRow = (r) => r.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
        const headers = parseRow(trimmed);
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].trim().includes('|')) {
          rows.push(parseRow(lines[i]));
          i += 1;
        }
        const thead = `<thead><tr>${headers.map((h) => `<th>${formatInline(h)}</th>`).join('')}</tr></thead>`;
        const tbody = `<tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${formatInline(c)}</td>`).join('')}</tr>`).join('')}</tbody>`;
        out.push(`<div class="reader-table-wrap"><table>${thead}${tbody}</table></div>`);
        continue;
      }
      // 无序列表 (- / * / •)
      if (/^\s*[-*•]\s+/.test(raw)) {
        const items = [];
        while (i < lines.length && /^\s*[-*•]\s+/.test(lines[i])) {
          items.push(`<li>${formatInline(lines[i].replace(/^\s*[-*•]\s+/, ''))}</li>`);
          i += 1;
        }
        out.push(`<ul>${items.join('')}</ul>`);
        continue;
      }
      // 有序列表 (1. / 2.)
      if (/^\s*\d+\.\s+/.test(raw)) {
        const items = [];
        while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
          items.push(`<li>${formatInline(lines[i].replace(/^\s*\d+\.\s+/, ''))}</li>`);
          i += 1;
        }
        out.push(`<ol>${items.join('')}</ol>`);
        continue;
      }

      // 普通段落（合并连续非特殊行）
      const paraLines = [];
      while (
        i < lines.length &&
        lines[i].trim() &&
        !/^@@RTOK_\d+@@$/.test(lines[i].trim()) &&
        !/^[-*_]{3,}$/.test(lines[i].trim()) &&
        !/^#{1,4}\s+/.test(lines[i].trim()) &&
        !lines[i].trim().startsWith('>') &&
        !/^\s*[-*•]\s+/.test(lines[i]) &&
        !/^\s*\d+\.\s+/.test(lines[i])
      ) {
        paraLines.push(formatInline(lines[i].trim()));
        i += 1;
      }
      if (paraLines.length) {
        out.push(`<p>${paraLines.join('<br/>')}</p>`);
      }
    }

    let html = `<div class="reader-md">${out.join('')}</div>`;
    tokens.forEach((tok, idx) => {
      html = html.replace(`@@RTOK_${idx}@@`, tok);
    });
    return html;
  }

  function openReaderWorkspace(targetFilename = null) {
    if (tabDialogue) tabDialogue.classList.remove('active');
    if (tabRadar) tabRadar.classList.remove('active');
    tabReader.classList.add('active');
    if (layoutDialogue) layoutDialogue.style.display = 'none';
    if (layoutRadar) layoutRadar.style.display = 'none';
    layoutReader.style.display = 'grid';

    loadDocumentList(targetFilename);
  }

  window.openReaderWorkspace = openReaderWorkspace;

  // Coordinate switching back to dialogue or radar
  if (tabDialogue) {
    tabDialogue.addEventListener('click', () => {
      tabReader.classList.remove('active');
      layoutReader.style.display = 'none';
      hideSelectionPopover();
    });
  }
  if (tabRadar) {
    tabRadar.addEventListener('click', () => {
      tabReader.classList.remove('active');
      layoutReader.style.display = 'none';
      hideSelectionPopover();
    });
  }
  tabReader.addEventListener('click', () => openReaderWorkspace());

  // 划词按需翻译浮动条（用户选中段落后自主决定是否翻译、复制或发给伴读）
  let selPopoverEl = null;
  let selPopoverText = '';
  let selPopoverLastTranslation = '';
  let selPopoverTargetLang = 'zh';

  function ensureSelectionPopover() {
    if (selPopoverEl) return selPopoverEl;
    const pop = document.createElement('div');
    pop.id = 'reader-sel-popover';
    pop.className = 'reader-sel-popover';
    pop.innerHTML = `
      <div class="reader-sel-popover-bar">
        <button type="button" class="reader-sel-btn primary" data-act="translate">🌐 翻译选中</button>
        <button type="button" class="reader-sel-btn" data-act="copy">📋 复制</button>
        <button type="button" class="reader-sel-btn" data-act="ask">💬 问伴读</button>
        <button type="button" class="reader-sel-btn close-btn" data-act="close" title="关闭">✕</button>
      </div>
      <div class="reader-sel-trans-box">
        <div class="reader-sel-trans-body"></div>
        <div class="reader-sel-trans-foot">
          <span class="reader-sel-trans-meta"></span>
          <div style="display:flex;gap:5px;">
            <button type="button" class="reader-sel-btn" data-act="toggle-lang">🔄 译为英文</button>
            <button type="button" class="reader-sel-btn" data-act="copy-trans">📋 复制译文</button>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(pop);
    selPopoverEl = pop;

    const transBox = pop.querySelector('.reader-sel-trans-box');
    const transBody = pop.querySelector('.reader-sel-trans-body');
    const transMeta = pop.querySelector('.reader-sel-trans-meta');
    const toggleLangBtn = pop.querySelector('[data-act="toggle-lang"]');

    async function doTranslate(targetLang) {
      if (!selPopoverText) return;
      selPopoverTargetLang = targetLang;
      selPopoverLastTranslation = '';
      transBox.style.display = 'flex';
      transBody.innerHTML = '<span style="color:#64748b;">⏳ 正在精准翻译选中段落...</span>';
      transMeta.textContent = '';
      if (toggleLangBtn) {
        toggleLangBtn.textContent = targetLang === 'zh' ? '🔄 译为英文' : '🔄 译为中文';
      }
      clampPopoverToViewport();
      try {
        const resp = await fetch('/api/reader/translate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: selPopoverText,
            filename: state.filename || null,
            target_lang: targetLang,
          }),
        });
        const res = await resp.json();
        if (!resp.ok || res.status !== 'success') {
          throw new Error(res.detail || `HTTP ${resp.status}`);
        }
        const translated = (res.data && res.data.translation) || '';
        selPopoverLastTranslation = translated;
        transBody.innerHTML = renderBasicMarkdown(translated);
        transMeta.textContent = res.data && res.data.model ? `模型: ${res.data.model}` : '';
        clampPopoverToViewport();
      } catch (err) {
        transBody.innerHTML = `<span style="color:#b91c1c;">❌ 翻译失败: ${escapeHtml(err.message)}</span>`;
      }
    }

    pop.addEventListener('mousedown', (e) => e.stopPropagation());
    pop.addEventListener('click', async (e) => {
      const btn = e.target.closest('button[data-act]');
      if (!btn) return;
      const act = btn.getAttribute('data-act');
      if (act === 'close') {
        hideSelectionPopover();
      } else if (act === 'translate') {
        // 若原文主要是中文则默认译为英文，否则默认译为中文
        const zhChars = (selPopoverText.match(/[\u4e00-\u9fff]/g) || []).length;
        const autoLang = zhChars / Math.max(1, selPopoverText.length) > 0.35 ? 'en' : 'zh';
        await doTranslate(autoLang);
      } else if (act === 'toggle-lang') {
        await doTranslate(selPopoverTargetLang === 'zh' ? 'en' : 'zh');
      } else if (act === 'copy') {
        try {
          await navigator.clipboard.writeText(selPopoverText);
          const orig = btn.textContent;
          btn.textContent = '✅ 已复制';
          setTimeout(() => { btn.textContent = orig; }, 1200);
        } catch (_) {}
      } else if (act === 'copy-trans') {
        if (!selPopoverLastTranslation) return;
        try {
          await navigator.clipboard.writeText(selPopoverLastTranslation);
          const orig = btn.textContent;
          btn.textContent = '✅ 已复制';
          setTimeout(() => { btn.textContent = orig; }, 1200);
        } catch (_) {}
      } else if (act === 'ask') {
        state.selectedText = selPopoverText;
        updateContextPreview();
        hideSelectionPopover();
        if (aiInputEl) aiInputEl.focus();
      }
    });

    document.addEventListener('mousedown', (e) => {
      if (selPopoverEl && selPopoverEl.style.display !== 'none' && !selPopoverEl.contains(e.target)) {
        hideSelectionPopover();
      }
    });

    return pop;
  }

  function clampPopoverToViewport() {
    if (!selPopoverEl || selPopoverEl.style.display === 'none') return;
    const rect = selPopoverEl.getBoundingClientRect();
    let left = parseFloat(selPopoverEl.style.left) || 16;
    let top = parseFloat(selPopoverEl.style.top) || 16;
    if (left + rect.width > window.innerWidth - 14) {
      left = Math.max(14, window.innerWidth - rect.width - 14);
    }
    if (top + rect.height > window.innerHeight - 14) {
      top = Math.max(14, window.innerHeight - rect.height - 14);
    }
    selPopoverEl.style.left = `${Math.round(left)}px`;
    selPopoverEl.style.top = `${Math.round(top)}px`;
  }

  function showSelectionPopover(text, clientX, clientY) {
    const clean = (text || '').trim();
    if (clean.length < 2) return;
    const pop = ensureSelectionPopover();
    selPopoverText = clean;
    selPopoverLastTranslation = '';
    const transBox = pop.querySelector('.reader-sel-trans-box');
    if (transBox) transBox.style.display = 'none';
    pop.style.left = `${Math.max(14, Math.min(window.innerWidth - 260, (clientX || 200) - 40))}px`;
    pop.style.top = `${Math.max(14, (clientY || 160) + 12)}px`;
    pop.style.display = 'flex';
    clampPopoverToViewport();
  }

  function hideSelectionPopover() {
    if (selPopoverEl) selPopoverEl.style.display = 'none';
  }

  function updateContextPreview() {
    if (!ctxBoxEl) return;
    if (!state.cropImageData && !state.selectedText) {
      ctxBoxEl.style.display = 'none';
      return;
    }
    ctxBoxEl.style.display = 'flex';
    if (state.cropImageData) {
      if (ctxLabelEl) ctxLabelEl.textContent = `✂️ 已框选第 ${state.page} 页局部区域（多模态原图直传）`;
      if (ctxImgEl) {
        ctxImgEl.src = state.cropImageData;
        ctxImgEl.style.display = 'block';
      }
    } else if (ctxImgEl) {
      ctxImgEl.style.display = 'none';
      ctxImgEl.src = '';
    }
    if (state.selectedText) {
      if (!state.cropImageData && ctxLabelEl) {
        ctxLabelEl.textContent = `📝 已选中文档段落（${state.selectedText.length} 字）`;
      }
      if (ctxQuoteEl) {
        ctxQuoteEl.textContent = state.selectedText;
        ctxQuoteEl.style.display = 'block';
      }
    } else if (ctxQuoteEl) {
      ctxQuoteEl.style.display = 'none';
      ctxQuoteEl.textContent = '';
    }
  }

  function clearSelectedContext() {
    state.selectedText = '';
    state.cropImageData = '';
    updateContextPreview();
  }

  const READER_PREFS_KEY = 'tiangong_reader_progress_v1';

  function loadReaderPrefs() {
    try {
      const raw = localStorage.getItem(READER_PREFS_KEY);
      const parsed = raw ? JSON.parse(raw) : {};
      return {
        lastDoc: parsed.lastDoc || null,
        docs: parsed.docs && typeof parsed.docs === 'object' ? parsed.docs : {},
      };
    } catch (_) {
      return { lastDoc: null, docs: {} };
    }
  }

  function saveReaderPrefs(prefs) {
    try {
      localStorage.setItem(READER_PREFS_KEY, JSON.stringify(prefs));
    } catch (_) {}
  }

  function saveCurrentDocProgress() {
    if (!state.filename) return;
    const prefs = loadReaderPrefs();
    prefs.lastDoc = state.filename;
    prefs.docs[state.filename] = {
      page: state.page || 1,
      zoom: state.zoom || 1.0,
      updatedAt: Date.now(),
    };
    saveReaderPrefs(prefs);
    if (docListEl) {
      const item = Array.from(docListEl.querySelectorAll('.reader-doc-item')).find(
        (el) => el.getAttribute('data-filename') === state.filename
      );
      if (item && state.mode === 'pdf') {
        let badge = item.querySelector('.reader-doc-progress');
        if (!badge) {
          const metaEl = item.querySelector('.reader-doc-meta');
          if (metaEl) {
            badge = document.createElement('span');
            badge.className = 'reader-doc-progress';
            metaEl.appendChild(badge);
          }
        }
        if (badge) badge.textContent = `读至 P.${state.page}`;
      }
    }
  }

  function applyPanelCollapse(hideLeft, hideRight, shouldRelayout = false) {
    layoutReader.classList.toggle('hide-left', !!hideLeft);
    layoutReader.classList.toggle('hide-right', !!hideRight);
    if (btnToggleLeft) {
      btnToggleLeft.classList.toggle('active', !!hideLeft);
      btnToggleLeft.title = hideLeft ? '展开左侧书目与目录栏' : '收起左侧书目与目录栏';
    }
    if (btnToggleRight) {
      btnToggleRight.classList.toggle('active', !!hideRight);
      btnToggleRight.title = hideRight ? '展开右侧多模态伴读栏' : '收起右侧多模态伴读栏';
    }
    if (shouldRelayout && state.mode === 'pdf') {
      requestAnimationFrame(() => applyPdfZoom());
    }
  }

  // 面板宽度与折叠状态仅在本次会话生效，刷新/重启自动回到干净默认布局（阅读页码进度仍自动保留）
  // 左右分隔条支持拖拽拉伸，双击分隔条可立即恢复默认宽度
  function bindPanelResizer(handleId, side) {
    const handle = document.getElementById(handleId);
    if (!handle) return;
    let dragging = false;
    handle.addEventListener('dblclick', () => {
      if (side === 'left') {
        layoutReader.style.removeProperty('--reader-left-w');
      } else {
        layoutReader.style.removeProperty('--reader-right-w');
      }
      if (state.mode === 'pdf') {
        requestAnimationFrame(() => applyPdfZoom());
      }
    });
    handle.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      dragging = true;
      handle.classList.add('dragging');
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      e.preventDefault();
    });
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const rect = layoutReader.getBoundingClientRect();
      if (side === 'left') {
        const w = Math.round(Math.max(180, Math.min(420, e.clientX - rect.left)));
        layoutReader.style.setProperty('--reader-left-w', `${w}px`);
      } else {
        const maxRight = Math.min(960, Math.round(rect.width * 0.72));
        const w = Math.round(Math.max(300, Math.min(maxRight, rect.right - e.clientX)));
        layoutReader.style.setProperty('--reader-right-w', `${w}px`);
      }
    });
    window.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      handle.classList.remove('dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      if (state.mode === 'pdf') {
        requestAnimationFrame(() => applyPdfZoom());
      }
    });
  }
  bindPanelResizer('reader-resizer-left', 'left');
  bindPanelResizer('reader-resizer-right', 'right');

  async function loadDocumentList(autoSelectFilename = null) {
    try {
      const resp = await fetch('/api/documents');
      const data = await resp.json();
      state.docs = data.documents || [];
      if (!docListEl) return;
      if (!state.docs.length) {
        docListEl.innerHTML = '<div style="font-size:0.75rem;color:#64748b;padding:8px;">暂无本地文档，点击上方“上传”导入 PDF / Word / PPT / EPUB。</div>';
        return;
      }
      const prefs = loadReaderPrefs();
      docListEl.innerHTML = state.docs
        .map((d) => {
          const active = d.filename === state.filename ? 'active' : '';
          const mb = (d.size_bytes / (1024 * 1024)).toFixed(2);
          const saved = prefs.docs && prefs.docs[d.filename];
          const progHtml =
            saved && saved.page > 1 && String(d.type || '').toLowerCase() === 'pdf'
              ? `<span class="reader-doc-progress">读至 P.${saved.page}</span>`
              : '';
          return `
            <button type="button" class="reader-doc-item ${active}" data-filename="${escapeHtml(d.filename)}">
              <span class="reader-doc-name">📄 ${escapeHtml(d.filename)}</span>
              <span class="reader-doc-meta">${escapeHtml((d.type || '').toUpperCase())} · ${mb} MB${progHtml}</span>
            </button>
          `;
        })
        .join('');
      docListEl.querySelectorAll('.reader-doc-item').forEach((btn) => {
        btn.addEventListener('click', () => openDocument(btn.getAttribute('data-filename')));
      });

      const target =
        autoSelectFilename || (!state.filename && state.docs[0] ? state.docs[0].filename : null);
      if (target) {
        await openDocument(target);
      }
    } catch (err) {
      if (docListEl) docListEl.innerHTML = `<div style="color:#b91c1c;font-size:0.75rem;padding:8px;">加载文档列表失败: ${escapeHtml(err.message)}</div>`;
    }
  }

  async function openDocument(filename) {
    if (!filename) return;
    const prefs = loadReaderPrefs();
    const savedDoc = prefs.docs && prefs.docs[filename];
    const savedPage = savedDoc && savedDoc.page ? Math.max(1, parseInt(savedDoc.page, 10) || 1) : 1;
    const savedZoom =
      savedDoc && savedDoc.zoom ? Math.max(0.5, Math.min(2.5, parseFloat(savedDoc.zoom) || 1.0)) : 1.0;

    state.filename = filename;
    state.page = savedPage;
    state.zoom = savedZoom;
    renderDocChatHistory(filename);
    clearSelectedContext();
    hideSelectionPopover();
    if (searchResultsEl) searchResultsEl.innerHTML = '';
    if (searchStatusEl) searchStatusEl.textContent = '';
    if (activeDocTitleEl) activeDocTitleEl.textContent = filename;
    if (docListEl) {
      docListEl.querySelectorAll('.reader-doc-item').forEach((el) => {
        el.classList.toggle('active', el.getAttribute('data-filename') === filename);
      });
    }
    if (viewportEl) viewportEl.innerHTML = '<div class="reader-empty-hint">⏳ 正在加载高清文档页面...</div>';

    try {
      const resp = await fetch(`/api/reader/meta?filename=${encodeURIComponent(filename)}`);
      const meta = await resp.json();
      if (!resp.ok) throw new Error(meta.detail || `HTTP ${resp.status}`);
      state.mode = meta.mode;

      if (meta.mode === 'pdf') {
        state.pageCount = meta.page_count || 1;
        state.pageAspect = meta.page_aspect || 1.414;
        state.toc = Array.isArray(meta.toc) ? meta.toc : [];
        const resumePage = Math.max(1, Math.min(state.pageCount, savedPage));
        if (pageTotalEl) pageTotalEl.textContent = `/ ${state.pageCount} 页`;
        renderPdfToc(state.toc);
        buildPdfContinuousView(state.pageCount);
        if (resumePage > 1) {
          requestAnimationFrame(() => scrollToPdfPage(resumePage, false));
        }
        saveCurrentDocProgress();
        refreshReaderStateUI();
      } else {
        if (state.pdfObserver) {
          state.pdfObserver.disconnect();
          state.pdfObserver = null;
        }
        state.pageCount = 1;
        state.toc = [];
        if (pageTotalEl) pageTotalEl.textContent = `共 ${meta.section_count || 0} 节`;
        if (pageInput) pageInput.value = '1';
        renderSectionsView(meta.sections || []);
        saveCurrentDocProgress();
        refreshReaderStateUI();
      }
    } catch (err) {
      if (viewportEl) viewportEl.innerHTML = `<div class="reader-empty-hint" style="color:#b91c1c;">❌ 打开文档失败：${escapeHtml(err.message)}</div>`;
    }
  }

  function renderPdfToc(toc) {
    if (!tocListEl) return;
    if (!toc.length) {
      tocListEl.innerHTML = '<div style="font-size:0.74rem;color:#64748b;padding:8px;">该 PDF 未包含书签目录，可使用上方页码跳转。</div>';
      return;
    }
    tocListEl.innerHTML = toc
      .map((item) => {
        const pad = Math.min(3, Math.max(0, (item.level || 1) - 1)) * 12;
        return `<button type="button" class="reader-toc-item" data-page="${item.page}" style="padding-left:${8 + pad}px;">${escapeHtml(item.title)} <span style="color:#94a3b8;">P.${item.page}</span></button>`;
      })
      .join('');
    tocListEl.querySelectorAll('.reader-toc-item').forEach((btn) => {
      btn.addEventListener('click', () => scrollToPdfPage(parseInt(btn.getAttribute('data-page'), 10) || 1, true));
    });
  }

  function getPdfPagePixelWidth() {
    const availW = (viewportEl ? viewportEl.clientWidth : 900) - 52;
    const baseW = Math.max(420, Math.min(availW, 860));
    return Math.round(baseW * state.zoom);
  }

  function layoutPdfTextLayer(stage) {
    if (!stage) return;
    const pageNum = parseInt(stage.dataset.page || '0', 10);
    const lines = state.pageLinesMap.get(pageNum);
    const img = stage.querySelector('.reader-pdf-img');
    const textLayer = stage.querySelector('.reader-pdf-text-layer');
    if (!img || !textLayer || !lines || !lines.length) return;
    const W = img.clientWidth;
    const H = img.clientHeight;
    if (!W || !H) return;

    const frag = document.createDocumentFragment();
    lines.forEach((ln) => {
      const div = document.createElement('div');
      div.className = 'reader-pdf-text-line';
      div.textContent = ln.t;
      div.style.left = `${ln.x}%`;
      div.style.top = `${ln.y}%`;
      const targetH = (ln.h / 100) * H;
      div.style.fontSize = `${Math.max(6, targetH * 0.88)}px`;
      div.style.height = `${targetH}px`;
      div.style.lineHeight = `${targetH}px`;
      div.dataset.targetW = String((ln.w / 100) * W);
      frag.appendChild(div);
    });
    textLayer.innerHTML = '';
    textLayer.appendChild(frag);

    const children = textLayer.children;
    for (let i = 0; i < children.length; i += 1) {
      const el = children[i];
      const targetW = parseFloat(el.dataset.targetW || '0');
      const actualW = el.getBoundingClientRect().width;
      if (targetW > 0 && actualW > 0) {
        const sx = targetW / actualW;
        if (sx > 0.2 && sx < 4.0) {
          el.style.transform = `scaleX(${sx.toFixed(4)})`;
        }
      }
    }
  }

  function ensurePdfPageLoaded(stage, sig) {
    if (!stage || stage.dataset.loaded === '1' || !state.filename) return;
    stage.dataset.loaded = '1';
    const p = parseInt(stage.dataset.page || '1', 10);
    const img = stage.querySelector('.reader-pdf-img');
    const imgUrl = `/api/reader/pdf_page?filename=${encodeURIComponent(state.filename)}&page=${p}&scale=2.2`;

    img.onload = () => {
      stage.style.minHeight = '';
      layoutPdfTextLayer(stage);
    };
    img.src = imgUrl;

    fetch(`/api/reader/pdf_page_text?filename=${encodeURIComponent(state.filename)}&page=${p}`, { signal: sig })
      .then((r) => r.json())
      .then((data) => {
        if (sig && sig.aborted) return;
        const lines = Array.isArray(data.lines) ? data.lines : [];
        state.pageLinesMap.set(p, lines);
        if (img.complete && img.clientWidth > 0) {
          layoutPdfTextLayer(stage);
        }
        if (p === state.page && textDrawerEl && textDrawerEl.style.display !== 'none' && textDrawerBodyEl) {
          if (textDrawerTitleEl) textDrawerTitleEl.textContent = `📄 第 ${p} 页可复制文本（可直接划选复制或提问）`;
          const txt = (data.text || '').trim();
          textDrawerBodyEl.textContent = txt || '（本页为纯扫描图片或无内嵌文本层，请使用上方「✂️ 框选区域问 AI」直接多模态识别）';
        }
      })
      .catch(() => {});
  }

  function scrollToPdfPage(pageNum, smooth = true) {
    if (!state.filename || state.mode !== 'pdf') return;
    const p = Math.max(1, Math.min(state.pageCount, pageNum));
    state.page = p;
    if (pageInput) pageInput.value = String(p);
    const stage = viewportEl ? viewportEl.querySelector(`.reader-pdf-stage[data-page="${p}"]`) : null;
    if (stage) {
      ensurePdfPageLoaded(stage, state.pageAbort ? state.pageAbort.signal : undefined);
      stage.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'start' });
    }
    saveCurrentDocProgress();
    refreshReaderStateUI();
    if (textDrawerEl && textDrawerEl.style.display !== 'none') {
      loadCurrentPdfPageText();
    }
  }

  function renderPdfPage(pageNum) {
    scrollToPdfPage(pageNum, true);
  }

  function buildPdfContinuousView(pageCount) {
    if (!viewportEl) return;
    if (state.pageAbort) state.pageAbort.abort();
    if (state.pdfObserver) {
      state.pdfObserver.disconnect();
      state.pdfObserver = null;
    }
    const abortCtrl = new AbortController();
    state.pageAbort = abortCtrl;
    const sig = abortCtrl.signal;

    state.page = 1;
    state.pageLinesMap.clear();
    if (pageInput) pageInput.value = '1';
    if (zoomLabelEl) zoomLabelEl.textContent = `${Math.round(state.zoom * 100)}%`;

    const pageW = getPdfPagePixelWidth();
    const estH = Math.round(pageW * (state.pageAspect || 1.414));

    const container = document.createElement('div');
    container.id = 'reader-pdf-scroll-container';
    container.className = `reader-pdf-scroll-container ${state.cropMode ? 'crop-mode' : ''}`;

    const frag = document.createDocumentFragment();
    for (let p = 1; p <= pageCount; p += 1) {
      const stage = document.createElement('div');
      stage.className = 'reader-pdf-stage';
      stage.dataset.page = String(p);
      stage.style.width = `${pageW}px`;
      stage.style.minHeight = `${estH}px`;
      stage.innerHTML = `
        <div class="reader-pdf-page-badge">P.${p}</div>
        <img class="reader-pdf-img" data-page="${p}" alt="第 ${p} 页" />
        <div class="reader-pdf-text-layer" data-page="${p}"></div>
        <div class="reader-crop-rect"></div>
      `;
      frag.appendChild(stage);
    }
    container.appendChild(frag);
    viewportEl.innerHTML = '';
    viewportEl.appendChild(container);
    viewportEl.scrollTop = 0;

    // IntersectionObserver 懒加载可视区域及上下各 ~1.5 页，391 页长书也能流畅连续滚动
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            ensurePdfPageLoaded(entry.target, sig);
          }
        });
      },
      { root: viewportEl, rootMargin: '900px 0px 900px 0px', threshold: 0.01 }
    );
    state.pdfObserver = observer;
    container.querySelectorAll('.reader-pdf-stage').forEach((st) => observer.observe(st));

    // 滚动时实时同步顶部页码
    let scrollTick = false;
    viewportEl.addEventListener(
      'scroll',
      () => {
        if (scrollTick) return;
        scrollTick = true;
        requestAnimationFrame(() => {
          scrollTick = false;
          const vpRect = viewportEl.getBoundingClientRect();
          const anchorY = vpRect.top + vpRect.height * 0.28;
          const stages = container.querySelectorAll('.reader-pdf-stage');
          let bestPage = state.page;
          let bestDist = Infinity;
          for (let i = 0; i < stages.length; i += 1) {
            const r = stages[i].getBoundingClientRect();
            if (r.bottom < vpRect.top - 200) continue;
            if (r.top > vpRect.bottom + 200) break;
            const dist = Math.abs(r.top - anchorY);
            if (dist < bestDist) {
              bestDist = dist;
              bestPage = parseInt(stages[i].dataset.page || '1', 10);
            }
          }
          if (bestPage !== state.page) {
            state.page = bestPage;
            if (pageInput && document.activeElement !== pageInput) {
              pageInput.value = String(bestPage);
            }
            saveCurrentDocProgress();
            refreshReaderStateUI();
            if (textDrawerEl && textDrawerEl.style.display !== 'none') {
              loadCurrentPdfPageText();
            }
          }
        });
      },
      { signal: sig, passive: true }
    );

    // 直接在连续页面上拖选文字复制，弹出按需翻译浮动条并同步到右侧伴读助手
    container.addEventListener(
      'mouseup',
      (e) => {
        if (state.cropMode) return;
        const sel = (window.getSelection ? window.getSelection().toString() : '').trim();
        if (sel.length >= 2) {
          state.selectedText = sel;
          updateContextPreview();
          showSelectionPopover(sel, e.clientX, e.clientY);
        }
      },
      { signal: sig }
    );

    // 跨页委托拉框高清截图交互
    let dragging = false;
    let activeStage = null;
    let activeImg = null;
    let activeRect = null;
    let startX = 0;
    let startY = 0;

    container.addEventListener(
      'mousedown',
      (e) => {
        if (!state.cropMode || e.button !== 0) return;
        const stage = e.target.closest('.reader-pdf-stage');
        if (!stage) return;
        const img = stage.querySelector('.reader-pdf-img');
        const rectEl = stage.querySelector('.reader-crop-rect');
        if (!img || !rectEl) return;
        const bounds = stage.getBoundingClientRect();
        dragging = true;
        activeStage = stage;
        activeImg = img;
        activeRect = rectEl;
        state.page = parseInt(stage.dataset.page || String(state.page), 10);
        if (pageInput) pageInput.value = String(state.page);
        startX = e.clientX - bounds.left;
        startY = e.clientY - bounds.top;
        rectEl.style.left = `${startX}px`;
        rectEl.style.top = `${startY}px`;
        rectEl.style.width = '0px';
        rectEl.style.height = '0px';
        rectEl.style.display = 'block';
      },
      { signal: sig }
    );

    window.addEventListener(
      'mousemove',
      (e) => {
        if (!dragging || !activeStage || !activeRect) return;
        const bounds = activeStage.getBoundingClientRect();
        const curX = Math.max(0, Math.min(bounds.width, e.clientX - bounds.left));
        const curY = Math.max(0, Math.min(bounds.height, e.clientY - bounds.top));
        activeRect.style.left = `${Math.min(startX, curX)}px`;
        activeRect.style.top = `${Math.min(startY, curY)}px`;
        activeRect.style.width = `${Math.abs(curX - startX)}px`;
        activeRect.style.height = `${Math.abs(curY - startY)}px`;
      },
      { signal: sig }
    );

    window.addEventListener(
      'mouseup',
      () => {
        if (!dragging || !activeImg || !activeRect) return;
        dragging = false;
        const x = parseFloat(activeRect.style.left) || 0;
        const y = parseFloat(activeRect.style.top) || 0;
        const w = parseFloat(activeRect.style.width) || 0;
        const h = parseFloat(activeRect.style.height) || 0;
        activeRect.style.display = 'none';
        const img = activeImg;
        activeStage = null;
        activeImg = null;
        activeRect = null;
        if (w < 16 || h < 16 || !img.naturalWidth) return;

        const scaleX = img.naturalWidth / img.clientWidth;
        const scaleY = img.naturalHeight / img.clientHeight;
        const sx = Math.max(0, Math.round(x * scaleX));
        const sy = Math.max(0, Math.round(y * scaleY));
        const sw = Math.min(img.naturalWidth - sx, Math.round(w * scaleX));
        const sh = Math.min(img.naturalHeight - sy, Math.round(h * scaleY));
        const canvas = document.createElement('canvas');
        canvas.width = sw;
        canvas.height = sh;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
        state.cropImageData = canvas.toDataURL('image/png');
        updateContextPreview();
        if (aiInputEl) aiInputEl.focus();
      },
      { signal: sig }
    );
  }

  function applyPdfZoom() {
    if (state.mode !== 'pdf' || !viewportEl) return;
    const pageW = getPdfPagePixelWidth();
    const estH = Math.round(pageW * (state.pageAspect || 1.414));
    const curPage = state.page;
    viewportEl.querySelectorAll('.reader-pdf-stage').forEach((st) => {
      st.style.width = `${pageW}px`;
      if (st.dataset.loaded !== '1') {
        st.style.minHeight = `${estH}px`;
      } else {
        layoutPdfTextLayer(st);
      }
    });
    if (zoomLabelEl) zoomLabelEl.textContent = `${Math.round(state.zoom * 100)}%`;
    scrollToPdfPage(curPage, false);
  }

  function renderSectionsView(sections) {
    if (tocListEl) {
      tocListEl.innerHTML = sections
        .map((s) => `<button type="button" class="reader-toc-item" data-sec="${s.index}">${escapeHtml(s.location_label)}</button>`)
        .join('');
      tocListEl.querySelectorAll('.reader-toc-item').forEach((btn) => {
        btn.addEventListener('click', () => {
          const target = document.getElementById(`reader-sec-${btn.getAttribute('data-sec')}`);
          if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
      });
    }
    viewportEl.innerHTML = `
      <div class="reader-sections-stage" id="reader-sections-stage">
        ${sections
          .map(
            (s) => `
          <div class="reader-section-card" id="reader-sec-${s.index}">
            <div class="reader-section-head">
              <span>${escapeHtml(s.location_label)}</span>
              <button type="button" class="reader-chip-btn btn-ask-sec" data-sec="${s.index}">💬 选中此段问 AI</button>
            </div>
            <div class="reader-section-body">${escapeHtml(s.text)}</div>
          </div>`
          )
          .join('')}
      </div>
    `;
    const secMap = new Map(sections.map((s) => [String(s.index), s.text]));
    viewportEl.querySelectorAll('.btn-ask-sec').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.selectedText = secMap.get(btn.getAttribute('data-sec')) || '';
        state.cropImageData = '';
        updateContextPreview();
        if (aiInputEl) aiInputEl.focus();
      });
    });
    // 支持直接鼠标划词选中文本（弹出按需翻译浮动条）
    const stage = document.getElementById('reader-sections-stage');
    if (stage) {
      stage.addEventListener('mouseup', (e) => {
        const sel = (window.getSelection ? window.getSelection().toString() : '').trim();
        if (sel.length >= 2) {
          state.selectedText = sel;
          updateContextPreview();
          showSelectionPopover(sel, e.clientX, e.clientY);
        }
      });
    }
  }

  const READER_CHAT_PREFIX = 'tiangong_reader_chat_v1::';
  const READER_STATE_PREFIX = 'tiangong_reader_state_v1::';

  function loadDocReaderState(filename) {
    if (!filename) return { notes: [], lastAskedPage: null, lastAskedChapter: '' };
    try {
      const raw = localStorage.getItem(READER_STATE_PREFIX + filename);
      const parsed = raw ? JSON.parse(raw) : {};
      return {
        notes: Array.isArray(parsed.notes) ? parsed.notes : [],
        lastAskedPage: parsed.lastAskedPage || null,
        lastAskedChapter: parsed.lastAskedChapter || '',
      };
    } catch (_) {
      return { notes: [], lastAskedPage: null, lastAskedChapter: '' };
    }
  }

  function saveDocReaderState(filename, docState) {
    if (!filename) return;
    try {
      localStorage.setItem(
        READER_STATE_PREFIX + filename,
        JSON.stringify({
          notes: (docState.notes || []).slice(-30),
          lastAskedPage: docState.lastAskedPage || null,
          lastAskedChapter: docState.lastAskedChapter || '',
        })
      );
    } catch (_) {}
  }

  function computeChapterPath(pageNum) {
    if (state.mode !== 'pdf' || !state.toc || !state.toc.length) return '';
    const stack = [];
    for (let i = 0; i < state.toc.length; i += 1) {
      const item = state.toc[i];
      if ((item.page || 1) > pageNum) break;
      const lvl = item.level || 1;
      while (stack.length && stack[stack.length - 1].level >= lvl) {
        stack.pop();
      }
      stack.push({ level: lvl, title: item.title });
    }
    return stack.map((s) => s.title).join(' › ');
  }

  function refreshReaderStateUI() {
    const chPath = computeChapterPath(state.page);
    if (stateChapterEl) {
      if (chPath) {
        stateChapterEl.textContent = `📖 ${chPath}`;
        stateChapterEl.title = `当前章节路径：${chPath}`;
      } else if (state.filename) {
        stateChapterEl.textContent = state.mode === 'pdf' ? `📖 第 ${state.page} 页` : `📖 ${state.filename}`;
        stateChapterEl.title = state.filename;
      } else {
        stateChapterEl.textContent = '📖 未定位章节';
      }
    }
    const docState = loadDocReaderState(state.filename);
    if (btnStateDrawer) {
      btnStateDrawer.textContent = `📌 本书状态 (${docState.notes.length})`;
    }
    if (stateDrawerListEl) {
      if (!docState.notes.length) {
        stateDrawerListEl.innerHTML =
          '<div style="font-size:0.72rem;color:#64748b;padding:4px;">暂无已沉淀的符号/要点记忆。点击任意伴读回答底部的「📌 记入本书状态」即可跨章节自动注入上下文。</div>';
      } else {
        stateDrawerListEl.innerHTML = docState.notes
          .map((n, idx) => {
            const pBadge = n.page
              ? `<button type="button" class="reader-chip-btn reader-state-jump" data-page="${n.page}" style="padding:1px 6px;font-size:0.67rem;flex-shrink:0;">P.${n.page}</button>`
              : '';
            return `
              <div class="reader-state-note-item">
                ${pBadge}
                <div class="reader-state-note-text">${escapeHtml(n.text)}</div>
                <button type="button" class="reader-chip-btn reader-state-del" data-idx="${idx}" title="移除此条状态记忆" style="padding:1px 5px;font-size:0.66rem;flex-shrink:0;">✕</button>
              </div>
            `;
          })
          .join('');
        stateDrawerListEl.querySelectorAll('.reader-state-jump').forEach((btn) => {
          btn.addEventListener('click', () => {
            const p = parseInt(btn.getAttribute('data-page') || '0', 10);
            if (p > 0 && state.mode === 'pdf') scrollToPdfPage(p, true);
          });
        });
        stateDrawerListEl.querySelectorAll('.reader-state-del').forEach((btn) => {
          btn.addEventListener('click', () => {
            const idx = parseInt(btn.getAttribute('data-idx') || '-1', 10);
            if (idx < 0 || !state.filename) return;
            const cur = loadDocReaderState(state.filename);
            cur.notes.splice(idx, 1);
            saveDocReaderState(state.filename, cur);
            refreshReaderStateUI();
          });
        });
      }
    }
  }

  if (btnStateDrawer && stateDrawerEl) {
    btnStateDrawer.addEventListener('click', () => {
      const hidden = stateDrawerEl.style.display === 'none';
      stateDrawerEl.style.display = hidden ? 'flex' : 'none';
      btnStateDrawer.classList.toggle('active', hidden);
      if (hidden) refreshReaderStateUI();
    });
  }
  if (btnStateClose && stateDrawerEl) {
    btnStateClose.addEventListener('click', () => {
      stateDrawerEl.style.display = 'none';
      if (btnStateDrawer) btnStateDrawer.classList.remove('active');
    });
  }

  function loadDocChatHistory(filename) {
    if (!filename) return [];
    try {
      const raw = localStorage.getItem(READER_CHAT_PREFIX + filename);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (_) {
      return [];
    }
  }

  function saveDocChatHistory(filename, records) {
    if (!filename) return;
    try {
      localStorage.setItem(READER_CHAT_PREFIX + filename, JSON.stringify((records || []).slice(-40)));
    } catch (_) {}
  }

  function clearDocChatHistory(filename) {
    if (!filename) return;
    try {
      localStorage.removeItem(READER_CHAT_PREFIX + filename);
    } catch (_) {}
  }

  function renderAiFooterMeta(model, page, rawAnswerText = '') {
    const parts = [];
    if (page) {
      parts.push(
        `<button type="button" class="reader-chip-btn reader-msg-page-jump" data-page="${page}" style="padding:1px 7px;font-size:0.68rem;">📍 P.${page}</button>`
      );
    }
    if (rawAnswerText) {
      const excerpt = String(rawAnswerText)
        .replace(/[#*`>\n]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .slice(0, 220);
      parts.push(
        `<button type="button" class="reader-chip-btn reader-msg-pin-state" data-page="${page || ''}" data-note="${escapeHtml(excerpt)}" style="padding:1px 7px;font-size:0.68rem;" title="将本条关键定义/结论记入本书状态记忆，跨章自动注入">📌 记入本书状态</button>`
      );
    }
    if (model) {
      parts.push(`<span style="font-size:0.7rem;color:#94a3b8;margin-left:auto;">模型: ${escapeHtml(model)}</span>`);
    }
    if (!parts.length) return '';
    return `<div style="display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-top:8px;padding-top:6px;border-top:1px dashed #e2e8f0;">${parts.join('')}</div>`;
  }

  function bindMsgPageJumps(containerEl) {
    if (!containerEl) return;
    containerEl.querySelectorAll('.reader-msg-page-jump').forEach((btn) => {
      btn.addEventListener('click', () => {
        const p = parseInt(btn.getAttribute('data-page') || '0', 10);
        if (p > 0 && state.mode === 'pdf') scrollToPdfPage(p, true);
      });
    });
    containerEl.querySelectorAll('.reader-msg-pin-state').forEach((btn) => {
      btn.addEventListener('click', () => {
        if (!state.filename) return;
        const noteText = (btn.getAttribute('data-note') || '').trim();
        const p = parseInt(btn.getAttribute('data-page') || '0', 10) || null;
        if (!noteText) return;
        const docState = loadDocReaderState(state.filename);
        if (!docState.notes.some((n) => n.text === noteText)) {
          docState.notes.push({ page: p, text: noteText, createdAt: Date.now() });
          saveDocReaderState(state.filename, docState);
          refreshReaderStateUI();
        }
        btn.textContent = '✅ 已记入本书状态';
        btn.disabled = true;
      });
    });
  }

  function renderDocChatHistory(filename) {
    const records = loadDocChatHistory(filename);
    state.history = records.map((r) => ({
      role: r.role,
      content: r.content,
      page: r.page || null,
      model: r.model || '',
    }));
    if (!aiFeedEl) return;
    if (!state.history.length) {
      aiFeedEl.innerHTML =
        '<div class="reader-empty-hint">📖 在左侧选中段落，或点击「✂️ 框选问AI」拉框截取模糊扫描件、公式或图表，即可直接提问。</div>';
      return;
    }
    aiFeedEl.innerHTML = '';
    state.history.forEach((rec) => {
      const div = appendAiMessage(rec.role, rec.content, null);
      if (div && rec.role === 'assistant') {
        const metaHtml = renderAiFooterMeta(rec.model, rec.page, rec.content);
        if (metaHtml) {
          div.insertAdjacentHTML('beforeend', metaHtml);
          bindMsgPageJumps(div);
        }
      }
    });
    aiFeedEl.scrollTop = aiFeedEl.scrollHeight;
  }

  function appendAiMessage(role, htmlOrText, thumbUrl = null) {
    if (!aiFeedEl) return;
    const emptyHint = aiFeedEl.querySelector('.reader-empty-hint');
    if (emptyHint) emptyHint.remove();
    const div = document.createElement('div');
    div.className = `reader-msg ${role}`;
    const thumbHtml = thumbUrl ? `<img src="${thumbUrl}" class="reader-msg-thumb" onclick="if(window.openImageModal)openImageModal('${thumbUrl}','框选原文区域')" />` : '';
    div.innerHTML = thumbHtml + (role === 'assistant' ? renderBasicMarkdown(htmlOrText) : escapeHtml(htmlOrText).replace(/\n/g, '<br/>'));
    aiFeedEl.appendChild(div);
    aiFeedEl.scrollTop = aiFeedEl.scrollHeight;
    return div;
  }

  async function sendReaderQuestion(overrideQuestion = null) {
    if (!state.filename) {
      alert('请先在左侧选择或上传一篇文档');
      return;
    }
    const q = (overrideQuestion !== null ? overrideQuestion : (aiInputEl ? aiInputEl.value : '')).trim();
    if (!q && !state.cropImageData && !state.selectedText) return;

    const usedImg = state.cropImageData;
    const usedText = state.selectedText;
    const askedPage = state.mode === 'pdf' ? state.page : null;
    const askedFile = state.filename;
    const chapterPath = computeChapterPath(askedPage || 1);
    const docState = loadDocReaderState(askedFile);
    const topChapNow = chapterPath.split(' › ')[0] || '';
    const topChapPrev = (docState.lastAskedChapter || '').split(' › ')[0] || '';
    const isTopicSwitch =
      Boolean(askedPage && docState.lastAskedPage && Math.abs(askedPage - docState.lastAskedPage) > 15) ||
      Boolean(topChapNow && topChapPrev && topChapNow !== topChapPrev);
    const topicMode = isTopicSwitch ? 'SWITCH' : 'INHERIT';

    // 跨章节切换主题时，仅保留近页对话历史，防止 200 页前的旧问题串台干扰
    const filteredHistory = isTopicSwitch && askedPage
      ? state.history.filter((h) => !h.page || Math.abs(h.page - askedPage) <= 15).slice(-6)
      : state.history.slice(-16);

    const displayUser = q || (usedImg ? '请识别并解读框选区域内容' : '请解读选中段落');
    appendAiMessage('user', displayUser, usedImg || null);
    if (aiInputEl && overrideQuestion === null) aiInputEl.value = '';

    const loadingDiv = appendAiMessage('assistant', '⏳ 多模态伴读助手正在识别与思考...');
    if (btnAiSend) btnAiSend.disabled = true;
    try {
      const resp = await fetch('/api/reader/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: askedFile,
          page: askedPage,
          chapter_path: chapterPath || null,
          topic_mode: topicMode,
          pinned_notes: docState.notes.slice(-12),
          selected_text: usedText || null,
          crop_image_data: usedImg || null,
          question: q,
          model_mode: modeSelectEl ? modeSelectEl.value : 'fast',
          history: filteredHistory.map((h) => ({ role: h.role, content: h.content })),
        }),
      });
      const res = await resp.json();
      if (!resp.ok || res.status !== 'success') throw new Error(res.detail || `HTTP ${resp.status}`);
      const ans = (res.data && res.data.answer) || '';
      const modelName = (res.data && res.data.model) || '';
      loadingDiv.innerHTML = renderBasicMarkdown(ans) + renderAiFooterMeta(modelName, askedPage, ans);
      bindMsgPageJumps(loadingDiv);
      state.history.push({ role: 'user', content: displayUser, page: askedPage, model: '' });
      state.history.push({ role: 'assistant', content: ans, page: askedPage, model: modelName });
      saveDocChatHistory(askedFile, state.history);
      docState.lastAskedPage = askedPage;
      docState.lastAskedChapter = chapterPath;
      saveDocReaderState(askedFile, docState);
      aiFeedEl.scrollTop = aiFeedEl.scrollHeight;
    } catch (err) {
      loadingDiv.innerHTML = `<span style="color:#b91c1c;">❌ 伴读解答失败: ${escapeHtml(err.message)}</span>`;
    } finally {
      if (btnAiSend) btnAiSend.disabled = false;
    }
  }

  async function loadCurrentPdfPageText() {
    if (!state.filename || state.mode !== 'pdf' || !textDrawerBodyEl) return;
    if (textDrawerTitleEl) textDrawerTitleEl.textContent = `📄 第 ${state.page} 页可复制文本（可直接划选复制或提问）`;
    textDrawerBodyEl.textContent = '⏳ 正在提取本页文本层...';
    try {
      const resp = await fetch(`/api/reader/pdf_page_text?filename=${encodeURIComponent(state.filename)}&page=${state.page}`);
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      const txt = (data.text || '').trim();
      textDrawerBodyEl.textContent = txt || '（本页为纯扫描图片或无内嵌文本层，请使用上方「✂️ 框选区域问 AI」直接多模态识别）';
    } catch (err) {
      textDrawerBodyEl.textContent = `提取失败: ${err.message}`;
    }
  }

  if (btnTextToggle && textDrawerEl) {
    btnTextToggle.addEventListener('click', () => {
      if (state.mode !== 'pdf') {
        alert('当前文档已直接以可复制文本形式展示，直接鼠标划选即可复制。');
        return;
      }
      const isHidden = textDrawerEl.style.display === 'none';
      textDrawerEl.style.display = isHidden ? 'flex' : 'none';
      btnTextToggle.classList.toggle('active', isHidden);
      if (isHidden) loadCurrentPdfPageText();
    });
  }
  if (btnCloseTextDrawer && textDrawerEl) {
    btnCloseTextDrawer.addEventListener('click', () => {
      textDrawerEl.style.display = 'none';
      if (btnTextToggle) btnTextToggle.classList.remove('active');
    });
  }
  if (btnCopyPage && textDrawerBodyEl) {
    btnCopyPage.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(textDrawerBodyEl.textContent || '');
        const orig = btnCopyPage.textContent;
        btnCopyPage.textContent = '✅ 已复制';
        setTimeout(() => { btnCopyPage.textContent = orig; }, 1600);
      } catch (err) {
        alert(`复制失败: ${err.message}`);
      }
    });
  }
  if (textDrawerBodyEl) {
    textDrawerBodyEl.addEventListener('mouseup', (e) => {
      const sel = (window.getSelection ? window.getSelection().toString() : '').trim();
      if (sel.length >= 2) {
        state.selectedText = sel;
        updateContextPreview();
        showSelectionPopover(sel, e.clientX, e.clientY);
      }
    });
  }

  // Toolbar & action bindings
  if (btnPrev) btnPrev.addEventListener('click', () => renderPdfPage(state.page - 1));
  if (btnNext) btnNext.addEventListener('click', () => renderPdfPage(state.page + 1));
  if (pageInput) {
    pageInput.addEventListener('change', () => renderPdfPage(parseInt(pageInput.value, 10) || 1));
  }
  if (btnZoomIn) {
    btnZoomIn.addEventListener('click', () => {
      state.zoom = Math.min(2.5, Math.round((state.zoom + 0.2) * 100) / 100);
      applyPdfZoom();
      saveCurrentDocProgress();
    });
  }
  if (btnZoomOut) {
    btnZoomOut.addEventListener('click', () => {
      state.zoom = Math.max(0.5, Math.round((state.zoom - 0.2) * 100) / 100);
      applyPdfZoom();
      saveCurrentDocProgress();
    });
  }
  if (btnCropToggle) {
    btnCropToggle.addEventListener('click', () => {
      state.cropMode = !state.cropMode;
      btnCropToggle.classList.toggle('active', state.cropMode);
      btnCropToggle.textContent = state.cropMode ? '✂️ 框选中（拖动画框）' : '✂️ 框选问AI';
      const container = document.getElementById('reader-pdf-scroll-container');
      if (container) container.classList.toggle('crop-mode', state.cropMode);
    });
  }
  if (btnToggleLeft) {
    btnToggleLeft.addEventListener('click', () => {
      const hideLeft = !layoutReader.classList.contains('hide-left');
      const hideRight = layoutReader.classList.contains('hide-right');
      applyPanelCollapse(hideLeft, hideRight, true);
    });
  }
  if (btnToggleRight) {
    btnToggleRight.addEventListener('click', () => {
      const hideLeft = layoutReader.classList.contains('hide-left');
      const hideRight = !layoutReader.classList.contains('hide-right');
      applyPanelCollapse(hideLeft, hideRight, true);
    });
  }

  // 书内关键词极速检索（Ctrl+K 或工具栏 🔍 搜书）
  function openSearchBar() {
    if (!searchBarEl) return;
    searchBarEl.style.display = 'flex';
    if (btnSearchToggle) btnSearchToggle.classList.add('active');
    if (searchInputEl) {
      searchInputEl.focus();
      searchInputEl.select();
    }
  }

  function closeSearchBar() {
    if (!searchBarEl) return;
    searchBarEl.style.display = 'none';
    if (btnSearchToggle) btnSearchToggle.classList.remove('active');
  }

  function highlightSnippet(snippet, query) {
    const safe = escapeHtml(snippet || '');
    const q = (query || '').trim();
    if (!q) return safe;
    const escapedQ = q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    return safe.replace(new RegExp(`(${escapedQ})`, 'gi'), '<mark class="reader-search-hl">$1</mark>');
  }

  async function runInDocSearch() {
    if (!state.filename || !searchInputEl || !searchResultsEl) return;
    const q = searchInputEl.value.trim();
    if (!q) {
      searchResultsEl.innerHTML = '';
      if (searchStatusEl) searchStatusEl.textContent = '';
      return;
    }
    if (searchStatusEl) searchStatusEl.textContent = '⏳ 检索中...';
    try {
      const resp = await fetch(
        `/api/reader/search?filename=${encodeURIComponent(state.filename)}&q=${encodeURIComponent(q)}&limit=300`
      );
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      const hits = data.results || [];
      if (!hits.length) {
        if (searchStatusEl) searchStatusEl.textContent = '未找到匹配内容';
        searchResultsEl.innerHTML =
          '<div style="font-size:0.74rem;color:#64748b;padding:4px 6px;">未在当前文档文本层中找到匹配项（若是纯扫描版 PDF 无文本层，可翻页后使用「✂️ 框选问AI」）。</div>';
        return;
      }
      if (searchStatusEl) {
        searchStatusEl.textContent = `命中 ${hits.length}${data.truncated ? '+' : ''} 处`;
      }
      searchResultsEl.innerHTML = hits
        .map((h) => {
          const locLabel = data.mode === 'pdf' ? `P.${h.page}` : escapeHtml(h.location_label || `节 ${h.section_index}`);
          const dataAttr = data.mode === 'pdf' ? `data-page="${h.page}"` : `data-sec="${h.section_index}"`;
          return `
            <button type="button" class="reader-search-hit" ${dataAttr}>
              <span class="reader-search-hit-page">${locLabel}</span>
              <span class="reader-search-hit-snippet">${highlightSnippet(h.snippet, q)}</span>
            </button>
          `;
        })
        .join('');
      searchResultsEl.querySelectorAll('.reader-search-hit').forEach((btn) => {
        btn.addEventListener('click', () => {
          const p = btn.getAttribute('data-page');
          const sec = btn.getAttribute('data-sec');
          if (p) {
            scrollToPdfPage(parseInt(p, 10) || 1, true);
          } else if (sec) {
            const target = document.getElementById(`reader-sec-${sec}`);
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        });
      });
    } catch (err) {
      if (searchStatusEl) searchStatusEl.textContent = '检索失败';
      searchResultsEl.innerHTML = `<div style="font-size:0.74rem;color:#b91c1c;padding:4px 6px;">${escapeHtml(err.message)}</div>`;
    }
  }

  if (btnSearchToggle) {
    btnSearchToggle.addEventListener('click', () => {
      if (!searchBarEl) return;
      if (searchBarEl.style.display === 'none') openSearchBar();
      else closeSearchBar();
    });
  }
  if (btnSearchClose) btnSearchClose.addEventListener('click', closeSearchBar);
  if (btnSearchRun) btnSearchRun.addEventListener('click', runInDocSearch);
  if (searchInputEl) {
    searchInputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        runInDocSearch();
      } else if (e.key === 'Escape') {
        closeSearchBar();
      }
    });
  }
  document.addEventListener('keydown', (e) => {
    if (layoutReader.style.display === 'none') return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      openSearchBar();
    } else if (e.key === 'Escape' && searchBarEl && searchBarEl.style.display !== 'none') {
      closeSearchBar();
    }
  });
  if (btnClearCtx) btnClearCtx.addEventListener('click', clearSelectedContext);
  if (ctxImgEl) {
    ctxImgEl.addEventListener('click', () => {
      if (state.cropImageData && typeof window.openImageModal === 'function') {
        window.openImageModal(state.cropImageData, `${state.filename} 第 ${state.page} 页框选区域`);
      }
    });
  }
  if (btnAiSend) btnAiSend.addEventListener('click', () => sendReaderQuestion());
  if (aiInputEl) {
    aiInputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendReaderQuestion();
      }
    });
    // 支持 Ctrl+V 直接粘贴剪贴板截图
    aiInputEl.addEventListener('paste', (e) => {
      const items = (e.clipboardData && e.clipboardData.items) || [];
      for (const item of items) {
        if (item.type && item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (!file) continue;
          const reader = new FileReader();
          reader.onload = (evt) => {
            state.cropImageData = evt.target.result;
            updateContextPreview();
            if (ctxLabelEl) ctxLabelEl.textContent = '🖼️ 已粘贴剪贴板截图（多模态原图直传）';
          };
          reader.readAsDataURL(file);
          e.preventDefault();
          break;
        }
      }
    });
  }
  if (btnAttachImg && imgFileInput) {
    btnAttachImg.addEventListener('click', () => imgFileInput.click());
    imgFileInput.addEventListener('change', (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => {
        state.cropImageData = evt.target.result;
        updateContextPreview();
        if (ctxLabelEl) ctxLabelEl.textContent = `🖼️ 已附带图片: ${file.name}`;
      };
      reader.readAsDataURL(file);
      imgFileInput.value = '';
    });
  }
  if (btnReaderMic) {
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    btnReaderMic.addEventListener('click', async () => {
      if (!isRecording) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          audioChunks = [];
          mediaRecorder = new MediaRecorder(stream);
          mediaRecorder.ondataavailable = (ev) => {
            if (ev.data && ev.data.size > 0) audioChunks.push(ev.data);
          };
          mediaRecorder.onstop = async () => {
            stream.getTracks().forEach((t) => t.stop());
            btnReaderMic.textContent = '⏳ 转写中...';
            btnReaderMic.disabled = true;
            try {
              const blob = new Blob(audioChunks, { type: 'audio/webm' });
              const fd = new FormData();
              fd.append('file', blob, 'reader_voice.webm');
              const resp = await fetch('/api/voice/transcribe', { method: 'POST', body: fd });
              const data = await resp.json();
              const text = (data.text || data.transcript || '').trim();
              if (text && aiInputEl) {
                aiInputEl.value = (aiInputEl.value ? aiInputEl.value + ' ' : '') + text;
                aiInputEl.focus();
              }
            } catch (err) {
              alert(`语音转写失败: ${err.message}`);
            } finally {
              btnReaderMic.textContent = '🎙️ 语音';
              btnReaderMic.disabled = false;
            }
          };
          mediaRecorder.start();
          isRecording = true;
          btnReaderMic.textContent = '⏹️ 停止录音';
          btnReaderMic.style.color = '#b91c1c';
        } catch (err) {
          alert(`无法开启麦克风: ${err.message}`);
        }
      } else {
        isRecording = false;
        btnReaderMic.style.color = '';
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
          mediaRecorder.stop();
        }
      }
    });
  }
  if (btnAiClear) {
    btnAiClear.addEventListener('click', () => {
      state.history = [];
      clearDocChatHistory(state.filename);
      if (aiFeedEl) aiFeedEl.innerHTML = '<div class="reader-empty-hint">📖 选中段落或点击工具栏「✂️ 框选问AI」拉框截取公式/图表/扫描件，即可多模态解答疑惑。</div>';
    });
  }
  document.querySelectorAll('.reader-quick-prompt').forEach((btn) => {
    btn.addEventListener('click', () => sendReaderQuestion(btn.getAttribute('data-prompt')));
  });
  if (btnSendToMain) {
    btnSendToMain.addEventListener('click', () => {
      // 切回主研讨画卷，把当前问题/引用带入主输入框
      if (tabDialogue) tabDialogue.click();
      const qInput = document.getElementById('query-input');
      if (qInput) {
        const prefix = state.filename ? `结合文档《${state.filename}》` + (state.mode === 'pdf' ? `第 ${state.page} 页` : '') : '';
        const quote = state.selectedText ? `：\n"${state.selectedText.slice(0, 3000)}"\n请做深度多源考据与严谨推导。` : '请结合前沿文献深度剖析该问题。';
        qInput.value = prefix + quote;
        qInput.focus();
      }
    });
  }
  if (btnUploadDoc && uploadInput) {
    btnUploadDoc.addEventListener('click', () => uploadInput.click());
    uploadInput.addEventListener('change', async (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const fd = new FormData();
      fd.append('file', file);
      if (viewportEl) viewportEl.innerHTML = `<div class="reader-empty-hint">⏳ 正在上传并解析《${escapeHtml(file.name)}》...</div>`;
      try {
        const resp = await fetch('/api/documents/upload', { method: 'POST', body: fd });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || '上传失败');
        uploadInput.value = '';
        await loadDocumentList(data.filename);
      } catch (err) {
        alert(`上传失败: ${err.message}`);
      }
    });
  }
})();
