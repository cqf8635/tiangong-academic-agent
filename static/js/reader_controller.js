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
    zoom: 1.0,
    cropMode: false,
    selectedText: '',
    cropImageData: '',
    history: [],
    loadedOnce: false,
  };

  function escapeHtml(s) {
    return String(s || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function renderBasicMarkdown(md) {
    let html = escapeHtml(md || '');
    // Preserve display/inline math via KaTeX if available
    const mathTokens = [];
    if (window.katex) {
      html = html.replace(/\$\$([\s\S]*?)\$\$/g, (m, f) => {
        const id = `@@RMATH_${mathTokens.length}@@`;
        try {
          mathTokens.push(`<div class="math-display-wrap">${window.katex.renderToString(f.trim(), { displayMode: true, throwOnError: false, output: 'html' })}</div>`);
        } catch (_) {
          mathTokens.push(m);
        }
        return id;
      });
      html = html.replace(/(?<!\\)\$([^\$\n]+?)(?<!\\)\$/g, (m, f) => {
        const id = `@@RMATH_${mathTokens.length}@@`;
        try {
          mathTokens.push(window.katex.renderToString(f.trim(), { displayMode: false, throwOnError: false, output: 'html' }));
        } catch (_) {
          mathTokens.push(m);
        }
        return id;
      });
    }
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/`([^`\n]+)`/g, '<code class="inline-code">$1</code>');
    html = html.replace(/^###\s+(.+)$/gm, '<div style="font-weight:800;color:#184c43;margin:8px 0 4px;">$1</div>');
    html = html.replace(/^##\s+(.+)$/gm, '<div style="font-weight:900;color:#0f172a;margin:10px 0 4px;">$1</div>');
    html = html.replace(/\n/g, '<br/>');
    mathTokens.forEach((tok, i) => {
      html = html.replace(`@@RMATH_${i}@@`, tok);
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
    });
  }
  if (tabRadar) {
    tabRadar.addEventListener('click', () => {
      tabReader.classList.remove('active');
      layoutReader.style.display = 'none';
    });
  }
  tabReader.addEventListener('click', () => openReaderWorkspace());

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
      docListEl.innerHTML = state.docs
        .map((d) => {
          const active = d.filename === state.filename ? 'active' : '';
          const mb = (d.size_bytes / (1024 * 1024)).toFixed(2);
          return `
            <button type="button" class="reader-doc-item ${active}" data-filename="${escapeHtml(d.filename)}">
              <span class="reader-doc-name">📄 ${escapeHtml(d.filename)}</span>
              <span class="reader-doc-meta">${escapeHtml((d.type || '').toUpperCase())} · ${mb} MB</span>
            </button>
          `;
        })
        .join('');
      docListEl.querySelectorAll('.reader-doc-item').forEach((btn) => {
        btn.addEventListener('click', () => openDocument(btn.getAttribute('data-filename')));
      });

      const target = autoSelectFilename || (!state.filename && state.docs[0] ? state.docs[0].filename : null);
      if (target) {
        await openDocument(target);
      }
    } catch (err) {
      if (docListEl) docListEl.innerHTML = `<div style="color:#b91c1c;font-size:0.75rem;padding:8px;">加载文档列表失败: ${escapeHtml(err.message)}</div>`;
    }
  }

  async function openDocument(filename) {
    if (!filename) return;
    state.filename = filename;
    state.page = 1;
    state.zoom = 1.0;
    state.history = [];
    clearSelectedContext();
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
        if (pageTotalEl) pageTotalEl.textContent = `/ ${state.pageCount} 页`;
        renderPdfToc(meta.toc || []);
        await renderPdfPage(1);
      } else {
        state.pageCount = 1;
        if (pageTotalEl) pageTotalEl.textContent = `共 ${meta.section_count || 0} 节`;
        if (pageInput) pageInput.value = '1';
        renderSectionsView(meta.sections || []);
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
      btn.addEventListener('click', () => renderPdfPage(parseInt(btn.getAttribute('data-page'), 10) || 1));
    });
  }

  async function renderPdfPage(pageNum) {
    if (!state.filename || state.mode !== 'pdf') return;
    const p = Math.max(1, Math.min(state.pageCount, pageNum));
    state.page = p;
    if (pageInput) pageInput.value = String(p);
    if (zoomLabelEl) zoomLabelEl.textContent = `${Math.round(state.zoom * 100)}%`;
    if (textDrawerEl && textDrawerEl.style.display !== 'none') {
      loadCurrentPdfPageText();
    }

    const imgUrl = `/api/reader/pdf_page?filename=${encodeURIComponent(state.filename)}&page=${p}&scale=2.2`;
    viewportEl.innerHTML = `
      <div class="reader-pdf-stage ${state.cropMode ? 'crop-mode' : ''}" id="reader-pdf-stage">
        <img id="reader-pdf-img" class="reader-pdf-img" src="${imgUrl}" alt="第 ${p} 页" />
        <div id="reader-crop-rect" class="reader-crop-rect"></div>
      </div>
    `;
    const stage = document.getElementById('reader-pdf-stage');
    const img = document.getElementById('reader-pdf-img');
    const rectEl = document.getElementById('reader-crop-rect');

    img.onload = () => {
      const baseW = Math.min(viewportEl.clientWidth - 44, img.naturalWidth / 1.5);
      img.style.width = `${Math.round(Math.max(420, baseW) * state.zoom)}px`;
    };

    // 绑定拉框高清截图交互
    let dragging = false;
    let startX = 0;
    let startY = 0;
    stage.addEventListener('mousedown', (e) => {
      if (!state.cropMode || e.button !== 0) return;
      const bounds = stage.getBoundingClientRect();
      dragging = true;
      startX = e.clientX - bounds.left;
      startY = e.clientY - bounds.top;
      rectEl.style.left = `${startX}px`;
      rectEl.style.top = `${startY}px`;
      rectEl.style.width = '0px';
      rectEl.style.height = '0px';
      rectEl.style.display = 'block';
    });
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const bounds = stage.getBoundingClientRect();
      const curX = Math.max(0, Math.min(bounds.width, e.clientX - bounds.left));
      const curY = Math.max(0, Math.min(bounds.height, e.clientY - bounds.top));
      const x = Math.min(startX, curX);
      const y = Math.min(startY, curY);
      const w = Math.abs(curX - startX);
      const h = Math.abs(curY - startY);
      rectEl.style.left = `${x}px`;
      rectEl.style.top = `${y}px`;
      rectEl.style.width = `${w}px`;
      rectEl.style.height = `${h}px`;
    });
    window.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      const x = parseFloat(rectEl.style.left) || 0;
      const y = parseFloat(rectEl.style.top) || 0;
      const w = parseFloat(rectEl.style.width) || 0;
      const h = parseFloat(rectEl.style.height) || 0;
      rectEl.style.display = 'none';
      if (w < 16 || h < 16 || !img.naturalWidth) return;

      // 按原始高 DPI 像素裁剪，确保模糊扫描件与小号公式下标尽可能清晰
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
    });
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
    // 支持直接鼠标划词选中文本
    const stage = document.getElementById('reader-sections-stage');
    if (stage) {
      stage.addEventListener('mouseup', () => {
        const sel = (window.getSelection ? window.getSelection().toString() : '').trim();
        if (sel.length >= 4) {
          state.selectedText = sel;
          updateContextPreview();
        }
      });
    }
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
          filename: state.filename,
          page: state.mode === 'pdf' ? state.page : null,
          selected_text: usedText || null,
          crop_image_data: usedImg || null,
          question: q,
          model_mode: modeSelectEl ? modeSelectEl.value : 'fast',
          history: state.history.slice(-6),
        }),
      });
      const res = await resp.json();
      if (!resp.ok || res.status !== 'success') throw new Error(res.detail || `HTTP ${resp.status}`);
      const ans = (res.data && res.data.answer) || '';
      const modelTag = res.data && res.data.model ? `\n\n<span style="font-size:0.7rem;color:#94a3b8;">模型: ${escapeHtml(res.data.model)}</span>` : '';
      loadingDiv.innerHTML = renderBasicMarkdown(ans) + modelTag;
      state.history.push({ role: 'user', content: displayUser });
      state.history.push({ role: 'assistant', content: ans });
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
    textDrawerBodyEl.addEventListener('mouseup', () => {
      const sel = (window.getSelection ? window.getSelection().toString() : '').trim();
      if (sel.length >= 4) {
        state.selectedText = sel;
        updateContextPreview();
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
      const img = document.getElementById('reader-pdf-img');
      if (img && img.naturalWidth) {
        const baseW = Math.min(viewportEl.clientWidth - 44, img.naturalWidth / 1.5);
        img.style.width = `${Math.round(Math.max(420, baseW) * state.zoom)}px`;
      }
      if (zoomLabelEl) zoomLabelEl.textContent = `${Math.round(state.zoom * 100)}%`;
    });
  }
  if (btnZoomOut) {
    btnZoomOut.addEventListener('click', () => {
      state.zoom = Math.max(0.5, Math.round((state.zoom - 0.2) * 100) / 100);
      const img = document.getElementById('reader-pdf-img');
      if (img && img.naturalWidth) {
        const baseW = Math.min(viewportEl.clientWidth - 44, img.naturalWidth / 1.5);
        img.style.width = `${Math.round(Math.max(420, baseW) * state.zoom)}px`;
      }
      if (zoomLabelEl) zoomLabelEl.textContent = `${Math.round(state.zoom * 100)}%`;
    });
  }
  if (btnCropToggle) {
    btnCropToggle.addEventListener('click', () => {
      state.cropMode = !state.cropMode;
      btnCropToggle.classList.toggle('active', state.cropMode);
      btnCropToggle.textContent = state.cropMode ? '✂️ 框选中（拖动画框）' : '✂️ 框选区域问 AI';
      const stage = document.getElementById('reader-pdf-stage');
      if (stage) stage.classList.toggle('crop-mode', state.cropMode);
    });
  }
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
      if (aiFeedEl) aiFeedEl.innerHTML = '<div class="reader-empty-hint">📖 选中段落或点击工具栏「✂️ 框选区域问 AI」拉框截取公式/图表/扫描件，即可多模态解答疑惑。</div>';
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
        const quote = state.selectedText ? `：\n"${state.selectedText.slice(0, 600)}"\n请做深度多源考据与严谨推导。` : '请结合前沿文献深度剖析该问题。';
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
