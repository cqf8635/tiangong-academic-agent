/**
 * 系统配置 (.env) 可视化管理与本地模型一键下载控制器
 * 独立模块，与 app.js / radar_controller.js 解耦
 */
(function initSettingsController() {
  const btnOpen = document.getElementById('btn-open-settings');
  const modal = document.getElementById('settings-modal');
  const btnClose = document.getElementById('btn-close-settings');
  const btnCancel = document.getElementById('btn-cancel-settings');
  const btnSave = document.getElementById('btn-save-settings');
  const btnDownloadAll = document.getElementById('btn-download-all-models');
  const envPathEl = document.getElementById('settings-env-path');
  const modelsListEl = document.getElementById('settings-models-list');
  const downloadLogEl = document.getElementById('settings-download-log');
  const saveStatusEl = document.getElementById('settings-save-status');

  if (!btnOpen || !modal) return;

  const FIELD_KEYS = [
    'OPENAI_BASE_URL',
    'OPENAI_API_KEY',
    'OPENAI_MODEL',
    'OPENAI_FALLBACK_MODEL',
    'TRANSLATION_LLM_MODEL',
    'LLM_TIMEOUT',
    'OPENALEX_MAILTO',
    'SEMANTICSCHOLAR_API_KEY',
    'HF_ENDPOINT',
    'PAPER_EMBEDDING_MODEL',
    'PAPER_RERANKER_MODEL',
    'WHISPER_MODEL_PATH',
  ];

  let pollTimer = null;

  function renderModelsStatus(modelsStatus) {
    if (!modelsListEl || !modelsStatus) return;
    const order = ['embedding', 'reranker', 'whisper'];
    modelsListEl.innerHTML = order
      .filter((k) => modelsStatus[k])
      .map((key) => {
        const item = modelsStatus[key];
        const badge = item.ready
          ? '<span class="model-status-badge ready">✅ 已就绪</span>'
          : '<span class="model-status-badge missing">⚠️ 未下载</span>';
        const btnText = item.ready ? '🔄 重新校验/更新' : '⬇️ 一键下载';
        return `
          <div class="settings-model-row">
            <div class="settings-model-info">
              <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                <strong>${item.label}</strong>
                ${badge}
              </div>
              <div class="settings-model-meta">${item.repo} · ${item.path}</div>
            </div>
            <button type="button" class="action-btn-mini btn-dl-single-model" data-target="${key}">${btnText}</button>
          </div>
        `;
      })
      .join('');

    modelsListEl.querySelectorAll('.btn-dl-single-model').forEach((btn) => {
      btn.addEventListener('click', () => triggerModelDownload(btn.getAttribute('data-target')));
    });
  }

  async function loadSettings() {
    if (saveStatusEl) saveStatusEl.textContent = '';
    try {
      const resp = await fetch('/api/settings/env');
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      if (envPathEl && data.env_path) {
        envPathEl.textContent = `配置文件: ${data.env_path}（仅本机可修改 · API 与模型保存即生效）`;
      }
      const values = data.values || {};
      const secrets = data.secrets_configured || {};
      FIELD_KEYS.forEach((key) => {
        const input = document.getElementById(`cfg-${key}`);
        if (!input) return;
        input.value = values[key] || '';
        if (secrets[key]) {
          input.placeholder = `已配置 (${secrets[key]})，留空保持不变`;
        }
      });
      renderModelsStatus(data.models_status);
      await pollDownloadStatusOnce();
    } catch (err) {
      if (saveStatusEl) saveStatusEl.textContent = `加载配置失败: ${err.message}`;
    }
  }

  async function pollDownloadStatusOnce() {
    try {
      const resp = await fetch('/api/settings/models/download/status');
      if (!resp.ok) return;
      const st = await resp.json();
      renderModelsStatus(st.models_status);
      if (st.logs && st.logs.length > 0 && downloadLogEl) {
        downloadLogEl.style.display = 'block';
        downloadLogEl.textContent = st.logs.join('\n');
        downloadLogEl.scrollTop = downloadLogEl.scrollHeight;
      }
      if (st.running) {
        if (!pollTimer) {
          pollTimer = setInterval(pollDownloadStatusOnce, 1500);
        }
      } else if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    } catch (_) {}
  }

  async function triggerModelDownload(target) {
    const hfInput = document.getElementById('cfg-HF_ENDPOINT');
    const hfEndpoint = hfInput ? hfInput.value.trim() : '';
    try {
      if (downloadLogEl) {
        downloadLogEl.style.display = 'block';
        downloadLogEl.textContent = `⏳ 正在启动 ${target} 模型下载...`;
      }
      const resp = await fetch('/api/settings/models/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target, hf_endpoint: hfEndpoint || null }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      if (!pollTimer) {
        pollTimer = setInterval(pollDownloadStatusOnce, 1500);
      }
      await pollDownloadStatusOnce();
    } catch (err) {
      if (downloadLogEl) {
        downloadLogEl.style.display = 'block';
        downloadLogEl.textContent = `❌ 启动下载失败: ${err.message}`;
      }
    }
  }

  async function saveSettings() {
    const values = {};
    FIELD_KEYS.forEach((key) => {
      const input = document.getElementById(`cfg-${key}`);
      if (input) values[key] = input.value.trim();
    });
    if (btnSave) btnSave.disabled = true;
    if (saveStatusEl) saveStatusEl.textContent = '⏳ 正在写入 .env...';
    try {
      const resp = await fetch('/api/settings/env', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ values }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      const pill = document.getElementById('pill-model-name');
      if (pill && data.model_name) {
        pill.innerHTML = `<span class="dot gold"></span> ${data.model_name}`;
      }
      renderModelsStatus(data.models_status);
      if (saveStatusEl) {
        saveStatusEl.textContent = data.restart_required
          ? '✅ 已保存到 .env（本地模型路径已变更，重启后端后加载新权重）'
          : '✅ 已保存到 .env 并即时生效！';
      }
      if (typeof window.showAcademicToast === 'function') {
        window.showAcademicToast(
          data.restart_required ? '配置已保存，本地模型路径变更需重启生效' : '系统配置已保存并热生效',
          'success'
        );
      }
    } catch (err) {
      if (saveStatusEl) saveStatusEl.textContent = `❌ 保存失败: ${err.message}`;
    } finally {
      if (btnSave) btnSave.disabled = false;
    }
  }

  function closeModal() {
    modal.style.display = 'none';
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  const btnToolsMenu = document.getElementById('btn-header-tools-menu');
  const toolsMenu = document.getElementById('header-tools-menu');
  if (btnToolsMenu && toolsMenu) {
    btnToolsMenu.addEventListener('click', (e) => {
      e.stopPropagation();
      toolsMenu.classList.toggle('open');
    });
    toolsMenu.querySelectorAll('.header-dropdown-item').forEach((item) => {
      item.addEventListener('click', () => {
        toolsMenu.classList.remove('open');
      });
    });
    document.addEventListener('click', (e) => {
      if (!toolsMenu.contains(e.target) && e.target !== btnToolsMenu) {
        toolsMenu.classList.remove('open');
      }
    });
  }

  btnOpen.addEventListener('click', () => {
    if (toolsMenu) toolsMenu.classList.remove('open');
    modal.style.display = 'flex';
    loadSettings();
  });
  if (btnClose) btnClose.addEventListener('click', closeModal);
  if (btnCancel) btnCancel.addEventListener('click', closeModal);
  if (btnSave) btnSave.addEventListener('click', saveSettings);
  if (btnDownloadAll) btnDownloadAll.addEventListener('click', () => triggerModelDownload('all'));
  modal.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.style.display === 'flex') closeModal();
  });
})();
