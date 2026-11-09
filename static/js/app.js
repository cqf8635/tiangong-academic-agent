/**
 * Frontend Controller with Multi-Domain Sessions and Agent Network Topology.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements (兼容多种模板 ID)
  const chatFeed = document.getElementById('chat-feed');
  const userInput = document.getElementById('query-input') || document.getElementById('user-input');
  const btnSend = document.getElementById('btn-send');
  const btnMic = document.getElementById('btn-mic');
  const btnQuickSummary = document.getElementById('btn-quick-summary');
  const btnToggleGraph = document.getElementById('btn-toggle-graph');
  const agentGraphCard = document.getElementById('agent-graph-card');
  const graphStatusBadge = document.getElementById('graph-status-badge') || document.getElementById('graph-status-text');
  
  const agentStatePill = document.getElementById('agent-state-pill');
  
  const sessionChannelList = document.getElementById('session-channel-list');
  const btnNewSession = document.getElementById('btn-new-chat') || document.getElementById('btn-new-session');
  
  const fileInput = document.getElementById('file-input');
  const dropzoneTrigger = document.getElementById('dropzone-trigger') || document.getElementById('dropzone');
  const dropzone = document.getElementById('dropzone');
  const uploadProgress = document.getElementById('upload-progress');
  const progressText = document.getElementById('progress-text');
  const documentList = document.getElementById('document-list');
  const docCountBadge = document.getElementById('doc-count');
  
  const actionItemsList = document.getElementById('action-items-list');
  const evidencePanelList = document.getElementById('evidence-list') || document.getElementById('evidence-panel-list');
  const streamLogs = document.getElementById('stream-logs');
  const globalAudio = document.getElementById('global-tts-audio');

  let currentSessionId = "session_ai";
  let isRecording = false;

  const selectTtsVoice = document.getElementById('select-tts-voice');
  if (selectTtsVoice) {
    const savedVoice = localStorage.getItem('preferred_tts_voice');
    if (savedVoice) {
      selectTtsVoice.value = savedVoice;
    }
    selectTtsVoice.addEventListener('change', () => {
      localStorage.setItem('preferred_tts_voice', selectTtsVoice.value);
    });
  }

  function getCurrentVoice() {
    return selectTtsVoice ? selectTtsVoice.value : (localStorage.getItem('preferred_tts_voice') || 'zh-CN-XiaoyiNeural');
  }

  // =========================================================================
  // 1. LIVE ORCHESTRATION 宏观多智能体编排网络动效控制器
  // =========================================================================
  const orchestrationModal = document.getElementById('orchestration-modal');
  const btnCloseOrch = document.getElementById('btn-close-orch');
  const hubCoordinator = document.getElementById('hub-coordinator');
  const hubStatusText = document.getElementById('hub-status-text');

  function resetGraphNodes() {
    document.querySelectorAll('.orbit-card, .hub-center-node').forEach(node => {
      node.classList.remove('active-node', 'running-node');
    });
    document.querySelectorAll('.radial-line').forEach(line => {
      line.classList.remove('active');
    });
    for (let i = 1; i <= 7; i++) {
      const p = document.getElementById(`status-orbit-${i}`);
      if (p) p.textContent = '等待任务';
    }
    if (hubStatusText) hubStatusText.textContent = '可调度';
  }

  function setOrbitNode(nodeNum, state, pillText = null) {
    const node = document.getElementById(`orbit-node-${nodeNum}`);
    const line = document.getElementById(`line-orbit-${nodeNum}`);
    const status = document.getElementById(`status-orbit-${nodeNum}`);

    if (!node) return;
    node.classList.remove('active-node', 'running-node');
    if (line) line.classList.remove('active');

    if (state === 'running') {
      node.classList.add('running-node');
      if (line) line.classList.add('active');
    } else if (state === 'active') {
      node.classList.add('active-node');
      if (line) line.classList.add('active');
    }

    if (status && pillText) {
      status.textContent = pillText;
    }
  }

  function setHubStatus(text, isActive = true) {
    if (hubCoordinator) {
      if (isActive) hubCoordinator.classList.add('active-node');
      else hubCoordinator.classList.remove('active-node');
    }
    if (hubStatusText) hubStatusText.textContent = text;
  }

  function highlightGraphPipelineStep() {
    // 接口返回前没有节点进度事件，等待时间不能证明任何阶段已执行。
    resetGraphNodes();
    setHubStatus('正在处理请求');
  }

  function highlightGraphFromResult(data) {
    if (!data) return;
    resetGraphNodes();
    const stages = data.stage_statuses || {};
    const generated = stages.generation === 'completed' || stages.reflective_revision === 'completed';

    if (stages.intent_parsing === 'failed') {
      setOrbitNode(1, 'idle', '意图解析失败');
    } else if (data.intent_decision && data.intent_decision.intent !== 'error') {
      setOrbitNode(1, 'active', '✅ 已返回意图解析');
    } else {
      setOrbitNode(1, 'idle', '暂无意图解析结果');
    }

    const evidence = data.retrieved_evidence || [];
    let onlineCount = 0;
    let localCount = 0;
    evidence.forEach(e => {
      const sources = [e.doc_type, e.source, ...(e.sources_merged || [])]
        .filter(Boolean).map(src => String(src).toLowerCase());
      if (sources.some(src => ['arxiv', 'openalex', 'crossref', 'doaj', 'online', 'github', 'fused_academic'].includes(src))) onlineCount++;
      if (e.is_local_indexed || sources.some(src => ['local', 'local_pdf', 'pdf', 'pptx', 'docx', 'txt'].includes(src))) localCount++;
    });
    setOrbitNode(2, onlineCount ? 'active' : 'idle', onlineCount ? `返回在线资料 ${onlineCount} 条` : '本轮无在线资料');
    setOrbitNode(3, localCount ? 'active' : 'idle', localCount ? `返回本地资料 ${localCount} 条` : '本轮无本地资料');

    let reviewLabel = '暂无执行状态记录';
    let reviewActive = false;
    if (stages.reflective_revision === 'failed') reviewLabel = '修订未完成';
    else if (stages.reflective_revision === 'completed') {
      reviewLabel = '修订稿已生成，尚未复核';
      reviewActive = true;
    } else if (stages.reflective_revision === 'unchanged') reviewLabel = '修订未改变初稿，问题待确认';
    else if (stages.dialectical_inquiry === 'failed') reviewLabel = '模型质询未完成';
    else if (stages.dialectical_inquiry === 'completed') {
      reviewLabel = data.review_verdict?.needs_revision === false ? '模型审核未要求修订'
        : data.review_verdict?.needs_revision === true ? '模型审核要求修订' : '已生成审核意见，未记录修订决策';
      reviewActive = true;
    } else if (stages.dialectical_inquiry === 'checklist') reviewLabel = '仅生成待审查清单';
    else if (stages.dialectical_inquiry === 'skipped') reviewLabel = '未执行模型质询';
    else if (stages.reflective_revision === 'skipped') reviewLabel = '未执行模型修订';
    setOrbitNode(4, reviewActive ? 'active' : 'idle', reviewLabel);

    const targetAgent = data.target_agent || '';
    if (targetAgent.includes('CrossModal') || targetAgent.includes('PPT')) {
      setOrbitNode(5, generated && evidence.length ? 'active' : 'idle',
        generated && evidence.length ? '已返回课件候选' : '暂无课件匹配结果');
    }
    if (data.mindmap_markdown) setOrbitNode(6, 'active', '已生成内容导图');
    else if (targetAgent.includes('Meeting') || targetAgent.includes('Summarizer')) {
      setOrbitNode(6, generated ? 'active' : 'idle', generated ? '已生成会议摘要' : '会议摘要未生成');
    }

    if (data.audio_url) setOrbitNode(7, 'active', '语音已生成');
    else setOrbitNode(7, generated ? 'active' : 'idle', generated ? '回答已返回' : '暂无生成成果');
    const hasFailure = Object.values(stages).includes('failed');
    setHubStatus(hasFailure ? '部分步骤未完成' : generated ? '回答已返回' : '请求已返回', false);
  }

  function highlightGraphStep(stepName, targetAgent = null) {
    if (stepName === 'ROUTING') {
      highlightGraphPipelineStep(0);
    } else if (stepName === 'EXECUTING') {
      let nodeMap = {
        'OnlinePaperSearchAgent': 2,
        'ArxivSearchAgent': 2,
        'DocumentSearchAgent': 3,
        'CrossModalRAGAgent': 5,
        'CrossModalAgent': 5,
        'DerivationAgent': 4,
        'TutorDerivationAgent': 4,
        'ConceptTutorAgent': 4,
        'StudyPlannerAgent': 4,
        'MeetingSummarizer': 6,
        'SummarizerAgent': 6,
        'MeetingAgent': 6,
      };
      const nNum = nodeMap[targetAgent] || 4;
      setOrbitNode(nNum, 'running', `⚡ ${targetAgent || 'Agent'} 执行中`);
      setHubStatus(`执行: ${targetAgent || 'Agent'}`);
    } else if (stepName === 'TTS') {
      setOrbitNode(7, 'running', '🔊 音画呈现中');
      setHubStatus('羽若协同呈现中');
    }
  }

  const btnToggleLeft = document.getElementById('btn-toggle-left');
  const btnToggleRight = document.getElementById('btn-toggle-right');
  const sidebarLeft = document.getElementById('sidebar-left');
  const sidebarRight = document.getElementById('sidebar-right');
  const mainLayout = document.getElementById('main-layout');

  function updateLayoutColumns() {
    if (!mainLayout) return;
    const leftCollapsed = sidebarLeft && sidebarLeft.classList.contains('collapsed');
    const rightCollapsed = sidebarRight && sidebarRight.classList.contains('collapsed');
    const resizer = document.getElementById('sidebar-resizer');

    if (resizer) {
      resizer.style.display = rightCollapsed ? 'none' : 'block';
    }

    const rightW = 'var(--sidebar-right-width, 360px)';

    if (leftCollapsed && rightCollapsed) {
      mainLayout.style.gridTemplateColumns = '1fr';
    } else if (leftCollapsed) {
      mainLayout.style.gridTemplateColumns = `minmax(0, 1fr) auto ${rightW}`;
    } else if (rightCollapsed) {
      mainLayout.style.gridTemplateColumns = '280px minmax(0, 1fr)';
    } else {
      mainLayout.style.gridTemplateColumns = `280px minmax(0, 1fr) auto ${rightW}`;
    }
  }

  if (btnToggleLeft && sidebarLeft) {
    btnToggleLeft.addEventListener('click', () => {
      sidebarLeft.classList.toggle('collapsed');
      btnToggleLeft.classList.toggle('active', !sidebarLeft.classList.contains('collapsed'));
      updateLayoutColumns();
    });
  }

  if (btnToggleRight && sidebarRight) {
    btnToggleRight.addEventListener('click', () => {
      sidebarRight.classList.toggle('collapsed');
      btnToggleRight.classList.toggle('active', !sidebarRight.classList.contains('collapsed'));
      updateLayoutColumns();
    });
  }

  const btnToggleAvatar = document.getElementById('btn-toggle-avatar');
  if (btnToggleAvatar) {
    btnToggleAvatar.addEventListener('click', () => {
      if (sidebarRight && sidebarRight.classList.contains('collapsed')) {
        sidebarRight.classList.remove('collapsed');
        if (btnToggleRight) btnToggleRight.classList.add('active');
        updateLayoutColumns();
      }
      activateTab('tab-avatar');
    });
  }

  if (btnToggleGraph && orchestrationModal) {
    btnToggleGraph.addEventListener('click', () => {
      orchestrationModal.classList.add('active');
    });
  }

  if (btnCloseOrch && orchestrationModal) {
    btnCloseOrch.addEventListener('click', () => {
      orchestrationModal.classList.remove('active');
    });
  }

  if (orchestrationModal) {
    orchestrationModal.addEventListener('click', (e) => {
      if (e.target === orchestrationModal) {
        orchestrationModal.classList.remove('active');
      }
    });
  }

  // =========================================================================
  // 2. 自由会话管理、重命名、删除与历史记录 (User-Defined Sessions)
  // =========================================================================
  const btnNewChat = document.getElementById('btn-new-chat');

  async function initFreshBlankChat() {
    currentSessionId = null;
    localStorage.removeItem('antigravity_active_session');

    // 清除侧边栏所有高亮
    document.querySelectorAll('.session-item').forEach(item => {
      item.classList.remove('active');
    });

    if (activeSessionTitle) activeSessionTitle.textContent = '💬 兰台问学 · 新建学术会话';
    chatFeed.innerHTML = `
      <div class="welcome-scroll-card">
        <div class="scroll-seal">🪷</div>
        <h3 class="scroll-title">问学云山 · 寻真致远</h3>
        <p class="scroll-subtitle">「博学之，审问之，慎思之，明辨之，笃行之」</p>
        <div class="scroll-features">
          <span class="feature-tag">📜 全格式论文深潜</span>
          <span class="feature-tag">📐 LaTeX 数理公式推演</span>
          <span class="feature-tag">🎙️ 温润自然语音导读</span>
          <span class="feature-tag">📋 讲堂流水纪要提炼</span>
        </div>
      </div>
    `;
    updateEvidencePanel([]);
    showMindmapEmptyState();
    sessionTurnsMindmaps = [];
    currentFullBookMindmap = '';
    currentActiveDocumentName = '';
    if (userInput) {
      userInput.value = '';
      userInput.focus();
    }
  }

  async function loadSessions(targetSelectId = null) {
    try {
      const resp = await fetch('/api/sessions');
      const data = await resp.json();
      const sessions = data.sessions || [];
      
      renderSessionList(sessions);

      if (targetSelectId) {
        currentSessionId = targetSelectId;
        await switchSession(targetSelectId);
      } else {
        // 默认进入纯净新会话界面，不自动回放历史旧对话
        await initFreshBlankChat();
      }
    } catch (e) {
      console.error('Failed to load sessions:', e);
      throw e;
    }
  }

  function renderSessionList(sessions) {
    if (sessions.length === 0) {
      sessionChannelList.innerHTML = '<div class="empty-hint">暂无会话，点击上方新建</div>';
      return;
    }

    sessionChannelList.innerHTML = sessions.map(s => {
      const cleanTitle = (s.title || '学术研讨会话').replace(/^[💬\s]+/, '');
      const isActive = (s.session_id === currentSessionId);
      return `
      <div class="session-item ${isActive ? 'active' : ''}" data-id="${s.session_id}">
        <div class="session-title-box">
          <span class="session-title-text" title="${escapeHtml(cleanTitle)}">${escapeHtml(cleanTitle)}</span>
        </div>
        <span class="session-msg-badge">${s.message_count}</span>
        <div class="session-actions">
          <button class="action-btn-mini btn-rename" data-id="${s.session_id}" title="重命名会话">✏️</button>
          <button class="action-btn-mini btn-delete" data-id="${s.session_id}" title="删除会话">🗑️</button>
        </div>
      </div>
    `;
    }).join('');

    // 绑定点击切换会话
    document.querySelectorAll('.session-item').forEach(item => {
      item.addEventListener('click', (e) => {
        if (e.target.closest('.session-actions')) return;
        switchSession(item.dataset.id);
      });
    });

    // 绑定重命名事件
    document.querySelectorAll('.btn-rename').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const sid = btn.dataset.id;
        const currentItem = btn.closest('.session-item');
        const currentTitle = currentItem.querySelector('.session-title-text').textContent;
        const newTitle = prompt('修改会话名称:', currentTitle);
        if (newTitle && newTitle.trim()) {
          try {
            await fetch(`/api/sessions/${sid}`, {
              method: 'PUT',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ title: newTitle.trim() }),
            });
            loadSessions();
          } catch (err) {
            alert('重命名失败: ' + err.message);
          }
        }
      });
    });

    // 绑定删除事件
    document.querySelectorAll('.btn-delete').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const sid = btn.dataset.id;
        if (confirm('确定要删除这个研讨会话及其历史记录吗？')) {
          try {
            await fetch(`/api/sessions/${sid}`, { method: 'DELETE' });
            if (sid === currentSessionId) {
              await createNewChat();
            } else {
              await loadSessions();
            }
          } catch (err) {
            alert('删除失败: ' + err.message);
          }
        }
      });
    });
  }

  const activeSessionTitle = document.getElementById('active-session-title');
  const btnClearCurrentChat = document.getElementById('btn-clear-current-chat');
  const btnDeleteCurrentChat = document.getElementById('btn-delete-current-chat');
  const btnClearAllSessions = document.getElementById('btn-clear-all-sessions');

  // =========================================================================
  // 经卷脉络思维导图控制器 (Universal Markmap D3 Controller & Exporter)
  // =========================================================================
  // =========================================================================
  // 经卷脉络思维导图控制器 (Universal Multi-Scope Markmap D3 Controller)
  // =========================================================================
  let mainMarkmapInstance = null;
  let modalMarkmapInstance = null;
  let currentFullBookMindmap = '';
  let currentFullBookMindmapDoc = '';
  let currentActiveMarkdown = '';
  let currentActiveDocumentName = '';
  
  // 多层次多轮问答导图管理体系
  let sessionTurnsMindmaps = []; // Array of { turnIndex: number, title: string, markdown: string, docName: string }
  let currentMindmapScope = 'current'; // 'current' | 'book' | 'session'
  let currentActiveTurnIndex = 0;

  function showMindmapEmptyState() {
    const placeholder = document.getElementById('mindmap-empty-placeholder');
    const svg = document.getElementById('markmap-svg');
    const titleEl = document.getElementById('mindmap-current-title');
    const badgeEl = document.getElementById('mindmap-type-badge');
    const scopeBar = document.getElementById('mindmap-scope-bar');

    if (placeholder) placeholder.style.display = 'flex';
    if (svg) svg.style.display = 'none';
    if (titleEl) titleEl.textContent = '经卷知识脉络';
    if (badgeEl) badgeEl.textContent = '🌱 静待研讨';
    if (scopeBar) scopeBar.style.display = 'none';
    currentActiveMarkdown = '';
  }

  function showMindmapCanvas() {
    const placeholder = document.getElementById('mindmap-empty-placeholder');
    const svg = document.getElementById('markmap-svg');
    const scopeBar = document.getElementById('mindmap-scope-bar');

    if (placeholder) placeholder.style.display = 'none';
    if (svg) svg.style.display = 'block';
    if (scopeBar) scopeBar.style.display = 'flex';
  }

  function preprocessMindmapMath(mdText) {
    if (!mdText) return '';
    let res = String(mdText);

    {
      // 1. 转换标准 LaTeX 行内定界符 \( ... \)
      res = res.replace(/\\\(([\s\S]*?)\\\)/g, (match, formula) => {
        try {
          return renderMathSource(formula.trim(), false);
        } catch (error) { throw error; }
      });

      // 2. 转换标准行内公式 $ ... $
      res = res.replace(/(?<!\\)\$([^\$\n]+)\$/g, (match, formula) => {
        try {
          return renderMathSource(formula.trim(), false);
        } catch (error) { throw error; }
      });

      // 3. 转换导图节点中以括号包裹的显式 LaTeX 表达式 (如 (\sqrt{d_k})、(\mathbf{M}\in\mathbb{R}^{T\times T}))
      res = res.replace(/（([^）\n]*?\\[a-zA-Z]+[^）\n]*?)）/g, (match, formula) => {
        try {
          return `（${renderMathSource(formula.trim(), false)}）`;
        } catch (error) { throw error; }
      });

      res = res.replace(/\(([^)\n]*?\\[a-zA-Z]+[^)\n]*?)\)/g, (match, formula) => {
        try {
          return `(${renderMathSource(formula.trim(), false)})`;
        } catch (error) { throw error; }
      });
    }

    return res;
  }

  function renderMarkmapTree(svgEl, markdownText) {
    if (!svgEl || !markdownText) return;
    try {
      showMindmapCanvas();
      const cleanMd = preprocessMindmapMath(markdownText);
      const transformer = new window.markmap.Transformer();
      const { root } = transformer.transform(cleanMd);

      const triggerFit = () => {
        if (svgEl && svgEl._markmapInstance) {
          try {
            svgEl._markmapInstance.fit();
          } catch (error) { throw error; }
        }
      };

      if (svgEl._markmapInstance) {
        svgEl._markmapInstance.setData(root);
        setTimeout(triggerFit, 60);
        setTimeout(triggerFit, 250);
      } else {
        svgEl.innerHTML = '';
        const mm = window.markmap.Markmap.create(svgEl, {
          duration: 300,
          nodeMinHeight: 22,
          spacingVertical: 12,
          spacingHorizontal: 80,
          paddingX: 16,
          autoFit: true,
        }, root);
        svgEl._markmapInstance = mm;
        setTimeout(triggerFit, 80);
        setTimeout(triggerFit, 300);
      }
      return svgEl._markmapInstance;
    } catch (e) {
      throw e;
    }
  }

  function buildSynthesizedSessionMindmap() {
    if (sessionTurnsMindmaps.length === 0) return '';
    const cleanSessionTitle = (activeSessionTitle ? activeSessionTitle.textContent.replace(/^[💬\s]+/, '') : '本次学术研讨');
    const lines = [`# 🌐 《${cleanSessionTitle}》全景知识聚合图`];

    sessionTurnsMindmaps.forEach((turn) => {
      lines.push(`## [第${turn.turnIndex}讲] ${turn.title}`);
      const turnLines = turn.markdown.split('\n');
      for (const raw of turnLines) {
        const line = raw.trim();
        if (line.startsWith('# ') || !line) continue;
        if (line.startsWith('## ')) {
          lines.push(`### ${line.replace(/^##\s+/, '')}`);
        } else if (line.startsWith('### ')) {
          lines.push(`#### ${line.replace(/^###\s+/, '')}`);
        } else if (line.startsWith('- ') || line.startsWith('* ')) {
          lines.push(line);
        }
      }
    });
    return lines.join('\n');
  }

  async function updateMindmapScopeUI() {
    const titleEl = document.getElementById('mindmap-current-title');
    const badgeEl = document.getElementById('mindmap-type-badge');
    const turnSelectWrap = document.getElementById('mindmap-turn-select-wrap');
    const turnSelect = document.getElementById('mindmap-turn-select');
    const svg = document.getElementById('markmap-svg');

    // 更新作用域 tab 激活状态
    document.querySelectorAll('.scope-tab-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.scope === currentMindmapScope);
    });

    if (sessionTurnsMindmaps.length === 0 && !currentFullBookMindmap) {
      showMindmapEmptyState();
      return;
    }
    showMindmapCanvas();

    // 填充轮次下拉列表
    if (turnSelect) {
      turnSelect.innerHTML = sessionTurnsMindmaps.map((t, idx) => `
        <option value="${idx}" ${idx === currentActiveTurnIndex ? 'selected' : ''}>[第 ${t.turnIndex} 轮] ${t.title}</option>
      `).join('');
    }

    if (currentMindmapScope === 'current') {
      if (turnSelectWrap) {
        turnSelectWrap.style.display = (sessionTurnsMindmaps.length > 1) ? 'flex' : 'none';
      }
      const activeTurn = sessionTurnsMindmaps[currentActiveTurnIndex] || sessionTurnsMindmaps[sessionTurnsMindmaps.length - 1];
      if (activeTurn) {
        if (titleEl) titleEl.textContent = activeTurn.title;
        if (badgeEl) badgeEl.textContent = `🧠 第 ${activeTurn.turnIndex} 轮精析`;
        currentActiveMarkdown = activeTurn.markdown;
        mainMarkmapInstance = renderMarkmapTree(svg, activeTurn.markdown);
      } else if (currentFullBookMindmap) {
        currentMindmapScope = 'book';
        await updateMindmapScopeUI();
      }
    } else if (currentMindmapScope === 'book') {
      if (turnSelectWrap) turnSelectWrap.style.display = 'none';
      const docName = currentActiveDocumentName;
      if (!docName) throw new Error('未选择书籍，无法加载全书导图');
      const cleanDocName = docName.replace(/\.[^/.]+$/, '');
      if (titleEl) titleEl.textContent = `《${cleanDocName}》全景知识树`;
      if (badgeEl) badgeEl.textContent = '📚 全书宏观结构';

      if (!currentFullBookMindmap || currentFullBookMindmapDoc !== docName) {
        try {
          const resp = await fetch(`/api/documents/${encodeURIComponent(docName)}/mindmap`);
          if (!resp.ok) throw new Error('书籍导图接口失败');
          const data = await resp.json();
          if (!data.mindmap_markdown) throw new Error('未返回书籍导图');
          currentFullBookMindmap = data.mindmap_markdown;
          currentFullBookMindmapDoc = docName;
        } catch (e) {
          throw new Error(`书籍导图加载失败：${e.message}`);
        }
      }
      currentActiveMarkdown = currentFullBookMindmap;
      mainMarkmapInstance = renderMarkmapTree(svg, currentFullBookMindmap);
    } else if (currentMindmapScope === 'session') {
      if (turnSelectWrap) turnSelectWrap.style.display = 'none';
      const cleanSessionTitle = (activeSessionTitle ? activeSessionTitle.textContent.replace(/^[💬\s]+/, '') : '本次研讨');
      if (titleEl) titleEl.textContent = `《${cleanSessionTitle}》全景研讨聚合图`;
      if (badgeEl) badgeEl.textContent = `🌐 聚合 ${sessionTurnsMindmaps.length} 轮问答`;

      const synthesizedMd = buildSynthesizedSessionMindmap();
      currentActiveMarkdown = synthesizedMd;
      mainMarkmapInstance = renderMarkmapTree(svg, synthesizedMd);
    }
  }

  async function loadAndRenderBookMindmap(docName) {
    if (!docName) return;
    currentActiveDocumentName = docName;
    currentMindmapScope = 'book';
    currentFullBookMindmap = '';
    currentFullBookMindmapDoc = '';
    activateTab('tab-mindmap');
    await updateMindmapScopeUI();
  }

  function renderAnswerMindmap(markdownText, topicTitle, docName = '') {
    if (!markdownText) return;
    if (docName) currentActiveDocumentName = docName;
    
    // 加入或更新 turns
    const cleanTitle = sanitizeTopicTitle(topicTitle);
    const existingIndex = sessionTurnsMindmaps.findIndex(t => t.title === cleanTitle);
    if (existingIndex >= 0) {
      sessionTurnsMindmaps[existingIndex].markdown = markdownText;
      currentActiveTurnIndex = existingIndex;
    } else {
      sessionTurnsMindmaps.push({
        turnIndex: sessionTurnsMindmaps.length + 1,
        title: cleanTitle,
        markdown: markdownText,
        docName: docName || currentActiveDocumentName
      });
      currentActiveTurnIndex = sessionTurnsMindmaps.length - 1;
    }

    currentMindmapScope = 'current';
    updateMindmapScopeUI();
  }

  function activateTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.tab === tabId);
    });
    document.querySelectorAll('.tab-content').forEach(c => {
      c.classList.toggle('active', c.id === tabId);
    });
    if (tabId === 'tab-mindmap' && mainMarkmapInstance) {
      setTimeout(() => mainMarkmapInstance.fit(), 80);
    }
  }

  // 导出图片与大纲 (Universal High-Res Exporter)
  function downloadBlob(blob, filename) {
    const url = (window.URL || window.webkitURL).createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      (window.URL || window.webkitURL).revokeObjectURL(url);
    }, 1000);
  }

  function prepareCleanExportSvg(targetSvg) {
    if (!targetSvg) return null;

    const bbox = targetSvg.getBBox();
    if (!bbox || bbox.width <= 0 || bbox.height <= 0) {
      throw new Error('思维导图尺寸无效，无法导出');
    }
    const vw = Math.max(Math.ceil(bbox.width) + 160, 1200);
    const vh = Math.max(Math.ceil(bbox.height) + 160, 800);
    const vx = Math.floor(bbox.x) - 80;
    const vy = Math.floor(bbox.y) - 80;

    const svgClone = targetSvg.cloneNode(true);
    svgClone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    svgClone.setAttribute('width', vw);
    svgClone.setAttribute('height', vh);
    svgClone.setAttribute('viewBox', `${vx} ${vy} ${vw} ${vh}`);

    // 1. 插入实体纯白底色矩形 (彻底解决暗黑系统看图器背景透黑看不清线条的问题)
    const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bgRect.setAttribute('x', vx);
    bgRect.setAttribute('y', vy);
    bgRect.setAttribute('width', vw);
    bgRect.setAttribute('height', vh);
    bgRect.setAttribute('fill', '#ffffff');
    svgClone.insertBefore(bgRect, svgClone.firstChild);

    // 2. 将 Markmap 的 <foreignObject> 转为标准原生 SVG <text>
    // 彻底根治: 
    // a) 浏览器 Canvas/Image 对含 foreignObject 的 SVG 有安全限制，导出失败时明确报错;
    // b) 桌面看图软件 (Linux Loupe / eog / macOS Preview) 无法解析 foreignObject 导致文字完全隐形.
    const cloneFos = Array.from(svgClone.querySelectorAll('foreignObject'));
    const sourceFos = Array.from(targetSvg.querySelectorAll('foreignObject'));

    cloneFos.forEach((fo, idx) => {
      const sourceFo = sourceFos[idx] || null;
      const sourceDiv = sourceFo ? (sourceFo.querySelector('div') || sourceFo.firstElementChild) : null;
      const cloneDiv = fo.querySelector('div') || fo.firstElementChild;

      let text = '';
      if (sourceDiv) {
        text = sourceDiv.innerText || sourceDiv.textContent || '';
      } else if (cloneDiv) {
        text = cloneDiv.innerText || cloneDiv.textContent || '';
      } else {
        text = fo.textContent || '';
      }
      text = text.trim();
      if (!text) {
        fo.remove();
        return;
      }

      const x = parseFloat(fo.getAttribute('x') || '0');
      const y = parseFloat(fo.getAttribute('y') || '0');

      let color = '#10302a';
      let fontSize = '13px';
      let fontWeight = '500';
      if (sourceDiv) {
        try {
          const compStyle = window.getComputedStyle(sourceDiv);
          if (compStyle.color) color = compStyle.color;
          if (compStyle.fontSize) fontSize = compStyle.fontSize;
          if (compStyle.fontWeight) fontWeight = compStyle.fontWeight;
        } catch (error) { throw error; }
      }

      const textEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      textEl.setAttribute('x', x);
      textEl.setAttribute('y', y);
      textEl.setAttribute('dominant-baseline', 'hanging');
      textEl.setAttribute('fill', color);
      textEl.setAttribute('font-size', fontSize);
      textEl.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Serif SC", "PingFang SC", "Microsoft YaHei", sans-serif');
      textEl.setAttribute('font-weight', fontWeight);

      const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
      if (lines.length <= 1) {
        textEl.textContent = text;
      } else {
        lines.forEach((lineText, lineIdx) => {
          const tspan = document.createElementNS('http://www.w3.org/2000/svg', 'tspan');
          tspan.setAttribute('x', x);
          tspan.setAttribute('dy', lineIdx === 0 ? '0' : '1.3em');
          tspan.textContent = lineText;
          textEl.appendChild(tspan);
        });
      }

      if (fo.parentNode) {
        fo.parentNode.replaceChild(textEl, fo);
      }
    });

    const style = document.createElement('style');
    style.textContent = `
      text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Serif SC", "PingFang SC", "Microsoft YaHei", sans-serif; }
      .markmap-node-circle { fill: #ffffff; stroke-width: 2.2px; }
      line, path { fill: none; }
    `;
    svgClone.insertBefore(style, svgClone.firstChild);

    return { svgClone, width: vw, height: vh, vx, vy };
  }

  function exportSVGToFile(svgEl, filename = '学术知识思维导图.svg') {
    if (!currentActiveMarkdown) {
      alert('请先在左侧选择书籍或发起研讨提问以生成思维导图');
      return;
    }
    try {
      const targetSvg = (svgEl && svgEl.clientWidth > 0) ? svgEl : document.getElementById('markmap-svg');
      const prepared = prepareCleanExportSvg(targetSvg);
      if (!prepared) return;

      const svgString = new XMLSerializer().serializeToString(prepared.svgClone);
      const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
      downloadBlob(blob, filename.endsWith('.svg') ? filename : `${filename}.svg`);
    } catch (err) {
      console.error('Export SVG error:', err);
      alert(`SVG 导出失败: ${err.message}`);
    }
  }

  async function exportSVGToPNG(svgEl, filename = '学术知识思维导图.png') {
    if (!currentActiveMarkdown) {
      if (!currentActiveDocumentName) {
        alert('请先选择书籍或发起研讨提问');
        return;
      }
      await loadAndRenderBookMindmap(currentActiveDocumentName);
    }
    if (!currentActiveMarkdown) {
      alert('请先在左侧选择书籍或发起研讨提问以生成思维导图');
      return;
    }

    try {
      const targetSvg = (svgEl && svgEl.clientWidth > 0) ? svgEl : document.getElementById('markmap-svg');
      const prepared = prepareCleanExportSvg(targetSvg);
      if (!prepared) return;

      const { svgClone, width, height } = prepared;
      const svgString = new XMLSerializer().serializeToString(svgClone);
      const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
      const blobUrl = (window.URL || window.webkitURL).createObjectURL(svgBlob);

      const img = new Image();
      img.onload = () => {
        try {
          const canvas = document.createElement('canvas');
          const scale = 2; // 2x 超高清分辨率采样
          canvas.width = width * scale;
          canvas.height = height * scale;
          const ctx = canvas.getContext('2d');
          ctx.scale(scale, scale);
          ctx.fillStyle = '#ffffff';
          ctx.fillRect(0, 0, width, height);
          ctx.drawImage(img, 0, 0);

          canvas.toBlob((pngBlob) => {
            if (pngBlob) {
              downloadBlob(pngBlob, filename.endsWith('.png') ? filename : `${filename}.png`);
            } else {
              alert('PNG 导出失败：未生成图像数据');
            }
            (window.URL || window.webkitURL).revokeObjectURL(blobUrl);
          }, 'image/png');
        } catch (e) {
          console.error('PNG 导出失败:', e);
          alert(`PNG 导出失败：${e.message}`);
          (window.URL || window.webkitURL).revokeObjectURL(blobUrl);
        }
      };
      img.onerror = (e) => {
        console.error('PNG 图像加载失败:', e);
        alert('PNG 导出失败：图像加载失败');
        (window.URL || window.webkitURL).revokeObjectURL(blobUrl);
      };
      img.src = blobUrl;
    } catch (err) {
      console.error('PNG 导出失败:', err);
      alert(`PNG 导出失败：${err.message}`);
    }
  }

  function copyActiveMindmapMarkdown() {
    if (!currentActiveMarkdown) {
      alert('当前尚无知识导图大纲可复制');
      return;
    }
    navigator.clipboard.writeText(currentActiveMarkdown).then(() => {
      alert('✅ 知识导图 Markdown 大纲已成功复制到剪贴板！');
    }).catch(() => {
      alert('复制失败，请手动选取');
    });
  }

  async function createNewChat(customTitle = '新建学术研讨会话') {
    try {
      stopTTS();
      // 新建会话时，主动清空底部输入框残留文本与上一会话锁定的文档状态
      if (userInput) {
        userInput.value = '';
        userInput.style.height = 'auto';
      }
      currentActiveDocumentName = '';
      currentFullBookMindmap = '';
      currentFullBookMindmapDoc = '';
      const safeTitle = (typeof customTitle === 'string' && customTitle.trim())
        ? customTitle.trim()
        : '新建学术研讨会话';
      const resp = await fetch('/api/sessions/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: safeTitle, domain: 'custom' }),
      });
      const data = await resp.json();
      currentSessionId = data.session.session_id;
      localStorage.setItem('antigravity_active_session', currentSessionId);
      await loadSessions(currentSessionId);
      if (userInput) {
        userInput.focus();
      }
      return data.session;
    } catch (e) {
      console.error('创建新会话失败:', e);
      throw e;
    }
  }

  function stopTTS() {
    if (globalAudio) {
      globalAudio.pause();
      globalAudio.currentTime = 0;
    }
    setAssistantState('IDLE');
    resetGraphNodes();
  }

  async function switchSession(sessionId) {
    if (!sessionId) return;
    currentSessionId = sessionId;
    localStorage.setItem('antigravity_active_session', sessionId);

    // 切换会话时，清空底部输入框残留并停止上一会话正在播放的任何语音
    if (userInput) {
      userInput.value = '';
      userInput.style.height = 'auto';
    }
    stopTTS();
    resetGraphNodes();

    // 严密重置当前会话的思维导图轮次列表与索引，彻底杜绝跨会话轮次串味污染
    sessionTurnsMindmaps = [];
    currentActiveTurnIndex = 0;
    currentMindmapScope = 'current';

    // 严密更新侧边栏高亮
    document.querySelectorAll('.session-item').forEach(el => {
      el.classList.toggle('active', el.dataset.id === sessionId);
    });

    chatFeed.innerHTML = '<div class="empty-hint">正在加载会话历史...</div>';

    try {
      const resp = await fetch(`/api/sessions/${sessionId}`);
      const data = await resp.json();
      const session = data.session;

      const rawTitle = session.title || '学术研讨会话';
      const cleanTitle = rawTitle.replace(/^[💬\s]+/, '');
      activeSessionTitle.textContent = `💬 ${cleanTitle}`;

      chatFeed.innerHTML = '';
      let latestEvidence = [];

      if (!session.messages || session.messages.length === 0) {
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'welcome-scroll-card';
        welcomeDiv.innerHTML = `
          <div class="scroll-seal">🪷</div>
          <h3 class="scroll-title">问学云山 · 寻真致远</h3>
          <p class="scroll-subtitle">「博学之，审问之，慎思之，明辨之，笃行之」</p>
          <div class="scroll-features">
            <span class="feature-tag">📜 全格式论文深潜</span>
            <span class="feature-tag">📐 LaTeX 数理公式推演</span>
            <span class="feature-tag">🎙️ 温润自然语音导读</span>
            <span class="feature-tag">📋 讲堂流水纪要提炼</span>
          </div>
        `;
        chatFeed.appendChild(welcomeDiv);
        updateEvidencePanel([]);
        // 空白对话：默认呈现虚席占位，不展示图谱
        showMindmapEmptyState();
      } else {
        let turnCounter = 0;
        let lastUserQuery = '';

        let userTurnCounter = 0;
        let agentTurnCounter = 0;

        session.messages.forEach(msg => {
          if (!msg) return;
          const textContent = String(msg.content || '').trim();
          if (!textContent && !msg.image_url) return;
          if (msg.sender === 'user') {
            userTurnCounter++;
            lastUserQuery = textContent;
            appendMessage('user', textContent, userTurnCounter, msg.image_url || null);
          } else {
            agentTurnCounter++;
            let mindmapMd = msg.mindmap_markdown;
            if (mindmapMd && mindmapMd.trim() && mindmapMd.includes('\n')) {
              turnCounter++;
              const turnTitle = sanitizeTopicTitle(lastUserQuery || `第 ${turnCounter} 轮研讨精要`);
              sessionTurnsMindmaps.push({
                turnIndex: turnCounter,
                title: turnTitle,
                markdown: mindmapMd,
                docName: (msg.retrieved_evidence && msg.retrieved_evidence[0]) ? msg.retrieved_evidence[0].doc_name : currentActiveDocumentName
              });
            }

            appendAgentMessage({
              final_answer: msg.content,
              intent_decision: msg.intent_decision,
              target_agent: msg.target_agent,
              retrieved_evidence: msg.retrieved_evidence || [],
              mindmap_markdown: mindmapMd,
              audio_url: msg.audio_url,
              topicTitle: lastUserQuery,
              agent_dialogue_history: msg.agent_dialogue_history || [],
              synthesized_evidence_slate: msg.synthesized_evidence_slate,
              dialectical_notes: msg.dialectical_notes,
              review_verdict: msg.review_verdict,
              stage_statuses: msg.stage_statuses || {}
            }, agentTurnCounter);

            // 追踪当前会话中最新一轮智能体的检索证据，绝不残留早期轮次文献
            latestEvidence = (msg.retrieved_evidence && Array.isArray(msg.retrieved_evidence)) ? msg.retrieved_evidence : [];
            if (latestEvidence.length > 0 && latestEvidence[0].doc_name) {
              currentActiveDocumentName = latestEvidence[0].doc_name;
            }
          }
        });

        updateEvidencePanel(latestEvidence);
        const lastAgentMessage = [...session.messages].reverse().find(msg => msg.sender === 'agent');
        if (lastAgentMessage) highlightGraphFromResult(lastAgentMessage);
      }

      currentActiveTurnIndex = Math.max(0, sessionTurnsMindmaps.length - 1);
      currentMindmapScope = 'current';
      await updateMindmapScopeUI();

      setTimeout(() => {
        renderLatexEquations(chatFeed);
        chatFeed.scrollTop = chatFeed.scrollHeight;
      }, 50);
    } catch (e) {
      chatFeed.innerHTML = `<div class="empty-hint">加载历史失败: ${e.message}</div>`;
    }
  }

  // 清空当前会话消息
  if (btnClearCurrentChat) {
    btnClearCurrentChat.addEventListener('click', async () => {
      if (!currentSessionId) return;
      try {
        await fetch(`/api/sessions/${currentSessionId}/clear`, { method: 'POST' });
        await switchSession(currentSessionId);
        await loadSessions();
      } catch (e) {
        console.error('清空失败:', e);
      }
    });
  }

  // 删除当前会话
  if (btnDeleteCurrentChat) {
    btnDeleteCurrentChat.addEventListener('click', async () => {
      if (!currentSessionId) return;
      if (confirm('确定要删除当前研讨会话吗？')) {
        try {
          await fetch(`/api/sessions/${currentSessionId}`, { method: 'DELETE' });
          await createNewChat();
        } catch (e) {
          console.error('删除失败:', e);
        }
      }
    });
  }

  // 清空所有历史会话
  if (btnClearAllSessions) {
    btnClearAllSessions.addEventListener('click', async () => {
      if (confirm('⚠️ 确定要清空全部历史会话记录吗？')) {
        try {
          const resp = await fetch('/api/sessions/clear_all', { method: 'POST' });
          const data = await resp.json();
          currentSessionId = data.initial_session.session_id;
          await loadSessions();
          switchSession(currentSessionId);
        } catch (e) {
          console.error('清空所有会话失败:', e);
        }
      }
    });
  }

  if (btnNewChat) {
    btnNewChat.addEventListener('click', async () => {
      await createNewChat();
    });
  }

  // =========================================================================
  // 3. 学术研讨状态指示器
  // =========================================================================
  function setAssistantState(state, customDesc = null) {
    let pillHtml = '';
    switch (state) {
      case 'IDLE':
        pillHtml = '<span class="state-dot idle"></span> 🌿 研讨就绪 (Ready)';
        break;
      case 'LISTENING':
        pillHtml = '<span class="state-dot listening"></span> 👂 倾听研讨 (Listening)';
        break;
      case 'THINKING':
        pillHtml = '<span class="state-dot thinking"></span> ✨ 经卷推演 (Thinking)';
        break;
      case 'SPEAKING':
        pillHtml = '<span class="state-dot speaking"></span> 🎙️ 语音播报 (Speaking)';
        break;
    }
    if (agentStatePill) agentStatePill.innerHTML = pillHtml;
  }

  // =========================================================================
  // 4. 图片附件与截图粘贴控制器
  // =========================================================================
  let currentAttachedImageBase64 = null;
  const btnAttachImg = document.getElementById('btn-attach-img');
  const chatImageInput = document.getElementById('chat-image-input');
  const chatImgPreviewContainer = document.getElementById('chat-image-preview-container');
  const chatImgPreview = document.getElementById('chat-image-preview');
  const btnRemovePreviewImg = document.getElementById('btn-remove-preview-img');

  function setAttachedImage(base64Url) {
    currentAttachedImageBase64 = base64Url;
    if (base64Url) {
      if (chatImgPreview) chatImgPreview.src = base64Url;
      if (chatImgPreviewContainer) chatImgPreviewContainer.style.display = 'flex';
    } else {
      if (chatImgPreview) chatImgPreview.src = '';
      if (chatImgPreviewContainer) chatImgPreviewContainer.style.display = 'none';
      if (chatImageInput) chatImageInput.value = '';
    }
  }

  if (btnAttachImg && chatImageInput) {
    btnAttachImg.addEventListener('click', () => {
      chatImageInput.click();
    });

    chatImageInput.addEventListener('change', (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => {
        setAttachedImage(evt.target.result);
      };
      reader.readAsDataURL(file);
    });
  }

  if (btnRemovePreviewImg) {
    btnRemovePreviewImg.addEventListener('click', () => {
      setAttachedImage(null);
    });
  }

  // 监听输入框的剪贴板粘贴事件 (支持直接 Ctrl+V 粘贴截图)
  if (userInput) {
    userInput.addEventListener('paste', (e) => {
      const items = e.clipboardData && e.clipboardData.items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
          const file = items[i].getAsFile();
          if (file) {
            e.preventDefault();
            const reader = new FileReader();
            reader.onload = (evt) => {
              setAttachedImage(evt.target.result);
            };
            reader.readAsDataURL(file);
            break;
          }
        }
      }
    });
  }

  // =========================================================================
  // 5. 发送消息与拓扑网络实时联动
  // =========================================================================
  async function handleSendMessage(text = null, isVoice = false) {
    const rawQuery = (text || userInput.value).trim();
    const attachedImg = currentAttachedImageBase64;

    // 既没有文字也没有图片时，不发送
    if (!rawQuery && !attachedImg) return;

    const query = rawQuery;

    const welcomeCard = chatFeed.querySelector('.welcome-scroll-card, .empty-hint');
    if (welcomeCard) welcomeCard.remove();

    const currentTurnNum = sessionTurnsMindmaps.length + 1;
    appendMessage('user', query, currentTurnNum, attachedImg);

    // 清空输入与预览
    userInput.value = '';
    userInput.style.height = 'auto';
    setAttachedImage(null);

    setAssistantState('THINKING');
    highlightGraphStep('ROUTING');

    // 在聊天流中插入实时动态思考与研讨状态指示器
    const thinkingId = 'thinking-' + Date.now();
    const thinkingDiv = document.createElement('div');
    thinkingDiv.id = thinkingId;
    thinkingDiv.className = 'message-bubble agent thinking-bubble';
    thinkingDiv.innerHTML = `
      <div class="msg-identity" aria-hidden="true"><img src="img/agent_avatar.jpg?v=2026" alt="羽若" class="msg-avatar-img"></div>
      <div class="msg-card thinking-card">
        <div class="thinking-spinner-wrap">
          <svg class="thinking-spinner-svg" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2.2" opacity="0.25" />
            <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" />
          </svg>
          <span class="thinking-sparkle-center">✨</span>
        </div>
        <div class="thinking-text-group">
          <div class="thinking-status-row">
            <span id="${thinkingId}-text" class="thinking-phase-text">🧭 正在解析学术意图与知识脉络</span>
            <span class="thinking-dots">
              <span class="dot-bounce"></span>
              <span class="dot-bounce"></span>
              <span class="dot-bounce"></span>
            </span>
          </div>
          <div class="thinking-sub-tip">正在调用领域多智能体协同研讨，请稍候...</div>
        </div>
      </div>
    `;
    chatFeed.appendChild(thinkingDiv);
    chatFeed.scrollTop = chatFeed.scrollHeight;

    const thinkingStages = [
      '🧭 正在解析学术意图与知识脉络',
      '🔍 正在靶向检索高相关学术文献与知识切片',
      '🧠 主研学者正在严密推导核心机理与数理公式',
      '🧐 学术审稿专员正在交叉核验证据与学术规约',
      '✍️ 正在反思重构并凝练高信噪比技术成果'
    ];
    let stageIndex = 0;
    const tickerInterval = setInterval(() => {
      const textEl = document.getElementById(`${thinkingId}-text`);
      if (!textEl) {
        clearInterval(tickerInterval);
        return;
      }
      textEl.classList.add('fade-out');
      setTimeout(() => {
        stageIndex = (stageIndex + 1) % thinkingStages.length;
        const currentEl = document.getElementById(`${thinkingId}-text`);
        if (currentEl) {
          currentEl.textContent = thinkingStages[stageIndex];
          currentEl.classList.remove('fade-out');
          currentEl.classList.add('fade-in');
          setTimeout(() => currentEl.classList.remove('fade-in'), 250);
        }
      }, 250);
    }, 3200);

    const cleanupThinking = () => {
      if (tickerInterval) clearInterval(tickerInterval);
      const placeholder = document.getElementById(thinkingId);
      if (placeholder) placeholder.remove();
    };

    highlightGraphPipelineStep();

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: query, 
          image_data: attachedImg || null,
          session_id: currentSessionId, 
          voice: getCurrentVoice() 
        }),
      });

      if (!resp.ok) throw new Error(`Server returned ${resp.status}`);

      const data = await resp.json();

      cleanupThinking();

      // 拓扑图执行节点根据后端实际执行情况精确高亮
      highlightGraphFromResult(data);

      appendAgentMessage(data, currentTurnNum);

      updateEvidencePanel(data.retrieved_evidence || []);

      if (data.mindmap_markdown) {
        renderAnswerMindmap(data.mindmap_markdown, query || data.query || '当前研讨深度导图');
      }

      // 过滤规则一：仅在语音交互模式下自动播报语音；文字输入模式保持静默文本回复 (用户点击按钮仍可随时播放)
      if (isVoice && data.audio_url) {
        playTTS(data.audio_url);
      } else {
        setAssistantState('IDLE');
      }

      // 若有新调取入库的文献，自动刷新左侧经卷列表
      if (data.active_doc) {
        currentActiveDocumentName = data.active_doc;
        loadDocumentList();
      }

      // 刷新会话列表与标题并保持当前会话聚焦
      if (data.session_id) {
        currentSessionId = data.session_id;
        const sessionsResp = await fetch('/api/sessions');
        const sData = await sessionsResp.json();
        renderSessionList(sData.sessions || []);
        document.querySelectorAll('.session-item').forEach(item => {
          item.classList.toggle('active', item.dataset.id === currentSessionId);
        });
      }
    } catch (err) {
      cleanupThinking();

      appendAgentMessage({
        final_answer: `⚠️ **[研讨请求发生异常]**: ${err.message}\n\n请检查网络或稍后重新发送。`,
        intent_decision: { intent: 'error' },
        target_agent: '',
      });
      setAssistantState('IDLE');
      resetGraphNodes();
    }
  }

  function scrollToChatTurn(turnIndex) {
    if (!turnIndex) return;
    const targetTurnEl = document.getElementById(`chat-turn-agent-${turnIndex}`) || document.getElementById(`chat-turn-user-${turnIndex}`);
    if (targetTurnEl) {
      targetTurnEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
      targetTurnEl.classList.remove('turn-pulse-highlight');
      void targetTurnEl.offsetWidth; // 触发重绘
      targetTurnEl.classList.add('turn-pulse-highlight');
    }
  }

  function appendMessage(sender, text, turnIndex = null, imageUrl = null) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message-bubble ${sender}`;
    if (turnIndex) {
      msgDiv.dataset.turnIndex = turnIndex;
      msgDiv.id = `chat-turn-${sender}-${turnIndex}`;
    }

    if (sender === 'user') {
      let imgHtml = '';
      if (imageUrl) {
        imgHtml = `<img src="${imageUrl}" alt="附带截图" class="user-msg-image-thumb" onclick="openImageModal('${imageUrl}', '学者附带截图')" />`;
      }
      let userHtml = `<p>${escapeHtml(text)}</p>`;
      if (/(\$[^$\n]+\$|（[^）\n]*?\\[a-zA-Z]+[^）\n]*?）|\([^)\n]*?\\[a-zA-Z]+[^)\n]*?\))/.test(text)) {
        const rendered = text
          .replace(/（([^）\n]*?\\[a-zA-Z]+[^）\n]*?)）/g, (m, f) => {
            try {
              return `（${renderMathSource(f.trim(), false)}）`;
            } catch (error) { throw error; }
          })
          .replace(/\(([^)\n]*?\\[a-zA-Z]+[^)\n]*?)\)/g, (m, f) => {
            try {
              return `(${renderMathSource(f.trim(), false)})`;
            } catch (error) { throw error; }
          })
          .replace(/\$([^$\n]+)\$/g, (m, f) => {
            try {
              return renderMathSource(f.trim(), false);
            } catch (error) { throw error; }
          });
        userHtml = `<p>${rendered}</p>`;
      }

      msgDiv.innerHTML = `
        <div class="msg-identity" aria-hidden="true"><img src="img/user_avatar.jpg?v=2026_neutral_v1" alt="学者" class="msg-avatar-img"></div>
        <div class="msg-card">${imgHtml}${userHtml}</div>
      `;
    }
    chatFeed.appendChild(msgDiv);
    chatFeed.scrollTop = chatFeed.scrollHeight;
  }

  function sanitizeTopicTitle(rawTitle) {
    if (!rawTitle) return '核心知识脉络导图';
    const raw = String(rawTitle || '').trim();

    // 提取书名号中的文献名（作为前缀保留，绝不粗暴覆盖真实研讨课题）
    let bookPrefix = '';
    const mBook = raw.match(/[《〈]([^》〉]+)[》〉]/);
    if (mBook) {
      bookPrefix = `《${mBook[1].trim()}》`;
    }

    // 清除前导“根据《...》专著/论文/教材”等前缀引用
    let cleaned = raw.replace(/^(根据|结合|按照|基于)?[《〈][^》〉]+[》〉][专著论文教材材料资料]*[，,、\s]*/g, '');
    cleaned = cleaned.replace(/^(你好[，,、\s]*|您好[，,、\s]*|请问[，,、\s]*|请[您]?|帮我|麻烦|我想知道|我想了解|能不能|可以|请详细|详细|请)/g, '');
    cleaned = cleaned.replace(/^(这个|该|这份|本篇|相关|关于)*/g, '');
    cleaned = cleaned.replace(/^(介绍|分析|概述|总结|推导|解析|解释|梳理|盘点|对比)[一下]*/g, '');
    cleaned = cleaned.replace(/(用流程图直观展示|用流程图展示|流程图直观展示|直观展示|用图表展示|用图表对比|用图表|流程图|架构图|大体内容|大概是整理了什么|整理了什么|主要讲了什么|讲的是什么|核心内容|大概内容|大体介绍|主要内容|概述一下|总结一下|介绍一下|是什么|怎么样|怎么做|如何|为什么|有何区别|核心要点|要旨|有哪些|？|\?|。|！|!)+$/g, '');
    cleaned = cleaned.replace(/(资源里面|文件里面|文档里面|切片里面|内容里面|里面|中|之中|关于)*$/g, '');
    cleaned = cleaned.replace(/(资源|文档|文件|材料|课件|讲义|这本书|该书|该文献|这篇论文)$/g, '');
    cleaned = cleaned.replace(/^[，,、：: 里的中关于]+|[，,、：: 里的中关于]+$/g, '');

    if (cleaned && cleaned.length >= 2) {
      if (bookPrefix && !cleaned.startsWith('《')) {
        return `${bookPrefix}${cleaned} 知识架构导图`;
      }
      return `${cleaned} 知识架构导图`;
    }

    if (bookPrefix) return `${bookPrefix}知识架构精要`;
    return '研讨核心脉络导图';
  }

  function appendAgentMessage(data, turnIndex = null) {
    const welcomeCard = chatFeed.querySelector('.welcome-scroll-card, .empty-hint');
    if (welcomeCard) welcomeCard.remove();

    const msgDiv = document.createElement('div');
    msgDiv.className = 'message-bubble agent';
    if (turnIndex) {
      msgDiv.dataset.turnIndex = turnIndex;
      msgDiv.id = `chat-turn-agent-${turnIndex}`;
    }

    const intentTag = data.intent_decision ? `🎯 意图: ${data.intent_decision.intent}` : 'AI 回复';
    const agentName = data.target_agent ? `🤖 ${data.target_agent}` : '🤖 科研搭档 · 羽若';

    let formattedText = formatMarkdown(data.final_answer);

    let evidenceTagsHtml = '';
    if (data.retrieved_evidence && data.retrieved_evidence.length > 0) {
      evidenceTagsHtml = '<div class="evidence-badges">';
      data.retrieved_evidence.forEach((ev) => {
        evidenceTagsHtml += `<span class="evidence-badge" data-doc="${escapeHtml(ev.doc_name)}">📌 ${formatMathInText(ev.doc_name)} (${escapeHtml(ev.location_label)})</span>`;
      });
      evidenceTagsHtml += '</div>';
    }

    // 渲染证据整理归纳备忘录 (去噪/分类/机制提炼)
    let evidenceSlateHtml = '';
    if (data.synthesized_evidence_slate && data.synthesized_evidence_slate.trim()) {
      evidenceSlateHtml = `
        <details class="evidence-slate-details" style="margin-top: 8px; font-size: 0.74rem; background: #f0fdfa; border: 1px solid #99f6e4; border-radius: 6px; padding: 6px 10px;">
          <summary style="cursor: pointer; font-weight: 700; color: #0f766e; user-select: none; display: flex; align-items: center; gap: 6px;">
            <span>📑 查看多源资料与证据归纳备忘录 (去噪/分类/机制提炼)</span>
          </summary>
          <div class="evidence-slate-body markdown-body-text" style="margin-top: 8px; color: var(--color-ink-regular, #334155); line-height: 1.5;">
            ${formatMarkdown(data.synthesized_evidence_slate)}
          </div>
        </details>
      `;
    }

    // 渲染学术博弈质询备忘录 (审视论据/去冗提纯)
    let dialecticalNotesHtml = '';
    if (data.dialectical_notes && data.dialectical_notes.trim()) {
      dialecticalNotesHtml = `
        <details class="dialectical-notes-details" style="margin-top: 8px; font-size: 0.74rem; background: #fef2f2; border: 1px solid #fecaca; border-radius: 6px; padding: 6px 10px;">
          <summary style="cursor: pointer; font-weight: 700; color: #991b1b; user-select: none; display: flex; align-items: center; gap: 6px;">
            <span>🧐 查看学术博弈质询备忘录 (审视论据/去冗提纯)</span>
          </summary>
          <div class="dialectical-notes-body markdown-body-text" style="margin-top: 8px; color: var(--color-ink-regular, #334155); line-height: 1.5;">
            ${formatMarkdown(data.dialectical_notes)}
          </div>
        </details>
      `;
    }

    let audioPlayBtnHtml = '';
    if (data.audio_url || data.final_answer) {
      audioPlayBtnHtml = `<button class="btn-replay-audio" data-url="${data.audio_url || ''}">🔊 播放语音</button>`;
    }

    let mindmapMd = data.mindmap_markdown;

    let mindmapBtnHtml = '';
    if (mindmapMd && mindmapMd.trim() && mindmapMd.includes('\n')) {
      mindmapBtnHtml = `
        <div class="msg-mindmap-wrapper" style="margin-top: 8px; display: flex; gap: 6px; flex-wrap: wrap;">
          <button class="msg-action-chip btn-toggle-msg-mindmap">🧠 展开本轮导图</button>
          <button class="msg-action-chip btn-sync-to-desk" title="将本轮问答的知识脉络投射至右侧大案头">🎋 投射至案头</button>
          <button class="msg-action-chip btn-export-msg-obsidian" title="将本轮学术研讨与思维导图一键导出至本地 Obsidian 知识库">💎 导出到 Obsidian</button>
          <div class="msg-mindmap-card" style="display: none; width: 100%; margin-top: 6px; background: #fafbf9; border: 1px solid var(--color-silk-border); border-radius: var(--radius-sm); padding: 8px;">
            <svg class="msg-markmap-svg" style="width: 100%; min-height: 260px;"></svg>
          </div>
        </div>
      `;
    }

    let multiAgentTrackHtml = '';
    if (data.agent_dialogue_history && data.agent_dialogue_history.length > 0) {
      const items = data.agent_dialogue_history.map(item => {
        let actionBadge = '';
        let roundBadge = `<span style="display:inline-block; font-size: 0.68rem; padding: 1px 5px; border-radius: 4px; background: #e0f2fe; color: #0369a1; font-weight: 700; margin-right: 4px;">R${item.round !== undefined ? item.round : 0}</span>`;
        if (item.action === 'EVIDENCE_CLEANSING_AND_SYNTHESIS') {
          actionBadge = `<span style="font-size: 0.68rem; padding: 1px 5px; border-radius: 4px; background: #ccfbf1; color: #0f766e; font-weight: 700; margin-right: 4px;">📑 整理归纳</span>`;
        } else if (item.action && item.action.endsWith('_FAILED')) {
          actionBadge = `<span style="color: #991b1b; font-weight: 700;">⚠️ 未完成</span>`;
        } else if (item.action === 'PREPARE_REVIEW_CHECKLIST') {
          actionBadge = `<span style="color: #92400e; font-weight: 700;">📋 待审查清单</span>`;
        } else if (item.action === 'REVISION_UNCHANGED') {
          actionBadge = '<span style="color: #92400e; font-weight: 700;">修订未改变初稿</span>';
        } else if (item.action === 'RETAIN_DRAFT') {
          actionBadge = `<span style="color: #475569; font-weight: 700;">📄 保留初稿</span>`;
        } else if (item.action === 'ORGANIZE_EVIDENCE') {
          actionBadge = `<span style="color: #0f766e; font-weight: 700;">📑 模板整理</span>`;
        } else if (item.action === 'DELIVER_DISCIPLINARY_FOUNDATION') {
          actionBadge = `<span style="font-size: 0.68rem; padding: 1px 5px; border-radius: 4px; background: #fef3c7; color: #92400e; font-weight: 700; margin-right: 4px;">🔬 学科奠基</span>`;
        } else if (item.action === 'REQUEST_BASELINE_PAPERS') {
          actionBadge = `<span style="font-size: 0.68rem; padding: 1px 5px; border-radius: 4px; background: #ede9fe; color: #6d28d9; font-weight: 700; margin-right: 4px;">🤝 基线委托</span>`;
        } else if (item.action === 'DELIVER_BASELINE_EVIDENCE') {
          actionBadge = `<span style="font-size: 0.68rem; padding: 1px 5px; border-radius: 4px; background: #dcfce7; color: #15803d; font-weight: 700; margin-right: 4px;">🌐 基线交付</span>`;
        } else if (item.action === 'SUBMIT_INITIAL_DRAFT') {
          actionBadge = `<span style="font-size: 0.68rem; padding: 1px 5px; border-radius: 4px; background: #e0e7ff; color: #3730a3; font-weight: 700; margin-right: 4px;">📄 提交初稿</span>`;
        } else if (item.action === 'CRITIQUE_AND_REQUIRE_REVISION' || item.action === 'DIALECTICAL_INQUIRY_AND_PRUNING') {
          actionBadge = `<span style="font-size: 0.68rem; padding: 1px 5px; border-radius: 4px; background: #fee2e2; color: #991b1b; font-weight: 700; margin-right: 4px;">🧐 辩证质询</span>`;
        } else if (item.action === 'APPROVE_INITIAL') {
          actionBadge = `<span style="font-size: 0.68rem; padding: 1px 5px; border-radius: 4px; background: #d1fae5; color: #065f46; font-weight: 700; margin-right: 4px;">审核未要求修订</span>`;
        } else if (item.action === 'SUBMIT_REVISED_DRAFT' || item.action === 'REFLECTIVE_RECONSTRUCTION') {
          actionBadge = `<span style="font-size: 0.68rem; padding: 1px 5px; border-radius: 4px; background: #fef3c7; color: #b45309; font-weight: 700; margin-right: 4px;">✍️ 反思重构</span>`;
        } else if (item.action === 'CONSENSUS_ACCEPTED') {
          actionBadge = `<span style="font-size: 0.68rem; padding: 1px 5px; border-radius: 4px; background: #d1fae5; color: #065f46; font-weight: 700; margin-right: 4px;">⚖️ 终审共识</span>`;
        }

        return `
        <li style="margin-bottom: 6px; padding: 5px 8px; background: rgba(255,255,255,0.75); border-radius: 4px; border-left: 3px solid #10b981; line-height: 1.45;">
          <div style="display: flex; align-items: center; gap: 4px; flex-wrap: wrap; margin-bottom: 2px;">
            ${roundBadge}
            ${actionBadge}
            <span style="font-weight: 700; color: #065f46;">${escapeHtml(item.sender || 'Agent')}</span>
            <span style="color: #94a3b8; font-size: 0.7rem;">➔</span>
            <span style="font-weight: 700; color: #0284c7;">${escapeHtml(item.receiver || 'Peer')}</span>
          </div>
          <div style="color: var(--color-ink-regular, #334155);">${escapeHtml(item.summary || '')}</div>
        </li>
        `;
      }).join('');

      multiAgentTrackHtml = `
        <details class="multi-agent-track-details" style="margin-top: 8px; font-size: 0.74rem; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 6px 10px;">
          <summary style="cursor: pointer; font-weight: 700; color: #15803d; user-select: none; display: flex; align-items: center; gap: 6px;">
            <span>👥 查看多智能体执行记录 (${data.agent_dialogue_history.length} 次席位交互)</span>
          </summary>
          <ul style="margin: 8px 0 0 0; padding-left: 0; list-style: none;">
            ${items}
          </ul>
        </details>
      `;
    }

    msgDiv.innerHTML = `
      <div class="msg-identity" aria-hidden="true"><img src="img/agent_avatar.jpg?v=2026" alt="羽若" class="msg-avatar-img"></div>
      <div class="msg-card">
        <div class="message-meta-line" style="margin-bottom: 8px; font-size: 0.72rem; color: var(--color-ink-muted); display: flex; align-items: center; gap: 8px;">
          <span style="font-weight: 800; color: var(--color-pine-jade);">${agentName}</span>
          <span style="background: var(--color-pine-soft); padding: 2px 8px; border-radius: 999px; border: 1px solid var(--color-silk-border); font-weight: 700;">${intentTag}</span>
        </div>
        <div class="markdown-body-text">${formattedText}</div>
        ${evidenceTagsHtml}
        ${evidenceSlateHtml}
        ${dialecticalNotesHtml}
        ${multiAgentTrackHtml}
        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 8px;">
          ${audioPlayBtnHtml}
        </div>
        ${mindmapBtnHtml}
      </div>
    `;

    chatFeed.appendChild(msgDiv);
    renderMermaidBlocks(msgDiv);
    renderChartBlocks(msgDiv);
    chatFeed.scrollTop = chatFeed.scrollHeight;

    // 绑定思维导图展开/折叠
    const toggleMindmapBtn = msgDiv.querySelector('.btn-toggle-msg-mindmap');
    if (toggleMindmapBtn && mindmapMd) {
      toggleMindmapBtn.addEventListener('click', () => {
        const card = msgDiv.querySelector('.msg-mindmap-card');
        const svg = msgDiv.querySelector('.msg-markmap-svg');
        if (card.style.display === 'none') {
          card.style.display = 'block';
          toggleMindmapBtn.textContent = '🗺️ 折叠本节知识导图';
          renderMarkmapTree(svg, mindmapMd);
        } else {
          card.style.display = 'none';
          toggleMindmapBtn.textContent = '🗺️ 展开本节深层知识导图';
        }
      });
    }

    // 绑定投射至案头
    const syncToDeskBtn = msgDiv.querySelector('.btn-sync-to-desk');
    if (syncToDeskBtn && mindmapMd) {
      syncToDeskBtn.addEventListener('click', async () => {
        activateTab('tab-mindmap');
        renderAnswerMindmap(mindmapMd, data.topicTitle || '当前研讨深度导图', (data.retrieved_evidence && data.retrieved_evidence[0]) ? data.retrieved_evidence[0].doc_name : '');
        if (turnIndex) {
          scrollToChatTurn(turnIndex);
        }
      });
    }

    // 绑定导出到 Obsidian
    const exportObsidianBtn = msgDiv.querySelector('.btn-export-msg-obsidian');
    if (exportObsidianBtn) {
      exportObsidianBtn.addEventListener('click', async () => {
        exportObsidianBtn.disabled = true;
        exportObsidianBtn.textContent = '⏳ 正在导出...';
        try {
          const resp = await fetch('/api/export/obsidian', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId, export_canvas: true })
          });
          const resJson = await resp.json();
          if (resp.ok && resJson.status === 'success') {
            exportObsidianBtn.textContent = '✅ 已存入 Obsidian Vault';
            alert(`🎉 导出成功！\n\n📁 Obsidian Vault 路径: ${resJson.vault_path}\n📄 双链笔记: ${resJson.note_path}\n🎨 Canvas 白板: ${resJson.canvas_path}`);
          } else {
            exportObsidianBtn.textContent = '❌ 导出失败';
            alert(`导出失败: ${resJson.detail || resJson.message}`);
          }
        } catch (e) {
          exportObsidianBtn.textContent = '❌ 网络异常';
          alert(`网络异常: ${e.message}`);
        } finally {
          setTimeout(() => {
            exportObsidianBtn.disabled = false;
            exportObsidianBtn.textContent = '💎 导出到 Obsidian';
          }, 4000);
        }
      });
    }

    // 绑定证据标签点击事件：点击切片标签直接联动右侧证据面板
    msgDiv.querySelectorAll('.evidence-badge').forEach((tag) => {
      tag.style.cursor = 'pointer';
      tag.title = '点击在右侧查看该切片详情与原图快照';
      tag.addEventListener('click', () => {
        // 激活右侧“证据与页码” Tab
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        const evTabBtn = document.querySelector('[data-tab="tab-evidence"]');
        const evTabContent = document.getElementById('tab-evidence');
        if (evTabBtn && evTabContent) {
          evTabBtn.classList.add('active');
          evTabContent.classList.add('active');
        }
        if (data.retrieved_evidence) {
          updateEvidencePanel(data.retrieved_evidence);
        }
      });
    });

    if (window.renderMathInElement) {
      const badgesContainer = msgDiv.querySelector('.evidence-badges');
      if (badgesContainer) {
        try {
          window.renderMathInElement(badgesContainer, {
            delimiters: [
              {left: '$', right: '$', display: false},
              {left: '\\(', right: '\\)', display: false}
            ],
            throwOnError: false,
            errorCallback: (message, error) => { console.warn('Badge math render warning:', message); }
          });
        } catch (error) { console.warn('Badge math error:', error); }
      }
    }

    const playBtn = msgDiv.querySelector('.btn-replay-audio');
    if (playBtn) {
      playBtn.addEventListener('click', async () => {
        // 若当前音频正在播放且正是本条消息，再次点击直接暂停
        if (globalAudio && !globalAudio.paused && playBtn.dataset.active === 'true') {
          globalAudio.pause();
          playBtn.textContent = '🔊 继续播放搭档语音';
          return;
        }

        // 若当前音频已暂停且正是本条消息，再次点击恢复播放
        if (globalAudio && globalAudio.paused && globalAudio.src && playBtn.dataset.active === 'true' && globalAudio.currentTime > 0 && !globalAudio.ended) {
          globalAudio.play();
          playBtn.textContent = '⏸️ 暂停播报';
          return;
        }

        document.querySelectorAll('.btn-replay-audio').forEach(b => {
          b.dataset.active = 'false';
          b.textContent = '🔊 播放搭档语音';
        });

        playBtn.dataset.active = 'true';
        playBtn.disabled = true;
        playBtn.textContent = '⏳ 正在调取语音...';

        try {
          await playTTS(playBtn.dataset.url, data.final_answer, playBtn);
        } catch (err) {
          console.error('Play audio error:', err);
          playBtn.textContent = '🔊 播放搭档语音';
        } finally {
          playBtn.disabled = false;
        }
      });
    }

  }

  async function playTTS(url, speechText = null, triggerBtn = null) {
    let targetUrl = url;

    // 若该条历史消息未生成独立音频或为旧版公共链接，动态调取按需合成
    if ((!targetUrl || targetUrl === '/api/audio/latest') && speechText) {
      try {
        // 过滤规则二：彻底过滤 Emoji 图标与机器人/意图标签
        const cleanSpeechText = speechText
          .replace(/[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, '')
          .replace(/🤖\s*[\w_]+/g, '')
          .replace(/🎯\s*意图[：:]\s*[\w_]+/g, '')
          .replace(/AI\s*回复/g, '')
          .trim();

        const res = await fetch('/api/tts/synthesize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: cleanSpeechText, voice: getCurrentVoice() }),
        });
        if (!res.ok) throw new Error('语音合成接口失败');
        const d = await res.json();
        if (!d.audio_url) throw new Error('语音合成未返回音频');
        if (d.audio_url) {
          targetUrl = d.audio_url;
          if (triggerBtn) triggerBtn.dataset.url = d.audio_url;
        }
      } catch (e) {
        throw new Error(`语音合成失败：${e.message}`);
      }
    }

    if (!targetUrl) {
      if (triggerBtn) {
        triggerBtn.textContent = '🔊 播放搭档语音';
        triggerBtn.disabled = false;
      }
      return;
    }

    if (globalAudio) {
      globalAudio.pause();
      globalAudio.currentTime = 0;
    }

    let currentSessionPlaybackId = Date.now();
    window.activePlaybackSessionId = currentSessionPlaybackId;

    setAssistantState('SPEAKING');
    globalAudio.src = targetUrl;

    // 自动切入右侧「🤝 伙伴座席」工作台
    if (typeof activateTab === 'function') {
      const sidebarRight = document.getElementById('sidebar-right');
      const btnToggleRight = document.getElementById('btn-toggle-right');
      if (sidebarRight && sidebarRight.classList.contains('collapsed')) {
        sidebarRight.classList.remove('collapsed');
        if (btnToggleRight) btnToggleRight.classList.add('active');
        if (typeof updateLayoutColumns === 'function') updateLayoutColumns();
      }
      activateTab('tab-avatar');
    }

    globalAudio.onplay = () => {
      setAssistantState('SPEAKING');
      if (window.digitalAvatar && typeof window.digitalAvatar.startSpeaking === 'function') {
        window.digitalAvatar.startSpeaking(globalAudio);
      }
      if (triggerBtn) {
        triggerBtn.textContent = '⏸️ 暂停播报';
        triggerBtn.disabled = false;
      }
    };

    globalAudio.onpause = () => {
      setAssistantState('IDLE');
      if (window.digitalAvatar && typeof window.digitalAvatar.stopSpeaking === 'function') {
        window.digitalAvatar.stopSpeaking();
      }
      if (triggerBtn) {
        triggerBtn.textContent = '🔊 恢复播报';
        triggerBtn.disabled = false;
      }
    };

    globalAudio.onended = () => {
      setAssistantState('IDLE');
      setOrbitNode(7, 'active', '✅ 播报完成');
      setHubStatus('可调度', false);
      if (window.digitalAvatar && typeof window.digitalAvatar.stopSpeaking === 'function') {
        window.digitalAvatar.stopSpeaking();
      }
      if (triggerBtn) {
        triggerBtn.textContent = '🔊 重新播报';
        triggerBtn.dataset.active = 'false';
        triggerBtn.disabled = false;
      }
    };

    globalAudio.onerror = (e) => {
      console.warn('Audio playback error:', e);
      setAssistantState('IDLE');
      setOrbitNode(7, 'error', '❌ 语音播放失败');
      setHubStatus('可调度', false);
      if (window.digitalAvatar && typeof window.digitalAvatar.stopSpeaking === 'function') {
        window.digitalAvatar.stopSpeaking();
      }
      if (triggerBtn) {
        triggerBtn.textContent = '🔊 播放搭档语音';
        triggerBtn.disabled = false;
      }
    };

    try {
      await globalAudio.play();
    } catch (e) {
      console.error('语音播放失败:', e);
      alert(`语音播放失败: ${e.message}`);
      if (triggerBtn) {
        triggerBtn.textContent = '🔊 播放搭档语音';
        triggerBtn.disabled = false;
      }
    }
  }

  window.playTTS = playTTS;

  // =========================================================================
  // 5. 知识库、会议总结与 WebSocket 实时语音
  // =========================================================================
  if (dropzoneTrigger && fileInput) {
    dropzoneTrigger.addEventListener('click', () => fileInput.click());
  }

  if (dropzone) {
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer && e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
    });
  }

  if (fileInput) {
    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length > 0) uploadFile(fileInput.files[0]);
    });
  }

  async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    uploadProgress.style.display = 'block';
    progressText.textContent = `正在切片入库 (${file.name})...`;

    try {
      const resp = await fetch('/api/documents/upload', { method: 'POST', body: formData });
      if (!resp.ok) throw new Error('Upload failed');
      const res = await resp.json();
      progressText.textContent = `✅ 入库成功！生成 ${res.chunks_count} 个切片`;
      setTimeout(() => { uploadProgress.style.display = 'none'; }, 2000);
      loadDocumentList();
    } catch (err) {
      progressText.textContent = `❌ 上传失败: ${err.message}`;
      setTimeout(() => { uploadProgress.style.display = 'none'; }, 3000);
    }
  }

  async function loadDocumentList() {
    try {
      const resp = await fetch('/api/documents');
      const data = await resp.json();
      const docs = data.documents || [];
      docCountBadge.textContent = `${docs.length} 篇`;
      if (docs.length === 0) {
        documentList.innerHTML = '<div class="empty-hint">暂未上传资料，支持随时投递专著、论文或全格式资料</div>';
        return;
      }
      if (docs.length > 0 && !currentActiveDocumentName) {
        currentActiveDocumentName = docs[0].filename;
      }
      documentList.innerHTML = docs.map(doc => `
        <div class="doc-item" data-filename="${escapeHtml(doc.filename)}" style="cursor: pointer; display: flex; align-items: center; justify-content: space-between;">
          <div style="display: flex; align-items: center; gap: 6px; flex: 1; overflow: hidden;">
            <span class="doc-name" title="${escapeHtml(doc.filename)}" style="flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(doc.filename)}</span>
            <span class="doc-badge ${doc.type}">${doc.type}</span>
          </div>
          <div style="display: flex; align-items: center; gap: 4px;">
            <a href="/api/documents/${encodeURIComponent(doc.filename)}/download" target="_blank" download="${escapeHtml(doc.filename)}" class="action-btn-mini btn-download-doc" title="下载或阅读该文献完整原卷" style="padding: 2px 6px; font-size: 0.65rem; text-decoration: none; color: inherit; display: inline-flex; align-items: center;" onclick="event.stopPropagation();">
              📥 下载
            </a>
            <button class="action-btn-mini btn-view-doc-mindmap" data-filename="${escapeHtml(doc.filename)}" title="在右侧展开该文献的全书深层知识脉络" style="padding: 2px 6px; font-size: 0.65rem;">
              🗺️ 脉络
            </button>
            <button class="action-btn-mini btn-delete-doc danger" data-filename="${escapeHtml(doc.filename)}" title="彻底删除该文献及对应向量切片" style="padding: 2px 5px; font-size: 0.65rem; color: #ef4444; border-color: rgba(239,68,68,0.3);">
              🗑️
            </button>
          </div>
        </div>
      `).join('');

      documentList.querySelectorAll('.btn-view-doc-mindmap').forEach(el => {
        el.addEventListener('click', (e) => {
          e.stopPropagation();
          const fn = el.dataset.filename;
          if (fn) {
            loadAndRenderBookMindmap(fn);
          }
        });
      });

      documentList.querySelectorAll('.btn-delete-doc').forEach(el => {
        el.addEventListener('click', async (e) => {
          e.stopPropagation();
          const fn = el.dataset.filename;
          if (!fn) return;
          if (!confirm(`确定要彻底删除文献《${fn}》吗？\n\n此操作将同时彻底清除向量数据库 (ChromaDB) 中的所有切片数据。`)) {
            return;
          }
          try {
            const delResp = await fetch(`/api/documents/${encodeURIComponent(fn)}`, { method: 'DELETE' });
            if (delResp.ok) {
              await loadDocumentList();
              console.log(`✅ 已成功删除文献: ${fn}`);
            } else {
              alert('删除文献失败，请检查网络或后端日志');
            }
          } catch (err) {
            console.error('Delete doc error:', err);
            alert('删除文献异常: ' + err.message);
          }
        });
      });

      documentList.querySelectorAll('.doc-item').forEach(el => {
        el.addEventListener('click', (e) => {
          if (e.target.closest('button')) return;
          const fn = el.dataset.filename;
          if (fn) {
            loadAndRenderBookMindmap(fn);
          }
        });
      });
    } catch (err) {
      console.error('Failed to load documents:', err);
      throw err;
    }
  }

  // 🌟 暴露全局经卷知识库与新建会话交互接口，支持天工雷达等跨模块联动
  window.createNewChat = createNewChat;
  window.loadDocumentList = loadDocumentList;
  window.fetchDocuments = loadDocumentList;
  window.loadAndRenderBookMindmap = loadAndRenderBookMindmap;
  window.setCurrentActiveDocument = function(fn) {
    if (!fn) return;
    currentActiveDocumentName = fn;
    currentFullBookMindmap = '';
    currentFullBookMindmapDoc = '';
  };
  window.switchMindmapScope = function(scope) {
    currentMindmapScope = scope;
    updateMindmapScopeUI();
  };

  // 🌟 东方雅致学术通知条 (Academic Toast)
  window.showAcademicToast = function(msg, type = 'info', duration = 4500) {
    let container = document.getElementById('academic-toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'academic-toast-container';
      container.style.cssText = 'position:fixed;bottom:24px;right:24px;z-index:99999;display:flex;flex-direction:column;gap:10px;pointer-events:none;';
      document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    const isSuccess = type === 'success';
    const isError = type === 'error';
    const bg = isSuccess ? '#184c43' : (isError ? '#8b2622' : '#253935');
    const border = isSuccess ? 'rgba(44,111,99,0.5)' : (isError ? 'rgba(182,76,72,0.5)' : 'rgba(179,120,48,0.4)');
    toast.style.cssText = `background:${bg};color:#ffffff;padding:10px 18px;border-radius:8px;font-size:13px;line-height:1.5;box-shadow:0 8px 24px rgba(0,0,0,0.22);border:1px solid ${border};display:flex;align-items:center;gap:8px;opacity:0;transform:translateY(12px);transition:all 0.3s cubic-bezier(0.16,1,0.3,1);pointer-events:auto;max-width:380px;font-family:system-ui,-apple-system,sans-serif;`;
    toast.innerHTML = `<span>${msg}</span>`;
    container.appendChild(toast);
    requestAnimationFrame(() => {
      toast.style.opacity = '1';
      toast.style.transform = 'translateY(0)';
    });
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(12px)';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  };

  function updateEvidencePanel(evidences) {
    if (!evidences || evidences.length === 0) {
      evidencePanelList.innerHTML = '<div class="empty-hint">🎋 当前轮次研讨依据先验理论与第一性原理推演，暂无引用的外部切片。<br><span style="font-size:0.75rem;color:var(--color-ink-muted);">若研讨结合专著教材或文献考据，将在此处实时联动段落快照与精确页码。</span></div>';
      return;
    }
    evidencePanelList.innerHTML = evidences.map((ev, i) => {
      let imgHtml = '';
      if (ev.image_url) {
        imgHtml = `
          <div class="evidence-img-box" onclick="openImageModal('${ev.image_url}', '《${escapeHtml(ev.doc_name || '文献快照')}》 ${escapeHtml(ev.location_label || '')}')">
            <img src="${ev.image_url}" alt="原图/快照" class="evidence-thumbnail" />
            <span class="img-zoom-tag">🔍 点击放大原图/页面快照</span>
          </div>
        `;
      }
      const scoreText = (typeof ev.score === 'number' && !isNaN(ev.score)) ? ` (匹配得分: ${ev.score.toFixed(4)})` : '';
      const docName = ev.doc_name || ev.title || '学术文献';
      const locLabel = ev.location_label || '';
      let linkHtml = '';
      const targetUrl = ev.pdf_url || ev.url;
      if (targetUrl) {
        linkHtml = `
          <div class="evidence-link-box" style="margin-top: 6px;">
            <a href="${escapeHtml(targetUrl)}" target="_blank" rel="noopener noreferrer" style="color: #0284c7; text-decoration: underline; font-size: 0.76rem; display: inline-flex; align-items: center; gap: 4px;">
              <span>🔗 官方文献/PDF直达</span>
            </a>
          </div>
        `;
      }
      return `
        <div class="evidence-card">
          <div class="evidence-card-header">
            [${i + 1}] 《${formatMathInText(docName)}》 ${escapeHtml(locLabel)}${scoreText}
          </div>
          <div class="evidence-card-body">${formatMathInText(ev.text || ev.summary || '')}</div>
          ${linkHtml}
          ${imgHtml}
        </div>
      `;
    }).join('');

    if (window.renderMathInElement && evidencePanelList) {
      try {
        window.renderMathInElement(evidencePanelList, {
          delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '\\[', right: '\\]', display: true},
            {left: '$', right: '$', display: false},
            {left: '\\(', right: '\\)', display: false}
          ],
          throwOnError: false,
          errorCallback: (message, error) => { console.warn('Evidence math render warning:', message); }
        });
      } catch (error) { console.warn('Evidence math error:', error); }
    }
  }

  window.openImageModal = function(src, title) {
    let modal = document.getElementById('image-preview-modal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'image-preview-modal';
      modal.className = 'image-modal';
      modal.innerHTML = `
        <div class="modal-backdrop" onclick="closeImageModal()"></div>
        <div class="modal-content">
          <div class="modal-header">
            <span id="modal-img-title" class="modal-title"></span>
            <button class="modal-close" onclick="closeImageModal()">✕</button>
          </div>
          <div class="modal-body">
            <img id="modal-img" src="" alt="原图大图" />
          </div>
        </div>
      `;
      document.body.appendChild(modal);
    }
    document.getElementById('modal-img-title').textContent = title || '原图与页面快照预览';
    document.getElementById('modal-img').src = src;
    modal.classList.add('active');
  };

  window.closeImageModal = function() {
    const modal = document.getElementById('image-preview-modal');
    if (modal) modal.classList.remove('active');
  };

  if (btnQuickSummary) {
    btnQuickSummary.addEventListener('click', async () => {
      setAssistantState('THINKING', '正在对当前会议流水进行 Map-Reduce 提炼与待办抽取...');
      highlightGraphStep('EXECUTING', 'MeetingSummarizer');
      try {
        const latestResp = await fetch('/api/meetings/latest');
        const latest = await latestResp.json();
        if (!latest.has_meeting || !latest.total_speeches) {
          throw new Error('暂无有效会议转写流水，请先选择音源并完成一段会议录音。');
        }
        const resp = await fetch('/api/meetings/summarize', { method: 'POST' });
        if (!resp.ok) {
          const errorData = await resp.json();
          throw new Error(errorData.detail || '会议纪要生成失败');
        }
        const data = await resp.json();
        const report = data.report;

        const tabActionBtn = document.querySelector('[data-tab="tab-actions"]');
        if (tabActionBtn) tabActionBtn.click();
        renderActionItems(report.action_items);
        appendMessage('agent', `### 📋 会议纪要已生成：${report.title}\n\n${report.overview}`);
        setAssistantState('IDLE');
        resetGraphNodes();
      } catch (e) {
        alert('生成纪要失败: ' + e.message);
        setAssistantState('IDLE');
        resetGraphNodes();
      }
    });
  }

  function renderActionItems(items) {
    if (!items || items.length === 0) {
      actionItemsList.innerHTML = '<div class="empty-hint">当前未提取到待办事项。</div>';
      return;
    }
    actionItemsList.innerHTML = items.map(item => `
      <div class="action-card priority-${item.priority === '高' ? 'high' : 'medium'}">
        <div class="action-top">
          <span class="priority-tag ${item.priority === '高' ? 'high' : 'medium'}">${item.priority}优先级</span>
          <span class="deadline-tag">截止: ${item.deadline}</span>
        </div>
        <p class="action-task">${escapeHtml(item.task)}</p>
        <div class="action-assignee">👤 责任人: ${item.assignee}</div>
      </div>
    `).join('');
  }

  const btnToggleMeetingRecord = document.getElementById('btn-toggle-meeting-record');
  const btnHeaderRecord = document.getElementById('btn-header-record');
  const headerRecordText = document.getElementById('header-record-text');
  const headerRecordIcon = document.getElementById('header-record-icon');
  const voiceCaptureMode = document.getElementById('voice-capture-mode');
  let isMeetingRecording = false;

  let meetingRecordRequestPending = false;

  function updateMeetingRecordingUI(recording, captureMode) {
    isMeetingRecording = recording;
    const sidebarText = btnToggleMeetingRecord?.querySelector('.record-text');
    if (btnToggleMeetingRecord) btnToggleMeetingRecord.classList.toggle('recording', recording);
    if (sidebarText) sidebarText.textContent = recording ? '录音中 (点击结束)' : '开启会议录音';
    if (btnHeaderRecord) {
      btnHeaderRecord.classList.toggle('active', recording);
      btnHeaderRecord.style.color = recording ? '#dc2626' : '';
      btnHeaderRecord.style.borderColor = recording ? '#f87171' : '';
    }
    if (headerRecordText) headerRecordText.textContent = recording ? '录制中 (点击停止)' : '开启会议录制';
    if (headerRecordIcon) headerRecordIcon.textContent = recording ? '🔴' : '🎙️';
    if (voiceCaptureMode) voiceCaptureMode.disabled = recording;
    if (recording) {
      const modeLabel = captureMode === 'system' ? '系统声音（腾讯会议/视频）' : '麦克风';
      setAssistantState('LISTENING', `已请求${modeLabel}监听，等待实时转写...`);
    } else {
      setAssistantState('IDLE');
    }
  }

  async function toggleMeetingRecording() {
    if (meetingRecordRequestPending) return;
    meetingRecordRequestPending = true;
    try {
      const nextRecording = !isMeetingRecording;
      const captureMode = voiceCaptureMode?.value;
      if (!['mic', 'system'].includes(captureMode)) throw new Error('未选择有效音源');
      const response = await fetch(nextRecording ? '/api/voice/start' : '/api/voice/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: nextRecording ? JSON.stringify({ mode: captureMode }) : undefined,
      });
      const data = await response.json();
      if (!response.ok || data.is_recording !== nextRecording) {
        throw new Error(data.detail || '录音状态未确认');
      }
      updateMeetingRecordingUI(nextRecording, captureMode);
    } catch (error) {
      alert(`会议录音操作失败: ${error.message}`);
    } finally {
      meetingRecordRequestPending = false;
    }
  }

  if (btnToggleMeetingRecord) {
    btnToggleMeetingRecord.addEventListener('click', toggleMeetingRecording);
  }
  if (btnHeaderRecord) {
    btnHeaderRecord.addEventListener('click', toggleMeetingRecording);
  }
  if (voiceCaptureMode) {
    voiceCaptureMode.addEventListener('change', (e) => {
      const mode = e.target.value;
      const modeText = mode === 'system' ? '🔊 系统声音（腾讯会议/视频内录）' : '🎤 麦克风';
      if (!isMeetingRecording) {
        appendStreamLog(new Date().toTimeString().split(' ')[0], '系统提示', `已切换至【${modeText}】通道。请点击上方【开启会议录音】或顶部【开启会议录制】按钮开始监听。`);
      }
    });
  }

  // WebSocket 与实时语音流
  function connectVoiceWebSocket() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${proto}//${window.location.host}/ws/voice_stream`;
    const ws = new WebSocket(wsUrl);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'error') {
          updateMeetingRecordingUI(false);
          alert(`实时语音已停止: ${data.message}`);
          return;
        }
        if (data.type === 'speech' && data.text) {
          appendStreamLog(data.timestamp, data.source, data.text);
          // 若开启了语音输入，将识别到的文本填入输入框，保留给用户确认和微调，不自动发送
          if (isRecording && data.text.trim()) {
            userInput.value = data.text.trim();
            userInput.style.height = 'auto';
            userInput.style.height = (userInput.scrollHeight) + 'px';
            userInput.focus();
            userInput.setSelectionRange(userInput.value.length, userInput.value.length);
          }
        }
      } catch (error) { throw error; }
    };
    ws.onclose = () => setTimeout(connectVoiceWebSocket, 3000);
  }

  function appendStreamLog(time, speaker, text) {
    const item = document.createElement('div');
    item.className = 'stream-item';
    item.innerHTML = `<span class="stream-time">[${time}]</span> <span class="stream-speaker">[${speaker}]</span><p class="stream-content">${escapeHtml(text)}</p>`;
    streamLogs.appendChild(item);
    streamLogs.scrollTop = streamLogs.scrollHeight;
  }

  // 智能格式化语音文本（中英文自然间隙排版与防粘连）
  function formatSpokenText(text) {
    if (!text) return '';
    return text
      .replace(/([\u4e00-\u9fa5])([a-zA-Z0-9])/g, '$1 $2')
      .replace(/([a-zA-Z0-9])([\u4e00-\u9fa5])/g, '$1 $2')
      .replace(/\s+/g, ' ')
      .trim();
  }

  // =========================================================================
  // 本地 OpenAI 官方原生 Whisper Large-v3 高保真语音研讨录音引擎 (MediaRecorder 驱动)
  // 彻底废弃易丢字、易截断、吞英文词的 Chrome 远程 Web Speech，全链路交由本地顶级 Whisper
  // =========================================================================
  let localMediaStream = null;
  let localMediaRecorder = null;
  let recordedAudioChunks = [];
  let isWhisperRecording = false;
  let whisperRecordTimer = null;
  let whisperRecordSeconds = 0;

  function formatDuration(sec) {
    const m = Math.floor(sec / 60).toString().padStart(2, '0');
    const s = (sec % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  }

  function updateWhisperRecordPrompt() {
    if (!isWhisperRecording) return;
    const durationStr = formatDuration(whisperRecordSeconds);
    userInput.placeholder = `🎙️ 正在录制学术语音 (已录制 ${durationStr}) · 支持长篇研讨与中英混杂，讲完后再次点击麦克风完成...`;
  }

  async function startWhisperRecording() {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('当前浏览器环境不支持麦克风音频采集，请检查权限或使用现代浏览器。');
        return;
      }
      localMediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true
        }
      });
      recordedAudioChunks = [];
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : (MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/ogg');
      
      localMediaRecorder = new MediaRecorder(localMediaStream, { mimeType });
      
      localMediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          recordedAudioChunks.push(event.data);
        }
      };

      localMediaRecorder.onstop = async () => {
        // 释放音轨硬件占用
        if (localMediaStream) {
          localMediaStream.getTracks().forEach(track => track.stop());
          localMediaStream = null;
        }

        if (recordedAudioChunks.length === 0) return;
        
        const audioBlob = new Blob(recordedAudioChunks, { type: mimeType });
        if (audioBlob.size < 1000) {
          userInput.placeholder = '向羽若垂询学术要旨、公式推导、专著论文研读，或探讨学术疑窦... (Enter 发送, Shift+Enter 换行)';
          setAssistantState('IDLE');
          return;
        }

        userInput.placeholder = '⚡ 本地 Whisper Large-v3 顶级引擎正在极速精准转录中，请稍候...';
        setAssistantState('THINKING', '本地 Whisper Large-v3 正在识别中英文长句及学术专有名词...');

        try {
          const formData = new FormData();
          formData.append('file', audioBlob, 'mic_recording.webm');
          const resp = await fetch('/api/voice/transcribe', {
            method: 'POST',
            body: formData
          });
          const result = await resp.json();
          if (result && result.status === 'success' && result.text) {
            const rawTranscribed = result.text.trim();
            const formatted = formatSpokenText(rawTranscribed);
            const currentVal = userInput.value.trim();
            const finalVal = currentVal ? `${currentVal} ${formatted}` : formatted;
            
            userInput.value = finalVal;
            userInput.style.height = 'auto';
            userInput.style.height = (userInput.scrollHeight) + 'px';
            userInput.focus();
            userInput.setSelectionRange(userInput.value.length, userInput.value.length);

            // 同步输出到左下角【临流听琴·课堂时间轴】
            const now = new Date().toTimeString().split(' ')[0];
            appendStreamLog(now, 'Whisper 麦克风录入', formatted);
            setAssistantState('IDLE', '已就绪');
          } else {
            console.warn('Whisper 转录返回异常:', result);
            setAssistantState('IDLE');
          }
        } catch (err) {
          console.error('Whisper 转录网络/服务异常:', err);
          alert(`转录失败: ${err.message}`);
          setAssistantState('IDLE');
        } finally {
          userInput.placeholder = '向羽若垂询学术要旨、公式推导、专著论文研读，或探讨学术疑窦... (Enter 发送, Shift+Enter 换行)';
        }
      };

      localMediaRecorder.start(250); // 每 250ms 切片收集一次音频数据
      isWhisperRecording = true;
      btnMic.classList.add('recording');
      whisperRecordSeconds = 0;
      updateWhisperRecordPrompt();
      setAssistantState('LISTENING', '正在录制高清人声：支持长篇学术阐述，停顿不掐断...');

      if (whisperRecordTimer) clearInterval(whisperRecordTimer);
      whisperRecordTimer = setInterval(() => {
        whisperRecordSeconds++;
        updateWhisperRecordPrompt();
      }, 1000);

    } catch (err) {
      console.error('无法启动麦克风录制:', err);
      alert('无法开启麦克风，请检查浏览器是否已授予音频权限。');
      btnMic.classList.remove('recording');
      isWhisperRecording = false;
      setAssistantState('IDLE');
    }
  }

  function stopWhisperRecording() {
    if (!isWhisperRecording) return;
    isWhisperRecording = false;
    btnMic.classList.remove('recording');
    if (whisperRecordTimer) {
      clearInterval(whisperRecordTimer);
      whisperRecordTimer = null;
    }
    if (localMediaRecorder && localMediaRecorder.state !== 'inactive') {
      try {
        localMediaRecorder.stop();
      } catch (error) { throw error; }
    }
  }

  if (btnMic) {
    btnMic.addEventListener('click', async () => {
      if (isWhisperRecording) {
        stopWhisperRecording();
      } else {
        startWhisperRecording();
      }
    });
  }

  // 绑定表单提交与按键事件 (文字模式: isVoice = false)
  const chatForm = document.getElementById('chat-form');
  if (chatForm) {
    chatForm.addEventListener('submit', (e) => {
      e.preventDefault();
      handleSendMessage(null, false);
    });
  }

  if (btnSend) {
    btnSend.addEventListener('click', (e) => {
      e.preventDefault();
      handleSendMessage(null, false);
    });
  }

  if (userInput) {
    userInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage(null, false);
      }
    });
  }

  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => handleSendMessage(chip.dataset.query, false));
  });

  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const targetContent = document.getElementById(btn.dataset.tab);
      if (targetContent) targetContent.classList.add('active');
      if (btn.dataset.tab === 'tab-mindmap' && mainMarkmapInstance) {
        setTimeout(() => mainMarkmapInstance.fit(), 100);
      }
    });
  });

  const btnMindmapFit = document.getElementById('btn-mindmap-fit');
  if (btnMindmapFit) {
    btnMindmapFit.addEventListener('click', () => {
      if (mainMarkmapInstance) mainMarkmapInstance.fit();
    });
  }

  // 顶部「导出至 Obsidian」按钮
  const btnExportObsidianHeader = document.getElementById('btn-export-obsidian');
  if (btnExportObsidianHeader) {
    btnExportObsidianHeader.addEventListener('click', async () => {
      if (!currentSessionId || currentSessionId === 'session_ai') {
        alert('请先新建或选择一个有效的学术会话，再导出到 Obsidian。');
        return;
      }
      btnExportObsidianHeader.disabled = true;
      const originalHtml = btnExportObsidianHeader.innerHTML;
      btnExportObsidianHeader.innerHTML = '<span class="icon">⏳</span> 正在导出至 Obsidian...';
      try {
        const resp = await fetch('/api/export/obsidian', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: currentSessionId, export_canvas: true })
        });
        const resJson = await resp.json();
        if (resp.ok && resJson.status === 'success') {
          btnExportObsidianHeader.innerHTML = '<span class="icon">✅</span> 已存入 Obsidian Vault';
          alert(`🎉 研讨全景知识库与白板已成功导出！\n\n📁 Vault 路径: ${resJson.vault_path}\n📄 双链笔记: ${resJson.note_path}\n🎨 Canvas 白板: ${resJson.canvas_path}`);
        } else {
          btnExportObsidianHeader.innerHTML = '<span class="icon">❌</span> 导出失败';
          alert(`导出失败: ${resJson.detail || resJson.message}`);
        }
      } catch (e) {
        btnExportObsidianHeader.innerHTML = '<span class="icon">❌</span> 网络异常';
        alert(`网络异常: ${e.message}`);
      } finally {
        setTimeout(() => {
          btnExportObsidianHeader.disabled = false;
          btnExportObsidianHeader.innerHTML = originalHtml;
        }, 4000);
      }
    });
  }

  // 绑定多层次作用域切换 (本轮精析 / 全书宏观 / 对话全景聚合)
  document.querySelectorAll('.scope-tab-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.preventDefault();
      currentMindmapScope = btn.dataset.scope;
      await updateMindmapScopeUI();
    });
  });

  // 绑定问答轮次下拉切换 (并实现双向联动：自动定位并高亮左侧对话轮次)
  const mindmapTurnSelect = document.getElementById('mindmap-turn-select');
  if (mindmapTurnSelect) {
    mindmapTurnSelect.addEventListener('change', async (e) => {
      currentActiveTurnIndex = parseInt(e.target.value, 10) || 0;
      currentMindmapScope = 'current';
      await updateMindmapScopeUI();

      // 双向联动：右侧切换轮次时，左侧对话区自动平滑滚动并呼吸高亮对应轮次
      const selectedTurnObj = sessionTurnsMindmaps[currentActiveTurnIndex];
      const targetTurnIndex = selectedTurnObj ? selectedTurnObj.turnIndex : (currentActiveTurnIndex + 1);
      scrollToChatTurn(targetTurnIndex);
    });
  }

  // 伸缩案头宽度 (360px -> 640px -> 880px -> 360px)
  const btnMindmapExpandPanel = document.getElementById('btn-mindmap-expand-panel');
  let currentPanelWidthState = 0; // 0: 360px, 1: 640px, 2: 880px
  const panelWidths = ['360px', '640px', '880px'];
  if (btnMindmapExpandPanel) {
    btnMindmapExpandPanel.addEventListener('click', (e) => {
      e.preventDefault();
      currentPanelWidthState = (currentPanelWidthState + 1) % panelWidths.length;
      const targetW = panelWidths[currentPanelWidthState];
      
      document.documentElement.style.setProperty('--sidebar-right-width', targetW);
      const mainLayout = document.getElementById('main-layout');
      if (mainLayout) {
        mainLayout.style.setProperty('--sidebar-right-width', targetW);
        mainLayout.style.gridTemplateColumns = `280px minmax(0, 1fr) auto ${targetW}`;
      }
      btnMindmapExpandPanel.textContent = currentPanelWidthState === 0 ? '📐 展宽案头' : currentPanelWidthState === 1 ? '📐 极宽视角' : '📐 恢复标准';
      setTimeout(() => {
        if (mainMarkmapInstance) mainMarkmapInstance.fit();
      }, 100);
    });
  }

  // 自由鼠标拖拽调节右侧案头宽度
  const sidebarResizer = document.getElementById('sidebar-resizer');
  if (sidebarResizer) {
    let isResizing = false;
    sidebarResizer.addEventListener('mousedown', (e) => {
      isResizing = true;
      sidebarResizer.classList.add('resizing');
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    });

    document.addEventListener('mousemove', (e) => {
      if (!isResizing) return;
      const newWidth = Math.max(300, Math.min(window.innerWidth - e.clientX, window.innerWidth * 0.7));
      document.documentElement.style.setProperty('--sidebar-right-width', `${newWidth}px`);
      const mainLayout = document.getElementById('main-layout');
      if (mainLayout) {
        mainLayout.style.gridTemplateColumns = `280px minmax(0, 1fr) auto ${newWidth}px`;
      }
    });

    document.addEventListener('mouseup', () => {
      if (isResizing) {
        isResizing = false;
        sidebarResizer.classList.remove('resizing');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        if (mainMarkmapInstance) mainMarkmapInstance.fit();
      }
    });
  }

  // 导出图片与复制大纲
  const btnMindmapExportPng = document.getElementById('btn-mindmap-export-png');
  if (btnMindmapExportPng) {
    btnMindmapExportPng.addEventListener('click', (e) => {
      e.preventDefault();
      const svg = document.getElementById('markmap-svg');
      exportSVGToPNG(svg, '学术知识思维导图.png');
    });
  }

  const btnMindmapCopyMd = document.getElementById('btn-mindmap-copy-md');
  if (btnMindmapCopyMd) {
    btnMindmapCopyMd.addEventListener('click', (e) => {
      e.preventDefault();
      copyActiveMindmapMarkdown();
    });
  }

  // 全屏大画卷弹窗控制
  const btnMindmapFullscreen = document.getElementById('btn-mindmap-fullscreen');
  const fullscreenModal = document.getElementById('mindmap-fullscreen-modal');
  const btnCloseMindmapModal = document.getElementById('btn-close-mindmap-modal');
  const modalMarkmapSvg = document.getElementById('modal-markmap-svg');
  const btnModalMindmapFit = document.getElementById('btn-modal-mindmap-fit');
  const btnModalExportPng = document.getElementById('btn-modal-export-png');
  const btnModalExportSvg = document.getElementById('btn-modal-export-svg');
  const modalMindmapTitle = document.getElementById('modal-mindmap-title');
  const modalMindmapBadge = document.getElementById('modal-mindmap-badge');

  async function openMindmapFullscreen() {
    if (!currentActiveMarkdown) {
      if (!currentActiveDocumentName) {
        alert('请先选择书籍或发起研讨提问');
        return;
      }
      await loadAndRenderBookMindmap(currentActiveDocumentName);
    }
    if (!currentActiveMarkdown) {
      alert('请先选择书籍或发起研讨提问');
      return;
    }
    const titleText = document.getElementById('mindmap-current-title')?.textContent || '经卷知识脉络';
    const badgeText = document.getElementById('mindmap-type-badge')?.textContent || '📚 经卷知识树';
    if (modalMindmapTitle) modalMindmapTitle.textContent = titleText;
    if (modalMindmapBadge) modalMindmapBadge.textContent = badgeText;

    if (fullscreenModal) {
      fullscreenModal.style.display = 'flex';
      fullscreenModal.classList.add('active');
    }
    setTimeout(() => {
      modalMarkmapInstance = renderMarkmapTree(modalMarkmapSvg, currentActiveMarkdown);
      if (modalMarkmapInstance) modalMarkmapInstance.fit();
    }, 60);
  }

  if (btnMindmapFullscreen) {
    btnMindmapFullscreen.addEventListener('click', (e) => {
      e.preventDefault();
      openMindmapFullscreen();
    });
  }
  if (btnCloseMindmapModal) {
    btnCloseMindmapModal.addEventListener('click', (e) => {
      e.preventDefault();
      if (fullscreenModal) {
        fullscreenModal.classList.remove('active');
        fullscreenModal.style.display = 'none';
      }
    });
  }
  if (fullscreenModal) {
    fullscreenModal.addEventListener('click', (e) => {
      if (e.target === fullscreenModal) {
        fullscreenModal.classList.remove('active');
        fullscreenModal.style.display = 'none';
      }
    });
  }
  if (btnModalMindmapFit) {
    btnModalMindmapFit.addEventListener('click', () => {
      if (modalMarkmapInstance) modalMarkmapInstance.fit();
    });
  }
  if (btnModalExportPng) {
    btnModalExportPng.addEventListener('click', () => {
      exportSVGToPNG(modalMarkmapSvg, '学术思维导图_高清全景.png');
    });
  }
  if (btnModalExportSvg) {
    btnModalExportSvg.addEventListener('click', () => {
      exportSVGToFile(modalMarkmapSvg, '学术思维导图_矢量全景.svg');
    });
  }

  const btnMindmapToggleMode = document.getElementById('btn-mindmap-toggle-mode');
  if (btnMindmapToggleMode) {
    btnMindmapToggleMode.addEventListener('click', () => {
      if (mindmapCurrentMode === 'answer' && currentFullBookMindmap) {
        mindmapCurrentMode = 'book';
        const titleEl = document.getElementById('mindmap-current-title');
        const badgeEl = document.getElementById('mindmap-type-badge');
        const cleanDocName = (currentActiveDocumentName || '文献').replace(/\.[^/.]+$/, '');
        if (titleEl) titleEl.textContent = `《${cleanDocName}》全景知识树`;
        if (badgeEl) badgeEl.textContent = '📚 全书宏观结构';
        currentActiveMarkdown = currentFullBookMindmap;
        const svg = document.getElementById('markmap-svg');
        mainMarkmapInstance = renderMarkmapTree(svg, currentFullBookMindmap);
      } else if (currentAnswerMindmap) {
        mindmapCurrentMode = 'answer';
        const titleEl = document.getElementById('mindmap-current-title');
        const badgeEl = document.getElementById('mindmap-type-badge');
        if (titleEl) titleEl.textContent = '当前研讨深层知识导图';
        if (badgeEl) badgeEl.textContent = '🧠 本题微观脉络';
        currentActiveMarkdown = currentAnswerMindmap;
        const svg = document.getElementById('markmap-svg');
        mainMarkmapInstance = renderMarkmapTree(svg, currentAnswerMindmap);
      } else if (currentFullBookMindmap) {
        loadAndRenderBookMindmap(currentActiveDocumentName);
      }
    });
  }

  function escapeHtml(value) {
    return String(value || '').replace(/[&<>'"]/g, (char) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
    }[char]));
  }

  // 073001 标准数学公式解析器
  function renderMathSource(source, displayMode = false) {
    let math = String(source || '').trim();
    if (!math) return '';
    // 规范化双反斜杠转义：仅针对字母开头的指令，严禁匹配中括号，避免破坏 '\\[2mm]' 行高间距
    math = math.replace(/\\\\([a-zA-Z]+)/g, '\\$1');
    if (!window.katex) {
      return displayMode ? `<div class="katex-error">$$${escapeHtml(math)}$$</div>` : `<span class="katex-error">$${escapeHtml(math)}$</span>`;
    }
    try {
      return window.katex.renderToString(math, { displayMode, throwOnError: false, trust: false });
    } catch (e) {
      console.warn('KaTeX render fallback:', e, math);
      return displayMode ? `<div class="katex-error">$$${escapeHtml(math)}$$</div>` : `<span class="katex-error">$${escapeHtml(math)}$</span>`;
    }
  }

  // 通用文本与公式混合渲染器（用于文献标题、卡片摘要、标签与检索结果）
  function formatMathInText(text) {
    if (!text) return '';
    let src = String(text);
    // 规范化双反斜杠转义与连字符
    src = src.replace(/\\\\([a-zA-Z]+)/g, '\\$1').replace(/\\([-_])/g, '$1');
    const mathTokens = [];

    // 1. 提取 display math: $$...$$, \\[...\\]
    src = src.replace(/(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\])/g, (match) => {
      const id = `___MATH_DISP_${mathTokens.length}___`;
      let content = match;
      if (content.startsWith('$$') && content.endsWith('$$')) content = content.slice(2, -2);
      else if (content.startsWith('\\[') && content.endsWith('\\]')) content = content.slice(2, -2);
      mathTokens.push({ id, content, display: true });
      return id;
    });

    // 2. 提取 inline math: $...$, \\(...\\)
    src = src.replace(/((?<!\\)\$([^\$\n]+?)(?<!\\)\$|\\\(([\s\S]*?)\\\))/g, (match, full, dollarContent, parenContent) => {
      const id = `___MATH_INL_${mathTokens.length}___`;
      const content = dollarContent !== undefined ? dollarContent : parenContent;
      mathTokens.push({ id, content, display: false });
      return id;
    });

    // 3. 转义非公式部分的 HTML 字符
    src = escapeHtml(src);
    // 将换行转换为 <br/>
    src = src.replace(/\n/g, '<br/>');

    // 4. 还原并渲染数学公式
    mathTokens.forEach(({ id, content, display }) => {
      const rendered = renderMathSource(content, display);
      src = src.replace(id, rendered);
    });

    return src;
  }

  // 复制代码块
  window.copyCodeToClipboard = function(btn) {
    const codeBlock = btn.closest('.code-box')?.querySelector('code');
    if (!codeBlock) return;
    const text = codeBlock.innerText;
    navigator.clipboard.writeText(text).then(() => {
      const orig = btn.innerText;
      btn.innerText = '✅ 已复制';
      setTimeout(() => { btn.innerText = orig; }, 2000);
    });
  };

  function formatInlineText(text) {
    if (!text) return '';
    let res = escapeHtml(text);
    // 加粗 **text** 或 __text__ (使用单词边界，避免误匹配标识符)
    res = res.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    res = res.replace(/\b__(.+?)__\b/g, '<strong>$1</strong>');
    // 删除线 ~~text~~
    res = res.replace(/~~(.+?)~~/g, '<del>$1</del>');
    // 斜体 *text* 或 _text_
    res = res.replace(/(?<!\*)\*([^\*\n]+?)\*(?!\*)/g, '<em>$1</em>');
    res = res.replace(/\b_([^_]+?)_\b/g, '<em>$1</em>');
    // Markdown 链接 [text](url)
    res = res.replace(/\[([^\]]+)\]\((https?:\/\/[^\s\)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="academic-link">$1</a>');
    return res;
  }

  // 行内代码与嵌入数学公式的智能渲染器 (解决标签中包裹公式反斜杠未渲染问题)
  function renderInlineCode(rawCode) {
    if (!rawCode) return '';
    let code = String(rawCode).replace(/\\\\([a-zA-Z]+)/g, '\\$1').replace(/\\([-_])/g, '$1').trim();

    // 1. 显式定界符包裹的纯公式: `\[\mathbb{C}P^{N-1}\]`, `\(\theta\)`, `$\theta$`
    if (/^\\\[([\s\S]*?)\\\]$/.test(code)) {
      const math = code.slice(2, -2).trim();
      return `<code class="inline-code inline-code-math">${renderMathSource(math, false)}</code>`;
    }
    if (/^\\\(([\s\S]*?)\\\)$/.test(code)) {
      const math = code.slice(2, -2).trim();
      return `<code class="inline-code inline-code-math">${renderMathSource(math, false)}</code>`;
    }
    if (/^\$([^\$\n]+)\$$/.test(code)) {
      const math = code.slice(1, -1).trim();
      return `<code class="inline-code inline-code-math">${renderMathSource(math, false)}</code>`;
    }

    // 2. 内部嵌套行内公式: 如 `\(\theta\)-vacua` 或 `$\theta$-vacua`
    if (/(\\\([\s\S]*?\\\)|(?<!\\)\$[^\$\n]+?\$)/.test(code)) {
      let rendered = code.replace(/\\\(([\s\S]*?)\\\)/g, (m, f) => renderMathSource(f.trim(), false));
      rendered = rendered.replace(/(?<!\\)\$([^\$\n]+?)\$/g, (m, f) => renderMathSource(f.trim(), false));
      return `<code class="inline-code">${rendered}</code>`;
    }

    // 3. 裸 LaTeX 公式或带后缀 (如 `\theta-vacua`, `\mathbb{C}P^{N-1}`)
    if (/^\\([a-zA-Z]+)/.test(code) && (code.includes('^') || code.includes('_') || code.includes('{') || /^\\[a-zA-Z]+$/.test(code) || /^\\[a-zA-Z]+-/.test(code))) {
      const mHyphen = code.match(/^(\\[a-zA-Z]+)(-[a-zA-Z0-9_\-]+)$/);
      if (mHyphen) {
        return `<code class="inline-code">${renderMathSource(mHyphen[1], false)}${escapeHtml(mHyphen[2])}</code>`;
      }
      return `<code class="inline-code inline-code-math">${renderMathSource(code, false)}</code>`;
    }

    return `<code class="inline-code">${escapeHtml(code)}</code>`;
  }

  function formatMarkdown(text) {
    if (!text) return '';
    let src = String(text);

    // 1. 提取多行代码块 ```lang ... ```
    const codeBlocks = [];
    src = src.replace(/```([a-zA-Z0-9_\-]*)\s*([\s\S]*?)```/g, (match, lang, code) => {
      const lowerLang = (lang || '').toLowerCase().trim();
      const rawCode = code.replace(/^\n+|\n+$/g, '');

      // Mermaid 顶会论文级机制与方法架构图
      if (lowerLang === 'mermaid') {
        const id = `@@CODEBLOCK_${codeBlocks.length}@@`;
        codeBlocks.push(`
          <div class="mermaid-diagram-card">
            <div class="figure-card-header">
              <span class="fig-title">FIGURE · 机制与方法架构示意</span>
              <button class="btn-fig-export" onclick="exportAcademicDiagram(this)" title="导出高清矢量图 (SVG)">🖼️ 导出矢量图</button>
            </div>
            <div class="mermaid-flowchart mermaid" data-raw-code="${escapeHtml(rawCode)}">${escapeHtml(rawCode)}</div>
          </div>
        `);
        return id;
      }

      // 学术数据可视化图表 (支持 chart, chartjs, echarts, echart)
      if (['chart', 'chartjs', 'echart', 'echarts', 'graph'].includes(lowerLang)) {
        const chartId = 'academic-chart-' + Math.random().toString(36).substring(2, 9);
        const id = `@@CODEBLOCK_${codeBlocks.length}@@`;
        codeBlocks.push(`
          <div class="academic-chart-box" id="${chartId}-box">
            <div class="chart-box-header">
              <span class="chart-tag">📊 学术数据可视化图表</span>
              <div class="chart-actions">
                <button class="btn-chart-action" onclick="exportAcademicChart('${chartId}')" title="导出图表为高清 PNG">🖼️ 导出图片</button>
                <button class="btn-chart-action" onclick="toggleAcademicChartData('${chartId}')" title="查看/隐藏原始配置数据">📋 原始数据</button>
              </div>
            </div>
            <div class="chart-canvas-container">
              <canvas id="${chartId}" class="academic-chart-canvas"></canvas>
            </div>
            <div class="chart-raw-json-panel" id="${chartId}-json" style="display: none;">
              <pre class="chart-raw-json-code"><code>${escapeHtml(rawCode)}</code></pre>
            </div>
          </div>
        `);
        return id;
      }

      // text / plaintext / pseudocode 等非可执行代码：去掉代码框壳，渲染为格式化正文段落
      if (['text', 'plaintext', 'pseudocode', 'algorithm', 'txt', ''].includes(lowerLang)) {
        const id = `@@CODEBLOCK_${codeBlocks.length}@@`;
        const lines = rawCode.split('\n').map(l => escapeHtml(l)).join('<br/>');
        codeBlocks.push(`<div class="text-block-unwrapped">${lines}</div>`);
        return id;
      }

      const id = `@@CODEBLOCK_${codeBlocks.length}@@`;
      const displayLang = (lang || 'code').toUpperCase();
      const escaped = escapeHtml(rawCode);
      codeBlocks.push(`
        <div class="code-box">
          <div class="code-box-header">
            <span class="code-lang-tag">${escapeHtml(displayLang)}</span>
            <button class="btn-copy-code" onclick="copyCodeToClipboard(this)">📋 复制代码</button>
          </div>
          <pre><code class="language-${escapeHtml(lang || 'text')}">${escaped}</code></pre>
        </div>
      `);
      return id;
    });

    // 2. 提取行内代码 `code` (支持行内代码内嵌数学公式与特殊符号自适应渲染)
    const inlineCodes = [];
    src = src.replace(/`([^`\n]+)`/g, (match, code) => {
      const id = `@@INLINECODE_${inlineCodes.length}@@`;
      inlineCodes.push(renderInlineCode(code));
      return id;
    });

    // 3. 提取数学公式 $$...$$, \\[...\\] 与 $...$, \\(...\\) (采用无冲突占位符，杜绝被加粗/斜体语法破坏)
    const mathBlocks = [];
    // Display Math
    src = src.replace(/(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\])/g, (match) => {
      const id = `@@MATHBLOCK_${mathBlocks.length}@@`;
      let content = match;
      if (content.startsWith('$$') && content.endsWith('$$')) {
        content = content.slice(2, -2);
      } else if (content.startsWith('\\[') && content.endsWith('\\]')) {
        content = content.slice(2, -2);
      }
      mathBlocks.push(`<div class="math-display-wrap">${renderMathSource(content, true)}</div>`);
      return id;
    });

    // Inline Math
    src = src.replace(/((?<!\\)\$([^\$\n]+?)(?<!\\)\$|\\\(([\s\S]*?)\\\))/g, (match, full, dollarContent, parenContent) => {
      const id = `@@MATHBLOCK_${mathBlocks.length}@@`;
      const content = dollarContent !== undefined ? dollarContent : parenContent;
      mathBlocks.push(renderMathSource(content, false));
      return id;
    });

    // 仅在代码块与数学公式已被安全隔离提取后，再将非公式文本中字面量换行符转换（排除了反斜杠后跟英文字母的 LaTeX 指令）
    src = src.replace(/\\n(?![a-zA-Z])/g, '\n');

    // 4. 解析表格与行级 Markdown
    const lines = src.split('\n');
    const htmlLines = [];
    let inList = false;
    let listType = 'ul';

    const isTableRow = (line) => line.includes('|') && line.trim().length > 1;
    const isSeparatorRow = (line) => /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
    const cells = (line) => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim());

    for (let i = 0; i < lines.length; i++) {
      const rawLine = lines[i];
      const trimmed = rawLine.trim();

      // 表格处理
      if (i + 1 < lines.length && isTableRow(rawLine) && isSeparatorRow(lines[i + 1])) {
        if (inList) { htmlLines.push(`</${listType}>`); inList = false; }
        const headers = cells(rawLine);
        const rows = [];
        i += 2;
        while (i < lines.length && isTableRow(lines[i]) && !isSeparatorRow(lines[i])) {
          rows.push(cells(lines[i]));
          i += 1;
        }
        i -= 1;
        const thHtml = headers.map(h => `<th>${formatInlineText(h)}</th>`).join('');
        const trHtml = rows.map(r => `<tr>${headers.map((_, idx) => `<td>${formatInlineText(r[idx] || '')}</td>`).join('')}</tr>`).join('');
        htmlLines.push(`<div class="rich-table-wrap"><table class="rich-table"><thead><tr>${thHtml}</tr></thead><tbody>${trHtml}</tbody></table></div>`);
        continue;
      }

      // 空行
      if (!trimmed) {
        if (inList) { htmlLines.push(`</${listType}>`); inList = false; }
        continue;
      }

      // 分隔线 (---, ***, ___)
      if (/^(\-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
        if (inList) { htmlLines.push(`</${listType}>`); inList = false; }
        htmlLines.push('<hr class="academic-hr"/>');
        continue;
      }

      // 标题 (支持 h1 ~ h6 全层级)
      if (trimmed.startsWith('###### ')) {
        if (inList) { htmlLines.push(`</${listType}>`); inList = false; }
        htmlLines.push(`<h6>${formatInlineText(trimmed.slice(7))}</h6>`);
        continue;
      }
      if (trimmed.startsWith('##### ')) {
        if (inList) { htmlLines.push(`</${listType}>`); inList = false; }
        htmlLines.push(`<h5>${formatInlineText(trimmed.slice(6))}</h5>`);
        continue;
      }
      if (trimmed.startsWith('#### ')) {
        if (inList) { htmlLines.push(`</${listType}>`); inList = false; }
        htmlLines.push(`<h4>${formatInlineText(trimmed.slice(5))}</h4>`);
        continue;
      }
      if (trimmed.startsWith('### ')) {
        if (inList) { htmlLines.push(`</${listType}>`); inList = false; }
        htmlLines.push(`<h3>${formatInlineText(trimmed.slice(4))}</h3>`);
        continue;
      }
      if (trimmed.startsWith('## ')) {
        if (inList) { htmlLines.push(`</${listType}>`); inList = false; }
        htmlLines.push(`<h2>${formatInlineText(trimmed.slice(3))}</h2>`);
        continue;
      }
      if (trimmed.startsWith('# ')) {
        if (inList) { htmlLines.push(`</${listType}>`); inList = false; }
        htmlLines.push(`<h1>${formatInlineText(trimmed.slice(2))}</h1>`);
        continue;
      }

      // 引用块
      if (trimmed.startsWith('> ')) {
        if (inList) { htmlLines.push(`</${listType}>`); inList = false; }
        htmlLines.push(`<blockquote>${formatInlineText(trimmed.slice(2))}</blockquote>`);
        continue;
      }

      // 任务列表 / 复选框 (- [ ] / - [x])
      const taskMatch = trimmed.match(/^[-*]\s+\[([ xX])\]\s+(.*)$/);
      if (taskMatch) {
        if (!inList || listType !== 'ul') {
          if (inList) htmlLines.push(`</${listType}>`);
          htmlLines.push('<ul class="task-list">');
          inList = true;
          listType = 'ul';
        }
        const isChecked = taskMatch[1].toLowerCase() === 'x';
        htmlLines.push(`<li class="task-list-item"><input type="checkbox" ${isChecked ? 'checked' : ''} disabled /> ${formatInlineText(taskMatch[2])}</li>`);
        continue;
      }

      // 无序列表 (- or *)
      const bulletMatch = trimmed.match(/^[-*]\s+(.*)$/);
      if (bulletMatch) {
        if (!inList || listType !== 'ul') {
          if (inList) htmlLines.push(`</${listType}>`);
          htmlLines.push('<ul>');
          inList = true;
          listType = 'ul';
        }
        htmlLines.push(`<li>${formatInlineText(bulletMatch[1])}</li>`);
        continue;
      }

      // 有序列表 (1. 2.)
      const numMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
      if (numMatch) {
        if (!inList || listType !== 'ol') {
          if (inList) htmlLines.push(`</${listType}>`);
          htmlLines.push('<ol>');
          inList = true;
          listType = 'ol';
        }
        htmlLines.push(`<li>${formatInlineText(numMatch[2])}</li>`);
        continue;
      }

      // 普通段落
      if (inList) { htmlLines.push(`</${listType}>`); inList = false; }
      htmlLines.push(`<p>${formatInlineText(trimmed)}</p>`);
    }

    if (inList) { htmlLines.push(`</${listType}>`); inList = false; }

    let result = htmlLines.join('\n');

    // 5. 还原占位符 (杜绝段落包裹污染，兼容历史旧格式)
    mathBlocks.forEach((mb, idx) => {
      result = result.replace(`<p>@@MATHBLOCK_${idx}@@</p>`, () => mb);
      result = result.replace(`@@MATHBLOCK_${idx}@@`, () => mb);
      result = result.replace(`<p>__MATH_BLOCK_${idx}__</p>`, () => mb);
      result = result.replace(`__MATH_BLOCK_${idx}__`, () => mb);
    });
    inlineCodes.forEach((ic, idx) => {
      result = result.replace(`@@INLINECODE_${idx}@@`, () => ic);
      result = result.replace(`__INLINE_CODE_${idx}__`, () => ic);
    });
    codeBlocks.forEach((cb, idx) => {
      result = result.replace(`<p>@@CODEBLOCK_${idx}@@</p>`, () => cb);
      result = result.replace(`@@CODEBLOCK_${idx}@@`, () => cb);
      result = result.replace(`<p>__CODE_BLOCK_${idx}__</p>`, () => cb);
      result = result.replace(`__CODE_BLOCK_${idx}__`, () => cb);
    });

    return result;
  }

  // 暴露问答窗口成熟的 Markdown 与数学公式渲染引擎供全局组件 (如天工雷达论文全文翻译) 统一调用
  window.formatMarkdown = formatMarkdown;
  window.renderMathSource = renderMathSource;
  window.formatInlineText = formatInlineText;

  const ACADEMIC_PALETTE = {
    backgrounds: [
      'rgba(24, 76, 67, 0.78)',    // 松柏墨绿
      'rgba(44, 74, 111, 0.78)',   // 黛蓝
      'rgba(184, 134, 11, 0.78)',  // 琥珀金
      'rgba(194, 79, 77, 0.78)',   // 胭红
      'rgba(107, 82, 128, 0.78)',  // 黛紫
      'rgba(45, 133, 117, 0.78)',  // 苍绿
      'rgba(202, 138, 4, 0.78)'    // 赭石
    ],
    borders: [
      '#184c43',
      '#2c4a6f',
      '#b8860b',
      '#c24f4d',
      '#6b5280',
      '#2d8575',
      '#ca8a04'
    ]
  };

  window.exportAcademicChart = function(chartId) {
    const canvas = document.getElementById(chartId);
    if (!canvas) return;
    try {
      const link = document.createElement('a');
      link.download = `academic_chart_${chartId}_${Date.now()}.png`;
      link.href = canvas.toDataURL('image/png', 1.0);
      link.click();
    } catch (e) {
      alert(`图表导出失败: ${e.message}`);
    }
  };

  window.toggleAcademicChartData = function(chartId) {
    const panel = document.getElementById(`${chartId}-json`);
    if (!panel) return;
    const isHidden = panel.style.display === 'none';
    panel.style.display = isHidden ? 'block' : 'none';
  };

  window.exportAcademicDiagram = function(btn) {
    const card = btn.closest('.mermaid-diagram-card');
    if (!card) return;
    const svg = card.querySelector('svg');
    if (!svg) return;
    try {
      const svgData = new XMLSerializer().serializeToString(svg);
      const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.download = `academic_figure_${Date.now()}.svg`;
      link.href = url;
      link.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert(`流程图导出失败: ${e.message}`);
    }
  };

  async function renderMermaidBlocks(container) {
    if (!container) return;
    if (!window.mermaid) {
      let waited = 0;
      while (!window.mermaid && waited < 3000) {
        await new Promise(r => setTimeout(r, 100));
        waited += 100;
      }
    }
    if (!window.mermaid) return;

    const nodes = Array.from(container.querySelectorAll('.mermaid-flowchart:not([data-processed="true"])'));
    if (!nodes.length) return;

    try {
      window.mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'loose',
        theme: 'base',
        flowchart: {
          htmlLabels: true,
          curve: 'basis',
          nodeSpacing: 50,
          rankSpacing: 65,
          padding: 18
        },
        themeVariables: {
          fontFamily: '-apple-system, BlinkMacSystemFont, "PingFang SC", "Noto Sans SC", "Segoe UI", Roboto, sans-serif',
          fontSize: '12.5px',
          primaryColor: '#ffffff',
          primaryBorderColor: '#94a3b8',
          primaryTextColor: '#0f172a',
          lineColor: '#64748b',
          secondaryColor: '#f8fafc',
          secondaryBorderColor: '#cbd5e1',
          secondaryTextColor: '#334155',
          tertiaryColor: '#ffffff',
          tertiaryBorderColor: '#cbd5e1',
          tertiaryTextColor: '#475569',
          clusterBkg: '#f8fafc',
          clusterBorder: '#cbd5e1',
          edgeLabelBackground: '#ffffff',
          titleColor: '#0f172a'
        }
      });
      for (const node of nodes) {
        node.setAttribute('data-processed', 'true');
        try {
          const id = 'mermaid-' + Math.random().toString(36).substring(2, 9);
          let rawCode = node.getAttribute('data-raw-code') || node.textContent;
          // 若未包含显式指令，注入高层间距配置，确保连线标签有充裕纵向呼吸空间 (rankSpacing 85px)
          if (!rawCode.includes('%%{init') && !rawCode.startsWith('---')) {
            rawCode = `%%{init: {'flowchart': {'rankSpacing': 85, 'nodeSpacing': 50, 'curve': 'basis'}}}%%\n` + rawCode;
          }
          const { svg } = await window.mermaid.render(id, rawCode);
          node.innerHTML = svg;

          // 🌟 核心修复：SVG 依靠 DOM 顺序决定视觉层叠优先级 (后渲染覆盖先渲染)
          // Mermaid 默认先渲染 .edgeLabels，后渲染 .nodes，导致当连线紧凑时 Node 的白色方块遮挡连线文字下半部分！
          // 将 .edgeLabels 移动至 SVG 根节点的最末尾，确保连线标签卡片永远浮于最高层，绝不被下游方块裁剪或遮盖！
          const svgEl = node.querySelector('svg');
          if (svgEl) {
            const edgeLabels = svgEl.querySelector('.edgeLabels');
            if (edgeLabels && edgeLabels.parentNode) {
              edgeLabels.parentNode.appendChild(edgeLabels);
            }
          }
        } catch (err) {
          node.textContent = `流程图渲染失败: ${err.message}`;
          node.classList.add('render-error');
        }
      }
    } catch (error) {
      throw error;
    }
  }

  async function renderChartBlocks(container) {
    if (!container) return;
    if (!window.Chart) {
      let waited = 0;
      while (!window.Chart && waited < 3000) {
        await new Promise(r => setTimeout(r, 100));
        waited += 100;
      }
    }
    if (!window.Chart) return;

    const boxes = Array.from(container.querySelectorAll('.academic-chart-box:not([data-processed="true"])'));
    if (!boxes.length) return;

    boxes.forEach(box => {
      box.setAttribute('data-processed', 'true');
      const canvas = box.querySelector('canvas.academic-chart-canvas');
      const rawPre = box.querySelector('.chart-raw-json-code code') || box.querySelector('.chart-raw-json-code');
      if (!canvas || !rawPre) return;

      try {
        let rawText = rawPre.textContent.trim();
        rawText = rawText.replace(/^```[a-zA-Z]*\n?/, '').replace(/\n?```$/, '').trim();
        let config = JSON.parse(rawText);

        if (!config || typeof config !== 'object' || !config.type || !config.data) {
          throw new Error('图表缺少类型或数据，无法渲染');
        }
        if (!config.options) config.options = {};

        // 智能为 datasets 补充雅致的学术配色
        if (Array.isArray(config.data.datasets)) {
          config.data.datasets.forEach((ds, idx) => {
            if (!ds.backgroundColor) {
              if (['pie', 'doughnut', 'polarArea'].includes(config.type)) {
                ds.backgroundColor = ACADEMIC_PALETTE.backgrounds;
                ds.borderColor = ACADEMIC_PALETTE.borders;
                ds.borderWidth = 1;
              } else {
                const bg = ACADEMIC_PALETTE.backgrounds[idx % ACADEMIC_PALETTE.backgrounds.length];
                const border = ACADEMIC_PALETTE.borders[idx % ACADEMIC_PALETTE.borders.length];
                ds.backgroundColor = bg;
                ds.borderColor = border;
                ds.borderWidth = 1.5;
              }
            }
          });
        }

        config.options.responsive = true;
        config.options.maintainAspectRatio = false;

        new window.Chart(canvas.getContext('2d'), config);
      } catch (e) {
        console.warn('Academic chart parsing/rendering error:', e);
        const message = document.createElement('p');
        message.className = 'render-error';
        message.textContent = `图表渲染失败: ${e.message}`;
        box.appendChild(message);
      }
    });
  }

  function renderLatexEquations(domElement) {
    // 实时由 formatMarkdown 完成 KaTeX 渲染
  }

  async function loadSystemConfig() {
    try {
      const resp = await fetch('/api/config');
      if (!resp.ok) throw new Error('配置接口失败');
      if (resp.ok) {
        const cfg = await resp.json();
        const pill = document.getElementById('pill-model-name');
        if (pill && cfg.model_name) {
          pill.innerHTML = `<span class="dot gold"></span> ${escapeHtml(cfg.model_name)}`;
          pill.title = `当前驱动大模型: ${cfg.model_name}`;
        }
      }
    } catch (_) {
      const pill = document.getElementById('pill-model-name');
      if (pill) pill.textContent = '模型配置加载失败';
    }
  }

  // 初始化
  loadSystemConfig();
  loadSessions();
  loadDocumentList();
  connectVoiceWebSocket();
});
