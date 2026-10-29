/**
 * 云海星津 · 天工雷达 (TianGong Academic Radar Engine)
 * 东方文人雅致云山纸墨交互系统
 * 特性:
 * 1. 严格多学科隔离 (彻底修复 arXivDaily 子学科穿透 Bug);
 * 2. 万象星图子学科水墨分布画卷动态渲染;
 * 3. 闭环联动: 一键藏之石渠 (入本地向量库) + 兰台问道 (召羽若研讨).
 */

const RADAR_DISCIPLINES = [
  { key: "ALL", name_zh: "全部前沿", code: "ALL", icon: "🌐" },
  { key: "CS", name_zh: "计算机科学", code: "CS", icon: "💻" },
  { key: "PHYSICS", name_zh: "物理学", code: "PHYS", icon: "🌌" },
  { key: "MATH", name_zh: "数学", code: "MATH", icon: "📐" },
  { key: "Q-BIO", name_zh: "定量生物", code: "Q-BIO", icon: "🧬" },
  { key: "EESS", name_zh: "电气工程", code: "EESS", icon: "⚡" },
  { key: "STAT", name_zh: "统计学", code: "STAT", icon: "📊" },
  { key: "Q-FIN", name_zh: "定量金融", code: "Q-FIN", icon: "📈" },
  { key: "ECON", name_zh: "经济学", code: "ECON", icon: "💰" },
];

class TianGongRadarController {
  constructor() {
    this.taxonomy = null;
    this.currentMajor = "CS";
    this.selectedSubCategories = [];
    this.currentPage = 1;
    this.pageSize = 25;
    this.currentQuery = "";
    this.searchIn = "all";
    this.sortBy = "submittedDate";

    this.initElements();
    this.bindEvents();
    this.loadTaxonomy();
  }

  initElements() {
    // 顶层 Tab
    this.tabDialogue = document.getElementById("tab-nav-dialogue");
    this.tabRadar = document.getElementById("tab-nav-radar");
    this.mainLayoutDialogue = document.getElementById("main-layout");
    this.mainLayoutRadar = document.getElementById("radar-layout");

    // 顶级学科导航条与子学科全览按钮
    this.topMajorNavBar = document.getElementById("radar-major-nav-bar");
    this.btnDistAll = document.getElementById("btn-dist-all");

    // 雷达控制组件
    this.subTreeContainer = document.getElementById("radar-sub-tree");
    this.btnClearSub = document.getElementById("btn-radar-clear-sub");
    this.inputQuery = document.getElementById("radar-input-query");
    this.btnSearch = document.getElementById("btn-radar-search");
    this.selectSearchIn = document.getElementById("radar-search-in");
    this.selectPageSize = document.getElementById("radar-page-size");
    this.autoIngestCheckbox = document.getElementById("radar-auto-ingest");
    this.radioSorts = document.getElementsByName("radar-sort");

    // 状态与分布容器
    this.activeMajorLabel = document.getElementById("radar-active-major-label");
    this.statusText = document.getElementById("radar-status-text");
    this.exprPill = document.getElementById("radar-expr-pill");
    this.distBarsContainer = document.getElementById("dist-bars-container");
    this.paperListContainer = document.getElementById("radar-paper-list");

    // 分页
    this.btnPrev = document.getElementById("btn-radar-prev");
    this.btnNext = document.getElementById("btn-radar-next");
    this.pageIndicator = document.getElementById("radar-page-indicator");
  }

  bindEvents() {
    // 视图切换
    if (this.tabDialogue && this.tabRadar) {
      this.tabDialogue.addEventListener("click", () => this.switchView("dialogue"));
      this.tabRadar.addEventListener("click", () => this.switchView("radar"));
    }

    // 检索按钮与回车
    if (this.btnSearch) {
      this.btnSearch.addEventListener("click", () => {
        this.currentPage = 1;
        this.doSearch();
      });
    }

    if (this.inputQuery) {
      this.inputQuery.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          this.currentPage = 1;
          this.doSearch();
        }
      });
    }

    // 排序单选
    if (this.radioSorts) {
      this.radioSorts.forEach((r) => {
        r.addEventListener("change", (e) => {
          this.sortBy = e.target.value;
          this.currentPage = 1;
          this.doSearch();
        });
      });
    }

    // 检索范围
    if (this.selectSearchIn) {
      this.selectSearchIn.addEventListener("change", (e) => {
        this.searchIn = e.target.value;
      });
    }

    // 每卷容量变更 (15 / 25 / 50 篇)
    if (this.selectPageSize) {
      this.selectPageSize.addEventListener("change", (e) => {
        this.pageSize = parseInt(e.target.value, 10) || 25;
        this.currentPage = 1;
        this.doSearch();
      });
    }

    // 研讨联动自动纳卷入阁持久化偏好
    if (this.autoIngestCheckbox) {
      const saved = localStorage.getItem("radar_auto_ingest_enabled");
      if (saved !== null) {
        this.autoIngestCheckbox.checked = (saved === "true");
      }
      this.autoIngestCheckbox.addEventListener("change", (e) => {
        localStorage.setItem("radar_auto_ingest_enabled", e.target.checked ? "true" : "false");
      });
    }

    // 清空细选 / 还原全览
    if (this.btnClearSub) {
      this.btnClearSub.addEventListener("click", () => {
        this.selectedSubCategories = [];
        this.renderSubTree();
        this.currentPage = 1;
        this.doSearch();
      });
    }

    if (this.btnDistAll) {
      this.btnDistAll.addEventListener("click", () => {
        this.selectedSubCategories = [];
        this.renderSubTree();
        this.currentPage = 1;
        this.doSearch();
      });
    }

    // 分页控制
    if (this.btnPrev) {
      this.btnPrev.addEventListener("click", () => {
        if (this.currentPage > 1) {
          this.currentPage -= 1;
          this.doSearch();
        }
      });
    }

    if (this.btnNext) {
      this.btnNext.addEventListener("click", () => {
        this.currentPage += 1;
        this.doSearch();
      });
    }
  }

  switchView(mode) {
    if (mode === "radar") {
      this.tabDialogue.classList.remove("active");
      this.tabRadar.classList.add("active");
      this.mainLayoutDialogue.style.display = "none";
      this.mainLayoutRadar.style.display = "grid";
    } else {
      this.tabRadar.classList.remove("active");
      this.tabDialogue.classList.add("active");
      this.mainLayoutRadar.style.display = "none";
      this.mainLayoutDialogue.style.display = "grid";
    }
  }

  async loadTaxonomy() {
    try {
      const resp = await fetch("/api/radar/taxonomy");
      const data = await resp.json();
      this.taxonomy = data.taxonomy;
      this.renderTopMajorNavBar();
      this.renderSubTree();
      // 初始加载一次默认大类文献
      this.doSearch();
    } catch (err) {
      console.error("加载学科分类典籍失败:", err);
    }
  }

  renderTopMajorNavBar() {
    if (!this.topMajorNavBar) return;
    this.topMajorNavBar.innerHTML = "";

    RADAR_DISCIPLINES.forEach((item) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `radar-major-tab-btn ${item.key === this.currentMajor ? "active" : ""}`;
      btn.innerHTML = `
        <span class="tab-icon">${item.icon}</span>
        <span class="tab-title">${item.name_zh}</span>
        <span class="tab-code">${item.code}</span>
      `;
      btn.addEventListener("click", () => {
        this.switchMajor(item.key);
      });
      this.topMajorNavBar.appendChild(btn);
    });
  }

  switchMajor(majorKey) {
    this.currentMajor = majorKey;
    this.selectedSubCategories = [];
    this.currentPage = 1;
    this.renderTopMajorNavBar();
    this.renderSubTree();
    this.doSearch();
  }

  renderSubTree() {
    if (!this.subTreeContainer) return;

    if (this.currentMajor === "ALL") {
      if (this.activeMajorLabel) {
        this.activeMajorLabel.innerHTML = `<span>🌐</span> <span>全部前沿 · 跨领域综合学术</span>`;
      }
      this.subTreeContainer.innerHTML = `
        <div class="empty-hint" style="padding:16px 8px;text-align:center;font-size:0.76rem;line-height:1.6;color:var(--color-ink-muted);">
          🌐 当前处于全学科综合前沿模式。<br>
          点击上方学科标签（如计算机科学、物理学、数学等）可立刻展开细分子学科。
        </div>
      `;
      return;
    }

    const item = RADAR_DISCIPLINES.find((d) => d.key === this.currentMajor);
    const majorInfo = this.taxonomy ? this.taxonomy[this.currentMajor] : null;
    const displayName = item ? item.name_zh : (majorInfo ? majorInfo.name_zh : this.currentMajor);
    const displayIcon = item ? item.icon : (majorInfo ? majorInfo.icon : "📜");

    if (this.activeMajorLabel) {
      this.activeMajorLabel.innerHTML = `<span>${displayIcon}</span> <span>${displayName} · ${this.currentMajor}</span>`;
    }

    this.subTreeContainer.innerHTML = "";
    if (!majorInfo) return;
    const groups = majorInfo.groups || {};

    Object.entries(groups).forEach(([groupKey, groupInfo]) => {
      const groupBlock = document.createElement("div");
      groupBlock.className = "sub-group-block";

      const title = document.createElement("div");
      title.className = "sub-group-title";
      title.textContent = `🎋 ${groupInfo.name_zh} (${groupInfo.arxiv_prefix})`;
      groupBlock.appendChild(title);

      const subCats = groupInfo.subcategories || {};
      Object.entries(subCats).forEach(([subCode, meta]) => {
        const label = document.createElement("label");
        label.className = "sub-item-checkbox";

        const chk = document.createElement("input");
        chk.type = "checkbox";
        chk.value = subCode;
        chk.checked = this.selectedSubCategories.includes(subCode);

        chk.addEventListener("change", (e) => {
          if (e.target.checked) {
            if (!this.selectedSubCategories.includes(subCode)) {
              this.selectedSubCategories.push(subCode);
            }
          } else {
            this.selectedSubCategories = this.selectedSubCategories.filter((c) => c !== subCode);
          }
          this.currentPage = 1;
          this.doSearch();
        });

        label.appendChild(chk);
        const txt = document.createElement("span");
        txt.innerHTML = `<strong>${subCode}</strong> · ${meta.name_zh}`;
        label.appendChild(txt);

        groupBlock.appendChild(label);
      });

      this.subTreeContainer.appendChild(groupBlock);
    });
  }

  doSearch(immediate = false) {
    if (this._searchDebounceTimer) {
      clearTimeout(this._searchDebounceTimer);
      this._searchDebounceTimer = null;
    }
    if (immediate) {
      this._executeSearch();
    } else {
      this._searchDebounceTimer = setTimeout(() => {
        this._executeSearch();
      }, 250);
    }
  }

  async _executeSearch() {
    this.currentQuery = this.inputQuery ? this.inputQuery.value.trim() : "";
    this.searchSeq = (this.searchSeq || 0) + 1;
    const reqSeq = this.searchSeq;

    // 1. 立即更新当前检索表达式预览，防止显示旧学科的标签（如 Q-FIN 残留在 ECON 视图中）
    if (this.exprPill) {
      if (this.selectedSubCategories.length > 0) {
        this.exprPill.textContent = this.selectedSubCategories.length === 1 
          ? `cat:${this.selectedSubCategories[0]}` 
          : `(${this.selectedSubCategories.map((c) => 'cat:' + c).join(' OR ')})`;
      } else if (this.currentMajor && this.currentMajor !== "ALL") {
        const majorPrefMap = {
          "CS": "cat:cs*",
          "PHYSICS": "cat:physics* / cond-mat* ...",
          "MATH": "cat:math*",
          "Q-BIO": "cat:q-bio*",
          "EESS": "cat:eess*",
          "STAT": "cat:stat*",
          "Q-FIN": "cat:q-fin*",
          "ECON": "cat:econ*",
        };
        this.exprPill.textContent = majorPrefMap[this.currentMajor] || `cat:${this.currentMajor.toLowerCase()}*`;
      } else {
        this.exprPill.textContent = "全学科宏观前沿流 (All Disciplines)";
      }
    }

    if (this.statusText) {
      this.statusText.textContent = "「墨砚微温 · 考据求索中...」";
    }

    // 2. 立即清空旧学科的论文列表，显示水墨优雅加载动画，彻底杜绝混杂错觉
    if (this.paperListContainer) {
      this.paperListContainer.innerHTML = `
        <div style="text-align:center;padding:56px 20px;color:var(--color-ink-muted);">
          <div style="font-size:2rem;margin-bottom:12px;opacity:0.85;">⏳</div>
          <div style="font-size:0.95rem;letter-spacing:1px;color:var(--color-primary-forest);">墨砚微温 · 考据求索中...</div>
          <div style="font-size:0.75rem;margin-top:6px;opacity:0.65;">正在探寻前沿卷册与细分子学科经脉分布</div>
        </div>
      `;
    }

    if (this.distBarsContainer) {
      this.distBarsContainer.innerHTML = `<span class="dist-loading-text">正在统计细分子学科分布...</span>`;
    }

    const payload = {
      query: this.currentQuery,
      major: this.currentMajor,
      sub_categories: this.selectedSubCategories,
      search_in: this.searchIn,
      sort_by: this.sortBy,
      start: (this.currentPage - 1) * this.pageSize,
      max_results: this.pageSize,
    };

    try {
      const resp = await fetch("/api/radar/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (reqSeq !== this.searchSeq) {
        return; // 忽略由于快速点击而过期的异步请求
      }
      if (data.status === "success" && data.data) {
        this.renderResults(data.data);
      } else {
        throw new Error(data.detail || "考据失败");
      }
    } catch (err) {
      if (reqSeq !== this.searchSeq) return;
      if (this.statusText) {
        this.statusText.textContent = `⚠️ 考据受阻: ${err.message}`;
      }
    }
  }

  renderResults(radarData) {
    const totalCount = radarData.total_results || radarData.total_fetched || 0;
    const currentBatchCount = (radarData.papers && radarData.papers.length) ? radarData.papers.length : (radarData.total_fetched || 0);
    const startIdx = totalCount > 0 ? (this.currentPage - 1) * this.pageSize + 1 : 0;
    const endIdx = totalCount > 0 ? Math.min(startIdx + currentBatchCount - 1, totalCount) : 0;

    if (this.statusText) {
      if (radarData.is_cached_snapshot) {
        this.statusText.textContent = `「🍃 经卷容灾快照 · 当前处于无缝离线快照模式 (本卷共 ${currentBatchCount} 篇)」`;
      } else if (totalCount > 0) {
        this.statusText.textContent = `「星津澄澈 · 历史卷册总库共 ${totalCount.toLocaleString()} 卷，当前展阅第 ${startIdx} ~ ${endIdx} 篇 (本卷共 ${currentBatchCount} 篇)」`;
      } else {
        this.statusText.textContent = "「星津微茫 · 暂无匹配卷册」";
      }
    }
    if (this.exprPill) {
      this.exprPill.textContent = radarData.query_expression || "All";
    }

    // 渲染水墨分布画卷 (Sub-category Distribution)
    this.renderSubDistribution(radarData.sub_distribution || []);

    // 渲染论文卡片列表
    this.renderPaperCards(radarData.papers || []);

    // 更新翻卷状态
    const maxPages = Math.ceil(totalCount / this.pageSize) || 1;
    if (this.pageIndicator) {
      this.pageIndicator.textContent = `第 ${this.currentPage} / ${maxPages} 卷`;
    }
    if (this.btnPrev) {
      this.btnPrev.disabled = this.currentPage <= 1;
    }
    if (this.btnNext) {
      this.btnNext.disabled = this.currentPage >= maxPages;
    }
  }

  renderSubDistribution(distList) {
    if (this.btnDistAll) {
      this.btnDistAll.classList.toggle("active", this.selectedSubCategories.length === 0);
    }
    if (!this.distBarsContainer) return;
    this.distBarsContainer.innerHTML = "";

    if (!distList || distList.length === 0) {
      this.distBarsContainer.innerHTML = `<span class="dist-loading-text">尚无子学科分布统计</span>`;
      return;
    }

    distList.forEach((item) => {
      const isSelected = this.selectedSubCategories.includes(item.code);
      const chip = document.createElement("div");
      chip.className = `dist-bar-chip ${isSelected ? "active" : ""}`;
      chip.title = `点击聚焦/取消过滤: ${item.code} (${item.name_zh})`;
      chip.innerHTML = `
        <span class="dist-chip-code">${item.code}</span>
        <span class="dist-chip-name">${item.name_zh}</span>
        <span class="dist-chip-count">${item.count}</span>
      `;
      // 点击分布胶囊切换选择此子学科
      chip.addEventListener("click", () => {
        if (isSelected) {
          this.selectedSubCategories = this.selectedSubCategories.filter((c) => c !== item.code);
        } else {
          this.selectedSubCategories = [item.code];
        }
        this.renderSubTree();
        this.currentPage = 1;
        this.doSearch();
      });
      this.distBarsContainer.appendChild(chip);
    });
  }

  extractCoreThesis(summary, title, primaryCat) {
    if (!summary || !summary.trim()) {
      return `聚焦于「${primaryCat || "学术前沿"}」问题，探讨核心机制与推演模型。`;
    }
    const clean = summary.replace(/\s+/g, " ").trim();
    const sentences = clean.match(/[^.!?]+[.!?]+/g) || [clean];
    const actionKeywords = [
      "we present", "we propose", "we introduce", "we show", "we demonstrate",
      "we develop", "our goal is", "we compute", "in this paper we", "in this work, we",
      "in this work we", "this work presents", "this paper presents", "we investigate",
      "we find", "our results show", "our results demonstrate", "we establish", 
      "we formulate", "we perform", "here we", "this paper explores", "this study",
      "we examine", "we design", "we construct", "to this end, we", "we analyze"
    ];
    for (const s of sentences) {
      const sLower = s.toLowerCase();
      if (actionKeywords.some((kw) => sLower.includes(kw))) {
        let sent = s.trim();
        if (sent.length > 280) {
          sent = sent.substring(0, 277) + "...";
        }
        return sent;
      }
    }
    let first = sentences[0].trim();
    if (first.length > 280) {
      first = first.substring(0, 277) + "...";
    }
    return first;
  }

  formatPublishedDate(publishedAt) {
    if (!publishedAt) return "";

    const date = new Date(publishedAt);
    if (Number.isNaN(date.getTime())) {
      return String(publishedAt).substring(0, 10);
    }

    const parts = new Intl.DateTimeFormat("zh-CN", {
      timeZone: "Asia/Shanghai",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).formatToParts(date);
    const values = Object.fromEntries(
      parts
        .filter((part) => ["year", "month", "day"].includes(part.type))
        .map((part) => [part.type, part.value])
    );
    return `${values.year}-${values.month}-${values.day}`;
  }

  renderPaperCards(papers) {
    if (!this.paperListContainer) return;
    this.paperListContainer.innerHTML = "";

    if (papers.length === 0) {
      this.paperListContainer.innerHTML = `
        <div class="empty-hint" style="text-align:center; padding: 40px; color: var(--color-ink-muted);">
          🎋 沧海遗珠，当前考据经纬下暂无最新卷册。不妨调宽检索范围或清空细选。
        </div>
      `;
      return;
    }

    papers.forEach((paper) => {
      const card = document.createElement("article");
      card.className = "radar-paper-card";

      // 分类标签徽章
      const primaryCat = paper.primary_category || "Unclassified";
      const catTags = (paper.categories || [])
        .map((c) => `<span class="paper-cat-badge ${c === primaryCat ? "" : "secondary"}">${c}</span>`)
        .join(" ");

      // 作者列表
      const authorsStr = (paper.authors || []).join(", ") || "佚名学者";

      // 智能查找该子学科中文名
      let catZh = "";
      if (this.taxonomy) {
        for (const maj of Object.values(this.taxonomy)) {
          for (const grp of Object.values(maj.groups || {})) {
            if (grp.subcategories && grp.subcategories[primaryCat]) {
              catZh = grp.subcategories[primaryCat].name_zh;
              break;
            }
          }
          if (catZh) break;
        }
      }

      // 智能提取该论著的独特核心创新贡献与核心要旨句子 (杜绝模板化重复)
      const coreThesis = this.extractCoreThesis(paper.summary, paper.title, primaryCat);

      const publishedDate = this.formatPublishedDate(paper.published_at);
      const todayDate = this.formatPublishedDate(new Date().toISOString());
      const isLatestBatch = Boolean(publishedDate && publishedDate === todayDate);
      const dateLabel = isLatestBatch
        ? `✨ ${publishedDate} 今日新发`
        : `📅 著于 ${publishedDate || paper.year || "近期"}`;

      // 检查本地持久化缓存 (免重复消耗 Token)
      let cachedTrans = null;
      try {
        const cachedStr = localStorage.getItem(`radar_trans_${paper.paper_id}`);
        if (cachedStr) {
          const parsed = JSON.parse(cachedStr);
          // 只认全文翻译缓存；旧版摘要翻译缓存不会在卡片加载时自动展示。
          if (parsed && parsed.version >= 2 && parsed.full_text_zh) {
            cachedTrans = parsed;
          }
        }
      } catch (e) {}

      card.innerHTML = `
        <div class="radar-card-header-row">
          <span class="paper-id-tag">arXiv:${paper.paper_id}</span>
          <span class="paper-date-tag" title="arXiv 今日全网公开上线批次 (作者封稿于 ${publishedDate})">${dateLabel}</span>
          ${catTags}
        </div>
        <h3 class="radar-paper-title">${paper.title}</h3>
        <div class="radar-paper-authors">👤 著撰：${authorsStr}</div>
        <div class="radar-ai-summary-box">
          <div class="summary-meta-header">
            <span class="summary-badge-sub">🎋 ${catZh ? catZh + " · " : ""}${primaryCat}</span>
            <button class="btn-paper-translate ${cachedTrans ? 'cached' : ''}" id="btn-trans-${paper.paper_id}" title="按需翻译整篇论文">
              ${cachedTrans ? '🇨🇳 查看中文全文' : '🇨🇳 翻译全文'}
            </button>
          </div>
          <div class="summary-thesis-content">
            <strong>🎯 核心论断 (Core Thesis):</strong> <span>“${coreThesis}”</span>
          </div>
          <div class="summary-zh-box" id="trans-box-${paper.paper_id}" aria-live="polite" style="display:none;"></div>
        </div>
        <p class="radar-paper-abstract" id="abs-${paper.paper_id}">
          <strong class="abs-prefix">【英文摘要 Abstract】：</strong>${paper.summary}
        </p>
        <div class="radar-paper-actions">
          <button class="btn-paper-action primary-trans ${cachedTrans ? 'cached' : ''}" id="btn-action-trans-${paper.paper_id}" title="按需翻译整篇论文">
            ${cachedTrans ? '🇨🇳 查看中文全文' : '🇨🇳 翻译全文'}
          </button>
          <button class="btn-paper-action" id="btn-toggle-abs-${paper.paper_id}">📖 展开英文摘要</button>
          <a href="https://arxiv.org/abs/${paper.paper_id}" target="_blank" class="btn-paper-action">🔗 arXiv 原文</a>
          <a href="${paper.pdf_url}" target="_blank" class="btn-paper-action">📄 研读经卷 (PDF)</a>
          <button class="btn-paper-action gold" id="btn-ingest-${paper.paper_id}">📥 藏之石渠 (入书阁)</button>
          <button class="btn-paper-action highlight" id="btn-discuss-${paper.paper_id}">🔬 兰台问道 (召羽若研讨)</button>
        </div>
      `;

      this.paperListContainer.appendChild(card);

      // 按需翻译整篇论文 (用户点击后才下载 PDF、调用模型，并使用本地缓存)
      const btnTrans = document.getElementById(`btn-trans-${paper.paper_id}`);
      const btnActionTrans = document.getElementById(`btn-action-trans-${paper.paper_id}`);
      const transBox = document.getElementById(`trans-box-${paper.paper_id}`);

      const updateTransButtons = (text, isCached = false, disabled = false) => {
        [btnTrans, btnActionTrans].forEach(b => {
          if (!b) return;
          b.disabled = disabled;
          b.textContent = text;
          if (isCached) b.classList.add("cached");
          else b.classList.remove("cached");
        });
      };

      if (transBox) {
        const renderTransData = (data) => {
          const escapeHtml = (value) => String(value || "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/\"/g, "&quot;")
            .replace(/'/g, "&#039;");
          const fullTextZh = escapeHtml((data.full_text_zh || data.abstract_zh || "").replace(/\\\\/g, "\\"));
          transBox.innerHTML = `
            <div class="trans-full-paper-header">📚 【中文全文】</div>
            <div class="trans-full-paper-body">${fullTextZh}</div>
          `;
          transBox.style.display = "block";
          transBox.dataset.loaded = "true";
          updateTransButtons("📜 收起中文全文", true, false);
          this.renderMathEquations(transBox);
        };

        const handleTransClick = async () => {
          if (transBox.dataset.loaded === "true") {
            const isHidden = transBox.style.display === "none";
            transBox.style.display = isHidden ? "block" : "none";
            updateTransButtons(isHidden ? "📜 收起中文全文" : "🇨🇳 查看中文全文", true, false);
            if (isHidden) {
              transBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }
            return;
          }

          if (cachedTrans) {
            renderTransData(cachedTrans);
            transBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
            return;
          }

          updateTransButtons("⏳ 正在翻译全文...", false, true);
          try {
            const resp = await fetch("/api/radar/translate", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                title: paper.title,
                paper_id: paper.paper_id,
                pdf_url: paper.pdf_url,
                text: coreThesis,
                abstract: paper.summary,
              }),
            });
            const resData = await resp.json();
            if (resData.status === "success" && resData.data) {
              const d = resData.data;
              try {
                localStorage.setItem(`radar_trans_${paper.paper_id}`, JSON.stringify(d));
              } catch (e) {
                console.warn("存储翻译缓存失败:", e);
              }
              renderTransData(d);
              transBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
            } else {
              throw new Error(resData.detail || "翻译异常");
            }
          } catch (err) {
            updateTransButtons("🇨🇳 翻译全文", false, false);
            alert(`翻译失败: ${err.message}`);
          }
        };

        if (btnTrans) btnTrans.addEventListener("click", handleTransClick);
        if (btnActionTrans) btnActionTrans.addEventListener("click", handleTransClick);
      }

      // 展开/收起英文摘要 (使用 document.getElementById 避免 arXiv ID 包含小数点引起 querySelector 语法错误)
      const btnToggleAbs = document.getElementById(`btn-toggle-abs-${paper.paper_id}`);
      const absElem = document.getElementById(`abs-${paper.paper_id}`);
      if (btnToggleAbs && absElem) {
        btnToggleAbs.addEventListener("click", () => {
          if (absElem.classList.contains("expanded")) {
            absElem.classList.remove("expanded");
            btnToggleAbs.textContent = "📖 展开英文摘要";
          } else {
            absElem.classList.add("expanded");
            btnToggleAbs.textContent = "📜 收起英文摘要";
          }
        });
      }

      // 一键入书阁
      const btnIngest = document.getElementById(`btn-ingest-${paper.paper_id}`);
      if (btnIngest) {
        btnIngest.addEventListener("click", () => {
          this.ingestPaper(paper, btnIngest, false);
        });
      }

      // 呼叫羽若深度研讨 (自动新建专属研讨会话窗口，并根据论文题名、核心要旨与所属学科动态定制研读指令)
      const btnDiscuss = document.getElementById(`btn-discuss-${paper.paper_id}`);
      if (btnDiscuss) {
        btnDiscuss.addEventListener("click", async () => {
          // 1. 提取定制研讨提示词与专属会话窗口标题
          const discussPrompt = this.buildTargetedDiscussPrompt(paper, primaryCat, transBox);
          let zhTitle = "";
          if (transBox && transBox.dataset.loaded === "true") {
            const titleSpan = transBox.querySelector(".trans-title span");
            if (titleSpan) zhTitle = titleSpan.textContent.trim();
          }
          if (!zhTitle && paper.paper_id) {
            try {
              const cached = JSON.parse(localStorage.getItem(`radar_trans_${paper.paper_id}`) || "{}");
              if (cached.title_zh) zhTitle = cached.title_zh;
            } catch (e) {}
          }
          const sessionTitle = `📜 研读：${(zhTitle || paper.title || "前沿论文").slice(0, 28)}`;

          // 2. 切换回研讨对话视图，并新建独立专属学术研讨窗口 (隔离旧会话历史与旧教材上下文)
          this.switchView("dialogue");
          if (typeof window.createNewChat === "function") {
            await window.createNewChat(sessionTitle);
          }

          // 3. 检查是否开启【研讨联动自动纳卷入阁】
          const autoIngestEnabled = this.autoIngestCheckbox ? this.autoIngestCheckbox.checked : true;
          if (autoIngestEnabled) {
            // 异步后台静默入库，不阻断研讨流
            this.ingestPaper(paper, btnIngest, true);
          }

          // 4. 经卷脉络优先切至「本轮精析」视角，即刻呈现羽若本轮针对该论文的深度推演
          if (window.switchMindmapScope) {
            window.switchMindmapScope('current');
          }

          // 5. 发送定制研讨提示词
          const chatInput = document.getElementById("query-input") || document.getElementById("chat-input");
          const btnSend = document.getElementById("btn-send");
          if (chatInput) {
            chatInput.value = discussPrompt;
            if (btnSend) {
              btnSend.click();
            }
          }
        });
      }
    });

    // 🌟 全量渲染学术经卷标题与要旨中的 LaTeX 数学公式 (如 $\mathcal{N}=1$, $\mathcal{S}$, $\Sigma_2$ 等)
    this.renderMathEquations();
  }

  /**
   * 🌟 论文入阁统一处理（支持即时手动入库与研讨时后台静默异步入库）
   */
  async ingestPaper(paper, btnIngest, isSilent = false) {
    if (btnIngest) {
      btnIngest.disabled = true;
      btnIngest.textContent = "⏳ 采撷入阁中...";
    }
    if (isSilent && window.showAcademicToast) {
      window.showAcademicToast(`📥 正在后台采撷《${paper.title.slice(0, 30)}...》全文入书阁...`, "info");
    }
    try {
      const ingestResp = await fetch("/api/radar/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          paper_id: paper.paper_id,
          title: paper.title,
          pdf_url: paper.pdf_url,
          authors: paper.authors,
          year: paper.year,
          summary: paper.summary,
        }),
      });
      const res = await ingestResp.json();
      if (res.status === "success") {
        if (btnIngest) {
          btnIngest.textContent = "✅ 已藏入书阁";
          btnIngest.disabled = true;
        }
        const fileName = (res.result && res.result.file_name) || "";
        // 更新卡片上的 PDF 链接，指向本地极速下载/阅读端点
        const pdfLink = document.getElementById(`btn-pdf-${paper.paper_id}`);
        if (pdfLink && fileName) {
          pdfLink.href = `/api/documents/${encodeURIComponent(fileName)}/download`;
          pdfLink.textContent = "📄 研读本地卷 (PDF)";
          pdfLink.title = "已纳阁，点击直接本地高速下载与阅览";
        }
        // 1. 刷新左侧知识库经卷列表
        if (window.loadDocumentList) {
          await window.loadDocumentList();
        } else if (window.fetchDocuments) {
          await window.fetchDocuments();
        }
        // 2. 自动把当前激活经卷设为该文献
        if (fileName && window.setCurrentActiveDocument) {
          window.setCurrentActiveDocument(fileName);
        }
        if (window.showAcademicToast) {
          window.showAcademicToast(`📜 《${paper.title.slice(0, 30)}...》全文已纳库！全书宏观脉络与证据链已就绪`, "success");
        }
        return res;
      } else {
        throw new Error(res.detail || "入库失败");
      }
    } catch (err) {
      console.warn("Paper ingest error:", err);
      if (btnIngest) {
        btnIngest.disabled = false;
        btnIngest.textContent = "❌ 采撷受阻";
      }
      if (!isSilent) {
        alert(`藏入书阁失败: ${err.message}`);
      } else if (window.showAcademicToast) {
        window.showAcademicToast(`⚠️ 后台采撷受阻: ${err.message}`, "error");
      }
    }
  }

  /**
   * 🌟 基于论文真实题名、摘要及华译要旨动态构建学术研讨指令（零关键词分支硬编码）
   */
  buildTargetedDiscussPrompt(paper, primaryCat, transBox) {
    const title = (paper.title || "").trim();
    const summary = (paper.summary || "").trim();

    // 1. 若此前已生成【AI 提要与精译】，读取大模型生成的中文题名与核心要旨 (优先从 DOM，其次从本地缓存)
    let zhTitle = "";
    let zhThesis = "";
    if (transBox && transBox.dataset.loaded === "true") {
      const titleSpan = transBox.querySelector(".trans-title span");
      const thesisSpan = transBox.querySelector(".trans-thesis span");
      if (titleSpan) zhTitle = titleSpan.textContent.trim();
      if (thesisSpan) zhThesis = thesisSpan.textContent.trim();
    }
    if (!zhTitle && paper.paper_id) {
      try {
        const cached = JSON.parse(localStorage.getItem(`radar_trans_${paper.paper_id}`) || "{}");
        if (cached.title_zh) zhTitle = cached.title_zh;
        if (cached.thesis_zh) zhThesis = cached.thesis_zh;
      } catch (e) {}
    }

    // 2. 完全依据论文自身题名、摘要与核心要旨构造通用深度研读指令，不依赖任何学科词硬编码
    let prompt = `羽若，请重点研读这篇天工雷达学术论文：\n《${title}》(arXiv:${paper.paper_id}，所属学科：${primaryCat})`;
    if (zhTitle) {
      prompt += `\n【题名华译】：${zhTitle}`;
    }
    if (zhThesis) {
      prompt += `\n【核心要旨】：${zhThesis}`;
    } else if (summary) {
      prompt += `\n【核心摘要】：\n${summary}`;
    }
    prompt += `\n\n请紧密围绕《${zhTitle || title}》的核心科学问题与摘要内容，深入推导其核心数理模型、控制方程与关键机理构造，剖析其相较于已有方法的本质突破点，并评估其学术创新价值与适用边界！`;
    return prompt;
  }

  renderMathEquations(targetElement) {
    const root = targetElement || this.paperListContainer;
    const doRender = () => {
      if (!window.renderMathInElement || !root) return;
      try {
        window.renderMathInElement(root, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
            { left: "\\(", right: "\\)", display: false },
            { left: "\\[", right: "\\]", display: true },
          ],
          throwOnError: false,
          ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
        });
      } catch (e) {
        console.warn("KaTeX renderMathInElement error:", e);
      }
    };

    if (window.renderMathInElement) {
      doRender();
    } else {
      setTimeout(doRender, 300);
    }
  }
}

// 页面加载完成后实例化控制器 (兼容直接加载与异步载入)
function initTianGongRadar() {
  if (!window.tianGongRadar) {
    window.tianGongRadar = new TianGongRadarController();
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initTianGongRadar);
} else {
  initTianGongRadar();
}
