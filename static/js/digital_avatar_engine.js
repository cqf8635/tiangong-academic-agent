/**
 * 云山问学 · 科研搭档研讨座席引擎 (Pure High-Fidelity Scholar Partner Engine)
 * 
 * 核心准则：
 * 1. 彻底清除任何 Canvas 人工嘴部绘制与涂抹，100% 呈现羽若 4K 高清原貌，绝无任何怪异感；
 * 2. 纯净展示超清肖像原图，零缩放模糊、零变形；
 * 3. 伴随语音发声，驱动外层声学光晕律动与状态机；
 * 4. 研讨同步电子板书与 KaTeX 数理公式联动。
 */

(function(window) {
  'use strict';

  class DigitalAvatarEngine {
    constructor() {
      this.state = 'IDLE';
      this.stageFigureBox = document.getElementById('stage-figure-box');
      this.stageFigureImg = document.getElementById('stage-figure-img');
      this.stageVideo = document.getElementById('stage-avatar-video');
      this.stageBadgeStatus = document.getElementById('stage-badge-status');
      this.headerOrb = document.getElementById('status-orb-indicator');

      // Blackboard elements
      this.blackboardCard = document.getElementById('stage-blackboard-card');
      this.blackboardBadge = document.getElementById('blackboard-status-badge');
      this.blackboardContent = document.getElementById('blackboard-content');

      // Audio analysis
      this.audioCtx = null;
      this.analyser = null;
      this.audioSource = null;
      this.freqData = null;
      this.isAudioConnected = false;
      this.animationFrameId = null;

      this.greetings = [
        '来啦！今天我们一起研读哪篇前沿文献？',
        '研精覃思，探骊索珠。公式推导或文献疑窦，随时和我一起推演。',
        '搭档羽若已就位。让我们一起核验文献脉络与数理方程。',
        '轻灵如羽，从容若水。博学笃志，切问近思。经卷资料库皆已齐备。'
      ];

      this.init();
    }

    init() {
      // 彻底清理并移除任何 canvas
      const existingCanvas = document.querySelectorAll('.stage-live2d-canvas, .stage-avatar-canvas');
      existingCanvas.forEach(c => c.remove());

      if (this.stageFigureImg) {
        this.stageFigureImg.style.display = 'block';
      }
      if (this.stageVideo) {
        this.stageVideo.style.display = 'none';
        this.stageVideo.pause();
      }

      const greetBtn = document.getElementById('btn-stage-greet');
      if (greetBtn) {
        greetBtn.addEventListener('click', (e) => {
          e.preventDefault();
          this.handleGreet();
        });
      }

      const sampleBtn = document.getElementById('btn-stage-speak-sample');
      if (sampleBtn) {
        sampleBtn.addEventListener('click', (e) => {
          e.preventDefault();
          this.handleLectureDemonstration();
        });
      }
    }

    setupAudioContext(audioElement) {
      if (this.isAudioConnected && this.analyser) return;
      try {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) return;

        if (!this.audioCtx) {
          this.audioCtx = new AudioContextClass();
        }

        if (this.audioCtx.state === 'suspended') {
          this.audioCtx.resume();
        }

        if (!this.audioSource && audioElement) {
          this.audioSource = this.audioCtx.createMediaElementSource(audioElement);
          this.analyser = this.audioCtx.createAnalyser();
          this.analyser.fftSize = 256;
          this.analyser.smoothingTimeConstant = 0.5;
          this.freqData = new Uint8Array(this.analyser.frequencyBinCount);

          this.audioSource.connect(this.analyser);
          this.analyser.connect(this.audioCtx.destination);
          this.isAudioConnected = true;
        }
      } catch (err) {
        console.warn('[DigitalAvatar] AudioContext notice:', err);
      }
    }

    startSpeaking(audioElement) {
      this.setState('SPEAKING', '🎙️ 搭档领读 (Speaking)');
      this.setupAudioContext(audioElement);

      if (this.stageFigureImg) {
        this.stageFigureImg.style.display = 'none';
      }
      if (this.stageVideo) {
        this.stageVideo.style.display = 'block';
        this.stageVideo.currentTime = 0;
        this.stageVideo.play().catch(() => {});
      }

      if (this.animationFrameId) {
        cancelAnimationFrame(this.animationFrameId);
      }

      const updateAudioAura = () => {
        if (this.state !== 'SPEAKING') return;

        let volume = 0;
        if (this.analyser && this.freqData) {
          this.analyser.getByteFrequencyData(this.freqData);
          let sum = 0;
          const startBin = 2;
          const endBin = 32;
          for (let i = startBin; i < endBin; i++) {
            sum += this.freqData[i];
          }
          volume = Math.min(1.0, (sum / (endBin - startBin)) / 100);
        }

        // 仅在卡片外框做柔和光晕律动，绝不触碰面部像素
        if (this.stageFigureBox) {
          const glow = (volume * 22).toFixed(1);
          this.stageFigureBox.style.boxShadow = volume > 0.05
            ? `0 0 ${glow}px rgba(24, 150, 115, 0.55), 0 0 ${glow * 1.5}px rgba(212, 175, 55, 0.35)`
            : '';
          this.stageFigureBox.style.borderColor = volume > 0.05
            ? 'rgba(212, 175, 55, 0.9)'
            : 'rgba(212, 175, 55, 0.4)';
        }

        this.animationFrameId = requestAnimationFrame(updateAudioAura);
      };

      this.animationFrameId = requestAnimationFrame(updateAudioAura);
    }

    stopSpeaking() {
      this.setState('IDLE', '🌿 研讨就绪 (Ready)');

      if (this.stageVideo) {
        this.stageVideo.pause();
        this.stageVideo.style.display = 'none';
      }
      if (this.stageFigureImg) {
        this.stageFigureImg.style.display = 'block';
      }

      if (this.animationFrameId) {
        cancelAnimationFrame(this.animationFrameId);
        this.animationFrameId = null;
      }
      if (this.stageFigureBox) {
        this.stageFigureBox.style.boxShadow = '';
        this.stageFigureBox.style.borderColor = 'rgba(212, 175, 55, 0.4)';
      }
    }

    setState(state, desc = '') {
      this.state = state;
      const statusMap = {
        IDLE: { desc: '🌿 研讨就绪 (Ready)', class: 'idle' },
        LISTENING: { desc: '👂 倾听研讨 (Listening)', class: 'listening' },
        THINKING: { desc: '🧠 经卷推演 (Thinking)', class: 'thinking' },
        SPEAKING: { desc: '🎙️ 温润领读 (Speaking)', class: 'speaking' },
      };
      const info = statusMap[state] || statusMap.IDLE;

      if (this.stageBadgeStatus) {
        this.stageBadgeStatus.innerHTML = `<span class="state-dot ${info.class}"></span> ${desc || info.desc}`;
      }
      if (this.headerOrb) {
        this.headerOrb.className = `status-orb ${info.class}`;
      }
      if (this.stageFigureBox) {
        if (state === 'SPEAKING') {
          this.stageFigureBox.classList.add('speaking');
        } else {
          this.stageFigureBox.classList.remove('speaking');
        }
      }
    }

    clearBlackboard() {
      if (this.blackboardBadge) {
        this.blackboardBadge.textContent = '🌱 静待研讨';
      }
      if (this.blackboardContent) {
        this.blackboardContent.innerHTML = '<div class="empty-hint" style="padding: 14px 8px; text-align: center; color: #94a3b8; font-size: 0.76rem; line-height: 1.6;">🎋 提出学术问题或点击上方「领读当前研讨要点」后，核心论点、推导公式与经卷要旨将在此处实时同步板书呈现。</div>';
      }
    }

    updateBlackboard(title, text, formula = '', notes = '', badgeText = '第 1 讲 · 核心论点') {
      if (!this.blackboardContent) return;
      if (this.blackboardBadge) {
        this.blackboardBadge.textContent = badgeText;
      }
      
      let formulaHtml = '';
      if (formula) {
        const cleanMath = formula.replace(/^\$\$|\$\$$/g, '').trim();
        if (window.katex) {
          try {
            formulaHtml = window.katex.renderToString(cleanMath, { displayMode: true, throwOnError: false });
          } catch (_) {
            formulaHtml = `<span style="font-family: var(--font-mono); color: #fde047;">${cleanMath}</span>`;
          }
        } else {
          formulaHtml = `<span style="font-family: var(--font-mono); color: #fde047;">${cleanMath}</span>`;
        }
      }

      let html = '';
      if (title) html += `<h4 class="blackboard-heading">${title}</h4>`;
      if (text) html += `<p class="blackboard-text">${text}</p>`;
      if (formulaHtml) html += `<div class="blackboard-formula">${formulaHtml}</div>`;
      if (notes) html += `<div class="blackboard-notes">💡 <strong>研讨核心要点：</strong>${notes}</div>`;
      this.blackboardContent.innerHTML = html;
    }

    async handleGreet() {
      const idx = Math.floor(Math.random() * this.greetings.length);
      const text = this.greetings[idx];
      this.updateBlackboard(
        '云山研讨 · 同侪切磋',
        text,
        '',
        '随时提出数理推导、文献疑问或研究思路，我们一起并肩探索。',
        '随堂问答 · 灵感碰撞'
      );

      if (typeof window.playTTS === 'function') {
        window.playTTS(null, text);
      }
    }

    async handleLectureDemonstration() {
      const avatarTabBtn = document.querySelector('[data-tab="tab-avatar"]');
      if (avatarTabBtn && typeof avatarTabBtn.click === 'function') {
        avatarTabBtn.click();
      }

      const chatBubbles = document.querySelectorAll('.message-bubble.agent');
      if (chatBubbles.length === 0) {
        const introText = '你好！我是你的科研伙伴羽若。当前会话暂无讨论记录，请在下方输入框提问或在左侧上传文献，我们即刻开始并肩推演。';
        this.updateBlackboard(
          '云山问学 · 研讨发轫',
          introText,
          '',
          '支持全格式论文解析、LaTeX 公式严密推导与多智能体学术考据。',
          '🌱 研讨就绪'
        );
        
        if (typeof window.playTTS === 'function') {
          window.playTTS(null, introText);
        }
      } else {
        const lastAgentBubble = chatBubbles[chatBubbles.length - 1];
        // 严格定位学术正文容器，彻底避开顶部元数据栏 (.message-meta-line)
        const bodyEl = lastAgentBubble.querySelector('.markdown-body-text') || lastAgentBubble.querySelector('.msg-card');
        
        let textSource = '';
        if (bodyEl) {
          const clone = bodyEl.cloneNode(true);
          // 彻底剔除元数据行、证据标签、多智能体闭环研讨轨迹、导图卡片与按钮等无关元素
          clone.querySelectorAll('.message-meta-line, .evidence-badges, .multi-agent-track-details, .msg-mindmap-wrapper, .btn-replay-audio').forEach(el => el.remove());
          textSource = clone.textContent || '';
        } else {
          textSource = '研讨核心论点已同步。';
        }

        // 彻底清洗 Emoji 图标、机器人标签、意图元数据与 Markdown 语法
        let cleanText = textSource
          .replace(/🤖\s*[\w_]+/g, '')
          .replace(/🎯\s*意图[：:]\s*[\w_]+/g, '')
          .replace(/AI\s*回复/g, '')
          .replace(/[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, '') // 清除全部 Emoji，杜绝 TTS 念出“机器人/靶心”
          .replace(/```[\s\S]*?```/g, '')
          .replace(/\[\^?\d+\]/g, '')
          .replace(/[#*`_>~]/g, '')
          .trim();

        // 提取代表性数学公式
        let formula = '';
        const mForm = textSource.match(/\$\$([\s\S]+?)\$\$/) || textSource.match(/\\\[([\s\S]+?)\\\]/);
        if (mForm) formula = mForm[1].trim();

        // 清理正文中的公式占位
        const blackboardSummary = cleanText.replace(/\$\$[\s\S]+?\$\$/g, '').replace(/\\\[[\s\S]+?\\\]/g, '').replace(/\s+/g, ' ').trim().slice(0, 160);

        this.updateBlackboard(
          '本轮研讨核心要点导读',
          blackboardSummary ? `${blackboardSummary}...` : '当前论点已同步。',
          formula,
          '同侪并肩推演，针对上述定理与公式深入探讨。',
          '本轮要旨 · 精讲导读'
        );

        if (typeof window.playTTS === 'function') {
          // 自然口语领读，杜绝任何系统代号
          const spokenLead = blackboardSummary ? `本轮研讨的核心要点是：${blackboardSummary}` : '本轮核心数理推演与学术论点已同步至板书。';
          window.playTTS(null, spokenLead);
        }
      }
    }
  }

  window.DigitalAvatarEngine = DigitalAvatarEngine;

  document.addEventListener('DOMContentLoaded', () => {
    window.digitalAvatar = new DigitalAvatarEngine();
  });

})(window);
