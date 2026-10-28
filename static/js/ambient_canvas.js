/**
 * 云山问学 · 清幽竹海与学海微澜高动态意境引擎 (Vivid Zen Bamboo Lake Canvas)
 */

(function () {
  const canvas = document.getElementById('ambient-zen-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let width, height;

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  // =========================================================================
  // 1. 竹林与竹枝剪影 (Vivid Bamboo Grove & Branch Clusters)
  // =========================================================================
  const stalks = [
    { xRatio: 0.02, width: 28, swayPhase: 0, swaySpeed: 0.0008, color: 'rgba(20, 64, 54, 0.35)' },
    { xRatio: 0.06, width: 18, swayPhase: 1.2, swaySpeed: 0.0011, color: 'rgba(28, 78, 66, 0.28)' },
    { xRatio: 0.11, width: 12, swayPhase: 2.5, swaySpeed: 0.0009, color: 'rgba(38, 92, 78, 0.22)' },
    { xRatio: 0.89, width: 14, swayPhase: 0.8, swaySpeed: 0.0010, color: 'rgba(38, 92, 78, 0.22)' },
    { xRatio: 0.94, width: 22, swayPhase: 2.1, swaySpeed: 0.0007, color: 'rgba(28, 78, 66, 0.30)' },
    { xRatio: 0.98, width: 32, swayPhase: 3.4, swaySpeed: 0.0012, color: 'rgba(18, 58, 48, 0.38)' },
  ];

  function drawBambooStalks(time) {
    stalks.forEach((s) => {
      const baseX = s.xRatio * width;
      const sway = Math.sin(time * s.swaySpeed + s.swayPhase) * 18;
      
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(baseX - s.width * 0.5, height);
      ctx.quadraticCurveTo(baseX + sway * 0.35, height * 0.5, baseX + sway - s.width * 0.4, 0);
      ctx.lineTo(baseX + sway + s.width * 0.4, 0);
      ctx.quadraticCurveTo(baseX + sway * 0.35 + s.width * 0.5, height * 0.5, baseX + s.width * 0.5, height);
      ctx.closePath();
      ctx.fillStyle = s.color;
      ctx.fill();

      // 绘制竹节环
      for (let y = height * 0.12; y < height; y += height * 0.13) {
        const curSway = (1 - y / height) * sway;
        ctx.beginPath();
        ctx.ellipse(baseX + curSway, y, s.width * 0.6, 3.5, 0, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(12, 45, 38, 0.45)';
        ctx.fill();
      }

      ctx.restore();
    });
  }

  // =========================================================================
  // 2. 学海微澜多层动态波浪 (Dynamic Ocean of Learning Waves)
  // =========================================================================
  const waves = [
    {
      baseYRatio: 0.68,
      amplitude: 28,
      wavelength: 0.0024,
      speed: 0.0009,
      color: 'rgba(28, 88, 75, 0.22)',
    },
    {
      baseYRatio: 0.78,
      amplitude: 34,
      wavelength: 0.0030,
      speed: -0.0011,
      color: 'rgba(38, 108, 92, 0.25)',
    },
    {
      baseYRatio: 0.88,
      amplitude: 24,
      wavelength: 0.0038,
      speed: 0.0014,
      color: 'rgba(18, 68, 58, 0.30)',
    },
    {
      baseYRatio: 0.95,
      amplitude: 16,
      wavelength: 0.0045,
      speed: -0.0008,
      color: 'rgba(14, 52, 44, 0.35)',
    },
  ];

  function drawWaves(time) {
    waves.forEach((w) => {
      const baseY = height * w.baseYRatio;
      ctx.beginPath();
      ctx.moveTo(0, height);

      for (let x = 0; x <= width; x += 12) {
        const y = baseY 
          + Math.sin(x * w.wavelength + time * w.speed) * w.amplitude 
          + Math.cos((x * 0.55) * w.wavelength + time * (w.speed * 0.75)) * (w.amplitude * 0.45);
        ctx.lineTo(x, y);
      }

      ctx.lineTo(width, height);
      ctx.closePath();
      ctx.fillStyle = w.color;
      ctx.fill();
    });
  }

  // =========================================================================
  // 3. 翠竹叶飘絮粒子系统 (50 Drifting Vivid Bamboo Leaves)
  // =========================================================================
  const LEAF_COUNT = 50;
  const leaves = [];

  const leafColors = [
    { fill: 'rgba(24, 88, 72, 0.92)', stroke: 'rgba(12, 48, 38, 1)', spine: 'rgba(120, 210, 185, 1)' },   // 浓翠松柏
    { fill: 'rgba(42, 125, 105, 0.88)', stroke: 'rgba(18, 68, 55, 1)', spine: 'rgba(150, 235, 205, 1)' }, // 翠绿嫩竹
    { fill: 'rgba(65, 150, 128, 0.85)', stroke: 'rgba(28, 85, 70, 1)', spine: 'rgba(180, 245, 225, 1)' }, // 碧玉青叶
    { fill: 'rgba(180, 135, 55, 0.88)', stroke: 'rgba(120, 85, 30, 1)', spine: 'rgba(245, 215, 130, 1)' }, // 晨曦金竹
  ];

  class BambooLeaf {
    constructor() {
      this.reset(true);
    }

    reset(initial = false) {
      this.x = initial ? Math.random() * width : width + Math.random() * 120;
      this.y = initial ? Math.random() * height : -30 - Math.random() * 120;
      this.size = 20 + Math.random() * 24;        // 叶片长度 (20px ~ 44px)
      this.aspect = 0.25 + Math.random() * 0.08;
      this.speedX = -(0.8 + Math.random() * 1.3);
      this.speedY = 0.6 + Math.random() * 1.1;
      this.angle = Math.random() * Math.PI * 2;
      this.angularSpeed = (Math.random() - 0.5) * 0.04;
      this.flip = Math.random() * Math.PI;
      this.flipSpeed = 0.025 + Math.random() * 0.03;
      this.swingPhase = Math.random() * Math.PI * 2;
      this.swingSpeed = 0.014 + Math.random() * 0.02;
      this.swingAmplitude = 1.0 + Math.random() * 1.2;
      this.color = leafColors[Math.floor(Math.random() * leafColors.length)];
      this.opacity = 0.75 + Math.random() * 0.25;
    }

    update(time, mouseForce) {
      this.swingPhase += this.swingSpeed;
      this.flip += this.flipSpeed;
      this.angle += this.angularSpeed;

      const windX = Math.sin(this.swingPhase) * this.swingAmplitude;
      this.x += this.speedX + windX + mouseForce.x;
      this.y += this.speedY + mouseForce.y;

      if (this.x < -80 || this.y > height + 80) {
        this.reset(false);
      }
    }

    draw(ctx) {
      ctx.save();
      ctx.translate(this.x, this.y);
      ctx.rotate(this.angle);

      const scaleY = Math.cos(this.flip);
      ctx.scale(1, scaleY);
      ctx.globalAlpha = this.opacity * Math.max(0.4, Math.abs(scaleY));

      const len = this.size;
      const widthHalf = len * this.aspect;

      // 绘制竹叶轮廓
      ctx.beginPath();
      ctx.moveTo(0, -len * 0.5);
      ctx.bezierCurveTo(widthHalf * 1.4, -len * 0.15, widthHalf * 1.2, len * 0.3, 0, len * 0.5);
      ctx.bezierCurveTo(-widthHalf * 1.2, len * 0.3, -widthHalf * 1.4, -len * 0.15, 0, -len * 0.5);
      ctx.closePath();

      ctx.fillStyle = this.color.fill;
      ctx.fill();
      ctx.strokeStyle = this.color.stroke;
      ctx.lineWidth = 1.0;
      ctx.stroke();

      // 绘制叶脉中脊线
      ctx.beginPath();
      ctx.moveTo(0, -len * 0.46);
      ctx.quadraticCurveTo(widthHalf * 0.12, 0, 0, len * 0.46);
      ctx.strokeStyle = this.color.spine;
      ctx.lineWidth = 0.9;
      ctx.stroke();

      ctx.restore();
    }
  }

  for (let i = 0; i < LEAF_COUNT; i++) {
    leaves.push(new BambooLeaf());
  }

  // =========================================================================
  // 4. 鼠标微风交互 (Interactive Breeze)
  // =========================================================================
  let mouse = { x: -1000, y: -1000 };
  let mouseVelocity = { x: 0, y: 0 };
  let lastMouse = { x: -1000, y: -1000 };

  window.addEventListener('mousemove', (e) => {
    mouseVelocity.x = (e.clientX - lastMouse.x) * 0.06;
    mouseVelocity.y = (e.clientY - lastMouse.y) * 0.06;
    lastMouse.x = mouse.x = e.clientX;
    lastMouse.y = mouse.y = e.clientY;
  });

  // =========================================================================
  // 5. 渲染循环 (Animation Loop)
  // =========================================================================
  let startTime = performance.now();

  function animate(now) {
    const elapsed = now - startTime;

    // 1. 绘制清幽竹海晨雾渐变底色
    const bgGrad = ctx.createLinearGradient(0, 0, 0, height);
    bgGrad.addColorStop(0, '#bfe3d6');    // 顶部清幽青翠
    bgGrad.addColorStop(0.45, '#a4d4c4'); // 中部竹林碧玉
    bgGrad.addColorStop(0.8, '#82beac');  // 下部学海碧潭
    bgGrad.addColorStop(1, '#65a794');    // 底部深林苍翠
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    mouseVelocity.x *= 0.90;
    mouseVelocity.y *= 0.90;

    // 2. 绘制竹林剪影与学海微澜
    drawBambooStalks(elapsed);
    drawWaves(elapsed);

    // 3. 绘制飘絮翠竹叶
    leaves.forEach((leaf) => {
      const dx = leaf.x - mouse.x;
      const dy = leaf.y - mouse.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      let mouseForce = { x: mouseVelocity.x * 0.4, y: mouseVelocity.y * 0.4 };

      if (dist < 160) {
        const pushFactor = (1 - dist / 160) * 1.4;
        mouseForce.x += (dx / (dist || 1)) * pushFactor;
        mouseForce.y += (dy / (dist || 1)) * pushFactor;
      }

      leaf.update(elapsed, mouseForce);
      leaf.draw(ctx);
    });

    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
})();
