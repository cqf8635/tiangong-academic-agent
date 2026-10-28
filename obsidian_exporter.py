"""Obsidian Knowledge Exporter & Zettelkasten Knowledge Graph Engine.

Connects the Research Academic System with local Obsidian Vault:
1. Automatic Zettelkasten Markdown generation with Double-Bracket Wikilinks ([[Concept]]).
2. Native Obsidian Canvas (.canvas JSON) visual mindmap generation with nodes and edges.
3. Multi-source evidence annotation (arXiv, CrossRef DOI, GitHub).
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from config_utils import get_default_obsidian_vault_path
from session_manager import SessionManager, ConversationSession

DEFAULT_VAULT_PATH = get_default_obsidian_vault_path()


class ObsidianKnowledgeExporter:
    """Obsidian 知识图谱与原生白板画布导出引擎"""

    def __init__(self, vault_path: Optional[Path] = None):
        self.vault_path = vault_path or DEFAULT_VAULT_PATH
        self.research_notes_dir = self.vault_path / "学术研讨"
        self.mindmaps_dir = self.vault_path / "思维导图"
        self.concept_cards_dir = self.vault_path / "概念卡片"

        # 确保 Obsidian Vault 子目录结构健全
        for d in [self.research_notes_dir, self.mindmaps_dir, self.concept_cards_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def export_session(
        self,
        session_id: str,
        export_canvas: bool = True
    ) -> Dict[str, Any]:
        """将指定研讨会话全量导出为 Obsidian 双链笔记与 Canvas 白板"""
        session = SessionManager.get_session(session_id)
        if not session:
            return {"status": "error", "message": f"会话不存在: {session_id}"}

        # 提取会话核心主旨与研讨内容
        topic = (session.topic_state.get("current_topic") if session.topic_state else None) or session.title
        safe_title = re.sub(r'[\\/:*?"<>|]+', '_', topic)[:40].strip() or f"研讨记录_{session_id[:8]}"

        # 1. 构建结构化双链 Markdown 研讨笔记
        note_filename = f"{safe_title}.md"
        note_file_path = self.research_notes_dir / note_filename
        markdown_content = self._build_obsidian_markdown(session, safe_title)
        note_file_path.write_text(markdown_content, encoding="utf-8")

        canvas_path_str = None
        # 2. 构建 Obsidian 原生 Canvas 白板 (.canvas)
        if export_canvas:
            canvas_filename = f"{safe_title}.canvas"
            canvas_file_path = self.mindmaps_dir / canvas_filename
            canvas_data = self._build_obsidian_canvas(session, safe_title)
            canvas_file_path.write_text(json.dumps(canvas_data, ensure_ascii=False, indent=2), encoding="utf-8")
            canvas_path_str = str(canvas_file_path)

        # 3. 自动提炼并生成关联的实体概念卡片 (构建真正的网状知识星图)
        created_cards = self._generate_concept_cards(session)

        return {
            "status": "success",
            "vault_path": str(self.vault_path),
            "note_path": str(note_file_path),
            "canvas_path": canvas_path_str,
            "concept_cards_created": len(created_cards),
            "note_title": safe_title,
        }

    def _build_obsidian_markdown(self, session: ConversationSession, title: str) -> str:
        """构建兼容 Obsidian 知识图谱的双链 Markdown (原生 MathJax $$ 语法)"""
        topic_info = session.topic_state or {}
        curr_topic = topic_info.get("current_topic", title)
        sub_topic = topic_info.get("sub_topic", "")
        status = topic_info.get("status", "LOCKED")
        evolution = topic_info.get("evolution_path", [])

        # YAML Frontmatter
        lines = [
            "---",
            f"title: \"{title}\"",
            f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"session_id: \"{session.session_id}\"",
            f"domain: \"{session.domain}\"",
            f"topic: \"{curr_topic}\"",
            f"status: \"{status}\"",
            "tags:",
            "  - 学术研讨",
            "  - PaperIdea",
            "  - 云山科研搭档_羽若",
            f"  - 领域/{session.domain}",
            "---",
            "",
            f"# 📚 《{title}》 学术研讨全景笔记",
            "",
            "> [!NOTE] 研讨主题生命周期状态",
            f"> - **主干主题**：[[{curr_topic}]]",
        ]

        if sub_topic:
            lines.append(f"> - **演进子分支**：[[{sub_topic}]]")
        if evolution:
            evo_links = " $\\rightarrow$ ".join([f"[[{e}]]" for e in evolution])
            lines.append(f"> - **认知演进轨迹**：{evo_links}")

        lines.extend([
            f"> - **聚焦核心文献**：[[{session.active_doc or '多源文献协同研讨'}]]",
            f"> - **研讨搭档**：[[羽若]]",
            f"> - **可视化白板**：[[{title}.canvas|🎨 点击进入本研讨拓扑白板节点图]]",
            "",
            "> [!TIP] 快捷导航与图谱查看提示",
            f"> - **查看节点导图**：在左侧文件夹打开 `思维导图/{title}.canvas` 即可自由缩放拖拽节点卡片；",
            "> - **查看全局星网**：在 Obsidian 中按下 **`Ctrl + G`**，即可展开全局概念与研讨的网状星图！",
            "",
            "## 📖 研讨结构化长期记忆摘要",
            session.history_summary or "本次研讨暂无历史摘要沉淀。",
            "",
            "---",
            "",
            "## 💬 深度研讨问答与论证脉络",
            ""
        ])

        # 遍历消息流
        for idx, msg in enumerate(session.messages, 1):
            if msg.sender == "user":
                lines.append(f"### 👤 学术提问 #{idx}")
                lines.append(f"**提问内容**：{msg.content.strip()}\n")
            elif msg.sender == "agent":
                lines.append(f"### 🤖 羽若研讨解析 #{idx}")
                
                # 注入多源证据块
                if msg.retrieved_evidence:
                    lines.append("> [!INFO] 考据与理论支撑来源")
                    for ev in msg.retrieved_evidence:
                        doc = ev.get("doc_name", "未知来源")
                        loc = ev.get("location_label", "")
                        dtype = ev.get("doc_type", "doc")
                        lines.append(f"> - **[[{doc}]]** ({loc}) [{dtype}]")
                    lines.append("")

                # 1. 将 LaTeX 格式精确转为 Obsidian 原生 MathJax ($$...$$, $...$)
                clean_latex = self._format_latex_for_obsidian(msg.content.strip())
                # 2. 自动为关键学术实体添加双括号双链 [[Entity]]
                session_ents = [curr_topic]
                if sub_topic:
                    session_ents.append(sub_topic)
                formatted_answer = self._inject_wikilinks(clean_latex, known_entities=session_ents)
                lines.append(formatted_answer)
                lines.append("\n---\n")

        lines.extend([
            "## 🕸️ 知识图谱关联网络",
            f"- 上级研讨方向：[[{session.domain.upper()} 领域知识总览]]",
            f"- 对应思维导图白板：[[{title}.canvas]]",
            ""
        ])

        return "\n".join(lines)

    def _build_obsidian_canvas(self, session: ConversationSession, title: str) -> Dict[str, Any]:
        """构建 Obsidian 原生 JSON Canvas 白板"""
        nodes = []
        edges = []

        # 1. 中心主根节点 (Root Node)
        root_id = "root_node"
        topic_info = session.topic_state or {}
        curr_topic = topic_info.get("current_topic", title)

        nodes.append({
            "id": root_id,
            "x": 0,
            "y": 0,
            "width": 380,
            "height": 180,
            "type": "text",
            "text": f"# 🎯 {curr_topic}\n\n**研究领域**: #{session.domain}\n**搭档**: 羽若\n**状态**: {topic_info.get('status', 'LOCKED')}",
            "color": "1"  # 红色/核心高亮
        })

        # 2. 消息问答分支节点 (Turn Nodes)
        y_offset = -220
        for i, msg in enumerate(session.messages):
            if msg.sender == "agent":
                node_id = f"agent_node_{i}"
                x_pos = 460
                y_pos = y_offset
                y_offset += 260

                # 提炼简要卡片内容
                content_preview = msg.content[:240].replace("\n", " ") + "..."
                nodes.append({
                    "id": node_id,
                    "x": x_pos,
                    "y": y_pos,
                    "width": 420,
                    "height": 220,
                    "type": "text",
                    "text": f"### 💡 指导要点 #{i//2 + 1}\n\n{content_preview}",
                    "color": "4"  # 绿色
                })

                # 连线：从 Root 指向指导要点
                edges.append({
                    "id": f"edge_{root_id}_{node_id}",
                    "fromNode": root_id,
                    "fromSide": "right",
                    "toNode": node_id,
                    "toSide": "left",
                    "label": f"研讨递进 #{i//2 + 1}"
                })

                # 3. 证据卡片节点 (Evidence Sub-nodes)
                if msg.retrieved_evidence:
                    ev_x = 940
                    ev_y = y_pos - 40
                    for j, ev in enumerate(msg.retrieved_evidence[:2]):
                        ev_node_id = f"ev_node_{i}_{j}"
                        doc_name = ev.get("doc_name", "证据")
                        loc = ev.get("location_label", "")
                        nodes.append({
                            "id": ev_node_id,
                            "x": ev_x,
                            "y": ev_y + (j * 110),
                            "width": 300,
                            "height": 90,
                            "type": "text",
                            "text": f"📚 **{doc_name}**\n📍 {loc}",
                            "color": "6"  # 紫色
                        })
                        edges.append({
                            "id": f"edge_{node_id}_{ev_node_id}",
                            "fromNode": node_id,
                            "fromSide": "right",
                            "toNode": ev_node_id,
                            "toSide": "left",
                            "label": "理论考据"
                        })

        return {
            "nodes": nodes,
            "edges": edges
        }

    def _format_latex_for_obsidian(self, text: str) -> str:
        r"""将学术 LaTeX 格式 (\[...\], \(...\)) 转换为 Obsidian 原生 MathJax 语法 ($$...$$, $...$)"""
        if not text:
            return ""

        # 1. 转换块级公式 \[ ... \] -> $$ ... $$ 并清除内部空行
        def replace_block_brackets(m):
            inner = m.group(1).strip()
            lines = [l.strip() for l in inner.split('\n') if l.strip()]
            inner_clean = '\n'.join(lines)
            return f"\n$$\n{inner_clean}\n$$\n"

        res = re.sub(r'\\\[([\s\S]*?)\\\]', replace_block_brackets, text)

        # 2. 转换行内公式 \( ... \) -> $ ... $
        res = re.sub(r'\\\((.*?)\\\)', r'$\1$', res)

        # 3. 规范化已有的 $$...$$ 中的空行 (防止 Obsidian 解析失败)
        def replace_dollar_block(m):
            inner = m.group(1).strip()
            lines = [l.strip() for l in inner.split('\n') if l.strip()]
            inner_clean = '\n'.join(lines)
            return f"\n$$\n{inner_clean}\n$$\n"

        res = re.sub(r'\$\$([\s\S]*?)\$\$', replace_dollar_block, res)

        return res

    def _inject_wikilinks(self, text: str, known_entities: Optional[List[str]] = None) -> str:
        """为当前研讨核心学术实体与概念卡片自动注入 Obsidian 双向链接 [[...]] (严密保护 MathJax 公式内部)"""
        # 1. 暂存所有 LaTeX 数学公式块 ($$...$$ 和 $...$)，避免公式内部被注入 [[]] 导致语法报错
        math_blocks = []
        def save_math(m):
            math_blocks.append(m.group(0))
            return f"__MATH_BLOCK_{len(math_blocks)-1}__"

        # 保护代码块 ```...```
        code_blocks = []
        def save_code(m):
            code_blocks.append(m.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"

        protected = re.sub(r'```[\s\S]*?```', save_code, text)
        protected = re.sub(r'\$\$[\s\S]*?\$\$', save_math, protected)
        protected = re.sub(r'\$[^\$\n]+?\$', save_math, protected)

        # 动态获取当前知识库中已建立的概念卡片与传入的研讨实体，完全跨学科零硬编码
        entities = set(known_entities or [])
        if self.concept_cards_dir.exists():
            for p in self.concept_cards_dir.glob("*.md"):
                if p.stem and len(p.stem) >= 2 and not p.stem.startswith("."):
                    entities.add(p.stem)

        sorted_entities = sorted(entities, key=lambda x: len(x), reverse=True)
        
        for entity in sorted_entities:
            if not entity or len(entity) < 2:
                continue
            # 避免在已带 [[]] 的词中错误替换
            pattern = rf'(?<!\[\[)\b{re.escape(entity)}\b(?!\]\])' if entity.isascii() else rf'(?<!\[\[){re.escape(entity)}(?!\]\])'
            protected = re.sub(pattern, f'[[{entity}]]', protected)

        # 2. 还原公式与代码块
        for idx, math_content in enumerate(math_blocks):
            protected = protected.replace(f"__MATH_BLOCK_{idx}__", math_content)
        for idx, code_content in enumerate(code_blocks):
            protected = protected.replace(f"__CODE_BLOCK_{idx}__", code_content)

        return protected

    def _generate_concept_cards(self, session: ConversationSession) -> List[str]:
        """自动在 概念卡片/ 目录下生成原子卡片，夯实知识星网"""
        created = []
        topic_info = session.topic_state or {}
        curr_topic = topic_info.get("current_topic") or session.title
        
        entities_to_create = set()
        if curr_topic:
            entities_to_create.add(curr_topic)

        # 动态提取会话中的研讨概念实体（完全跨学科开放，杜绝任何学科硬编码）
        if topic_info.get("sub_topic"):
            entities_to_create.add(topic_info.get("sub_topic"))
        for step in topic_info.get("evolution_path", []):
            if isinstance(step, dict) and step.get("topic"):
                entities_to_create.add(step.get("topic"))
            elif isinstance(step, str) and step.strip():
                entities_to_create.add(step.strip())

        # 从正文中动态发现被重点强调的术语（粗体、行内代码或三级标题中的核心概念）
        full_text = " ".join([m.content for m in session.messages])
        emphasized = set(re.findall(r'\*\*([A-Za-z0-9\u4e00-\u9fa5\s\-_]{2,30})\*\*', full_text))
        for term in emphasized:
            term_s = term.strip()
            # 过滤过长句子或纯标点/数字
            if 2 <= len(term_s) <= 25 and not term_s.isdigit():
                entities_to_create.add(term_s)
                if len(entities_to_create) >= 15:
                    break

        for entity in entities_to_create:
            safe_c = re.sub(r'[\\/:*?"<>|]+', '_', entity)[:30].strip()
            card_path = self.concept_cards_dir / f"{safe_c}.md"
            if not card_path.exists():
                card_content = f"""---
title: "{entity}"
created: {time.strftime('%Y-%m-%d %H:%M:%S')}
type: 概念原子实体
tags:
  - 概念卡片
  - 学术实体
---

# 📌 {entity}

## 概述
`[[{entity}]]` 是在研讨会话 `[[{curr_topic}]]` 中引用的核心学术概念。

## 关联研讨记录
- [[{curr_topic}]]
- 所属研究领域：[[{session.domain.upper()}]]
"""
                card_path.write_text(card_content, encoding="utf-8")
                created.append(str(card_path))

        return created
