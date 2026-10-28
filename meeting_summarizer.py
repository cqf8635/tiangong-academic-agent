"""Meeting & Lecture Summarization and Action Item Extraction Engine.

Features:
1. Chunked Map-Reduce summarization: Solves long context overflow for hours of meeting logs.
2. Structured extraction: Core Decisions, Action Items (Who/What/When), Key Discussion Points.
3. Automated Report Generation: Generates clean, professional Markdown meeting minutes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from config_utils import get_default_llm_model, get_llm_timeout
from prompts import build_meeting_reduce_system_prompt

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)


class ActionItem(BaseModel):
    task: str = Field(description="具体的待办任务或作业内容")
    assignee: str = Field(description="负责人/责任方（如明确指派）或 全员/学生")
    deadline: str = Field(description="截止时间或交付时间节点（如未提则为 待定）")
    priority: str = Field(description="优先级: 高 (High) | 中 (Medium) | 低 (Low)")


class MeetingSummaryReport(BaseModel):
    title: str = Field(description="会议或课程讲座的简明主题标题")
    overview: str = Field(description="全局核心摘要（300~500字精炼总结）")
    key_decisions: List[str] = Field(description="达成的关键共识、定理结论或决策列表")
    discussion_topics: List[Dict[str, str]] = Field(
        description="主要讨论议题与核心观点列表，包含 topic 和 summary"
    )
    action_items: List[ActionItem] = Field(description="提取的待办事项与作业清单")


class MeetingSummarizer:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = get_default_llm_model()

        if not self.api_key or not self.base_url:
            raise ValueError(f"[错误] 未在 {ENV_PATH} 中正确找到 OPENAI_API_KEY 或 OPENAI_BASE_URL 配置！")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _read_records(self, record_path: Path) -> List[Dict[str, Any]]:
        """读取 JSONL 格式的流水日志"""
        records = []
        if not record_path.exists():
            return records

        with open(record_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line.strip()))
                    except Exception:
                        pass
        return records

    def summarize_session(
        self,
        record_path: str | Path,
        output_report_path: Optional[Path] = None,
        chunk_sentences: int = 20,
    ) -> MeetingSummaryReport:
        """
        对长会议/讲座流水执行分段 Map-Reduce 归纳与待办抽取
        """
        path = Path(record_path)
        records = self._read_records(path)

        if not records:
            # 兼容纯文本输入
            if path.exists():
                text_content = path.read_text(encoding="utf-8")
                records = [{"index": 1, "timestamp": "00:00:00", "source": "input", "text": text_content}]
            else:
                raise FileNotFoundError(f"未找到记录文件: {path}")

        print(f"📊 [MeetingSummarizer] 正在处理会话流水: {path.name} (共 {len(records)} 句发言)...")

        # 1. 如果流水较长，分块执行 Map 局部微摘要
        chunks = [records[i : i + chunk_sentences] for i in range(0, len(records), chunk_sentences)]
        map_summaries = []

        if len(chunks) > 1:
            print(f"✂️  [Map-Phase] 流水较长，分为 {len(chunks)} 个时间切片并行压缩...")
            for idx, chunk in enumerate(chunks, 1):
                chunk_text = "\n".join([f"[{r.get('timestamp')}] {r.get('text')}" for r in chunk])
                prompt = f"请简明总结以下时间段的讨论要点（保留关键观点与待办提及）：\n\n{chunk_text}\n\n请直接输出一小段总结（必须包含 JSON 格式输出）。"
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        timeout=get_llm_timeout(60.0),
                    )
                    map_summaries.append(f"【切片 {idx} 摘要】: " + resp.choices[0].message.content)
                except Exception:
                    map_summaries.append(chunk_text)
            combined_context = "\n\n".join(map_summaries)
        else:
            combined_context = "\n".join([f"[{r.get('timestamp')}] {r.get('text')}" for r in records])

        # 2. Reduce 全局聚合：生成结构化报告
        print("🧠 [Reduce-Phase] 正在由大模型生成全局决议与待办报告...")
        system_prompt = build_meeting_reduce_system_prompt()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"以下是完整的流水/切片摘要内容，请输出最终纪要 JSON：\n\n{combined_context}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=get_llm_timeout(90.0),
        )

        raw_json = response.choices[0].message.content
        data = json.loads(raw_json)
        report = MeetingSummaryReport(**data)

        # 3. 输出美化排版的 Markdown 报告文件
        if output_report_path is None:
            output_report_path = path.with_name(f"summary_{path.stem}.md")

        md_content = self._render_markdown_report(report)
        output_report_path.write_text(md_content, encoding="utf-8")
        print(f"✅ [MeetingSummarizer] 报告已生成并落盘: {output_report_path}")

        return report

    def _render_markdown_report(self, report: MeetingSummaryReport) -> str:
        """渲染为规范清晰的 Markdown 格式"""
        lines = [
            f"# 📋 {report.title}\n",
            f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## 📌 一、全局核心摘要",
            f"{report.overview}\n",
            "## 🎯 二、关键决议与定理结论",
        ]
        for idx, dec in enumerate(report.key_decisions, 1):
            lines.append(f"{idx}. {dec}")

        lines.append("\n## 💬 三、核心议题讨论脉络")
        for topic in report.discussion_topics:
            lines.append(f"### 🔹 {topic.get('topic')}")
            lines.append(f"{topic.get('summary')}\n")

        lines.append("## ✅ 四、待办事项清单 (Action Items)")
        lines.append("| 序号 | 待办任务 / 作业 | 责任人 | 截止时间 | 优先级 |")
        lines.append("| :--- | :--- | :--- | :--- | :---: |")
        for idx, item in enumerate(report.action_items, 1):
            lines.append(f"| {idx} | **{item.task}** | `{item.assignee}` | `{item.deadline}` | `{item.priority}` |")

        return "\n".join(lines)
