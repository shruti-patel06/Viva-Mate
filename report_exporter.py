"""
VivaMate - Report & Transcript Exporter
Generates downloadable Markdown reports and CSV transcripts.
"""

import csv
import io
from datetime import datetime
from typing import List


def generate_markdown_report(
    student_info: dict,
    subject: str,
    scores: List[int],
    answered: int,
    skipped: int,
    transcript: List[dict],
    ai_report: str,
) -> str:
    avg_score = sum(scores) / len(scores) if scores else 0.0
    total = answered + skipped
    completion = (answered / total * 100) if total > 0 else 0.0
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# 🎓 VivaMate – AI-Powered Adaptive Viva Examination Report",
        "",
        f"**Generated:** {timestamp}",
        "",
        "## 👨🎓 Student Information",
        f"- **Name:** {student_info.get('student_name') or 'N/A'}",
        f"- **Roll Number:** {student_info.get('roll_number') or 'N/A'}",
        f"- **Department:** {student_info.get('department') or 'N/A'}",
        f"- **Year:** {student_info.get('year') or 'N/A'}",
        f"- **Section:** {student_info.get('section') or 'N/A'}",
        f"- **Subject:** {subject}",
        "",
        "## 📊 Performance Summary",
        f"- **Average Score:** {avg_score:.2f}/10",
        f"- **Questions Answered:** {answered}",
        f"- **Questions Skipped:** {skipped}",
        f"- **Completion Rate:** {completion:.0f}%",
        "",
        "## 📄 AI Performance Report",
        ai_report or "_Not generated._",
        "",
        "## 📝 Question & Answer Transcript",
    ]

    for idx, item in enumerate(transcript, start=1):
        weaknesses = item.get("weaknesses") or []
        ideal_answer = item.get("ideal_answer") or ""
        lines.extend(
            [
                f"### Question {idx} ({item.get('difficulty', 'N/A')})",
                f"**Q:** {item.get('question', '')}",
                f"**Student Answer:** {item.get('answer', '[Skipped]')}",
                f"**Score:** {item.get('score', 0)}/10",
            ]
        )
        if weaknesses:
            lines.append(f"**Mistakes / Gaps:** {'; '.join(weaknesses)}")
        if ideal_answer:
            lines.append(f"**Ideal Answer:** {ideal_answer}")
        lines.append("")

    return "\n".join(lines)


def generate_csv_transcript(
    student_info: dict, subject: str, transcript: List[dict]
) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "Student Name",
            "Roll Number",
            "Department",
            "Year",
            "Subject",
            "Question #",
            "Difficulty",
            "Question",
            "Student Answer",
            "Score",
            "Status",
        ]
    )

    for idx, item in enumerate(transcript, start=1):
        writer.writerow(
            [
                student_info.get("student_name", ""),
                student_info.get("roll_number", ""),
                student_info.get("department", ""),
                student_info.get("year", ""),
                subject,
                idx,
                item.get("difficulty", ""),
                item.get("question", ""),
                item.get("answer", ""),
                item.get("score", 0),
                item.get("status", "Answered"),
            ]
        )

    return output.getvalue()