#!/usr/bin/env python3
import json
import re
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

BLOG_ROOT = Path('/home/fxlin/projects/fxlin1933.github.io')
DATA_FILE = BLOG_ROOT / 'data' / 'daily-discovery.json'
LOG_FILE = BLOG_ROOT / 'data' / 'daily-discovery-last-run.json'

FEEDS = [
    ('Simon Willison', 'https://simonwillison.net/atom/everything/'),
    ('Julia Evans', 'https://jvns.ca/atom.xml'),
]

NS = {'a': 'http://www.w3.org/2005/Atom'}

# 以后把这里替换成你的本地模型命令即可。
# 例如：
# MODEL_CMD = ['/home/fxlin/projects/LiteInfer/build/models/llama3_infer', '...']
MODEL_CMD = None


@dataclass
class Item:
    source: str
    title: str
    url: str
    date: str
    summary: str


def clean_html(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text or '')
    text = unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def fetch_feed(name: str, url: str) -> list[Item]:
    xml = urllib.request.urlopen(url, timeout=20).read()
    root = ET.fromstring(xml)
    out: list[Item] = []
    for entry in root.findall('a:entry', NS)[:3]:
        title = (entry.findtext('a:title', default='', namespaces=NS) or '').strip()
        link = entry.find('a:link', NS)
        href = link.attrib.get('href', '') if link is not None else ''
        updated = (entry.findtext('a:updated', default='', namespaces=NS) or '')[:10]
        summary = entry.findtext('a:summary', default='', namespaces=NS) or entry.findtext('a:content', default='', namespaces=NS) or ''
        summary = clean_html(summary)
        if title and href:
            out.append(Item(name, title, href, updated, summary[:1200]))
    return out


def collect_items() -> list[Item]:
    items: list[Item] = []
    for name, url in FEEDS:
        try:
            items.extend(fetch_feed(name, url))
        except Exception as e:
            print(f'[warn] feed failed: {name}: {e}', file=sys.stderr)
    return items


def build_prompt(items: list[Item]) -> str:
    blocks = []
    for i, item in enumerate(items[:5], start=1):
        blocks.append(
            f'''[Item {i}]\nsource: {item.source}\ntitle: {item.title}\nurl: {item.url}\ndate: {item.date}\nsummary: {item.summary}\n'''
        )
    return (
        '你是一个克制、简洁的技术博客助手。\n'
        '请从下面候选中选出今天最适合放在博客首页“今日发现”的 1 条。\n'
        '输出 JSON，字段只有: title, source, url, note。\n'
        '其中 note 用中文，80~120 字，自然、克制，不要营销腔。\n\n'
        + '\n'.join(blocks)
    )


def simulate_model_output(items: list[Item]) -> dict:
    item = items[0]
    if 'beats' in item.title.lower() or 'note' in item.title.lower():
        note = '今天看到一个挺对味的小更新：给博客里的外部内容流补上 notes 之后，那些原本只是顺手收录的链接，开始带上一点上下文和态度。对个人站点来说，这种轻量但持续的表达，比硬凑长文更有生命力。'
    else:
        note = f'今天翻到一篇挺值得点开的内容：{item.title}。它不只是给信息，还把问题讲得更清楚一些，属于那种看完会顺手记一笔、以后大概率还会再翻回来的东西。'
    return {
        'title': item.title,
        'source': item.source,
        'url': item.url,
        'note': note,
    }


def run_model(prompt: str, items: list[Item]) -> dict:
    if not MODEL_CMD:
        return simulate_model_output(items)
    proc = subprocess.run(MODEL_CMD, input=prompt, text=True, capture_output=True, check=True)
    return json.loads(proc.stdout)


def main() -> int:
    items = collect_items()
    if not items:
        raise SystemExit('no feed items fetched')

    prompt = build_prompt(items)
    picked = run_model(prompt, items)

    payload = {
        'date': datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d'),
        'title': picked['title'],
        'source': picked['source'],
        'url': picked['url'],
        'note': picked['note'],
    }
    DATA_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    LOG_FILE.write_text(json.dumps({
        'updated_at': datetime.now().astimezone().isoformat(),
        'item_count': len(items),
        'prompt_preview': prompt[:1200],
        'selected': payload,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
