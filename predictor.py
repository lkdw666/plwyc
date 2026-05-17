# -*- coding: utf-8 -*-
"""排列五预测器 - 爬取近50期数据并基于多特征加权评分给出预测"""
import json
import re
import math
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from collections import Counter
from itertools import combinations

import requests


API_URL = "https://jc.zhcw.com/port/client_json.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.zhcw.com/kjxx/pl5/",
}


# ============================================================
# 数据爬取
# ============================================================
def fetch_history(count: int = 50):
    """从中彩网爬取最近 count 期排列五开奖数据。

    返回列表，按期号从新到旧排列：
        [{"issue": "26127", "date": "2026-05-17", "nums": [9,3,6,0,3]}, ...]
    """
    params = {
        "transactionType": "10001001",
        "lotteryId": "284",  # 排列五
        "issueCount": str(count),
        "startIssue": "",
        "endIssue": "",
        "startDate": "",
        "endDate": "",
        "type": "0",
        "pageNum": "1",
        "pageSize": str(count),
        "tt": "0.123",
        "callback": "cb",
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    text = resp.text.strip()
    m = re.match(r"^\w+\((.*)\)\s*;?\s*$", text, re.DOTALL)
    if not m:
        raise ValueError("接口响应不是预期的 JSONP 格式")
    payload = json.loads(m.group(1))

    rows = payload.get("data", []) or []
    history = []
    for row in rows:
        nums_raw = row.get("frontWinningNum", "").strip()
        parts = nums_raw.split()
        if len(parts) != 5 or not all(p.isdigit() for p in parts):
            continue
        history.append({
            "issue": row.get("issue", ""),
            "date": row.get("openTime", ""),
            "nums": [int(x) for x in parts],
        })
    if not history:
        raise ValueError("未取到任何开奖数据")
    return history


# ============================================================
# 预测算法
# ============================================================
class Predictor:
    """多特征加权评分预测器。

    针对每个位置 (千/百/十/个) 给每个数字 0-9 打分，分数综合：
      1. 长期频率   (50期)
      2. 短期热度   (近10期)
      3. 遗漏值反弹 (上次出现距今的期数)
      4. 一阶马尔可夫转移概率 (上一期同位置数字 -> 本期数字)
    """

    POS_NAMES = ["千位", "百位", "十位", "个位"]  # 排列五取前4位
    SHORT_WINDOW = 10
    WEIGHTS = {
        "freq_long":  0.25,
        "freq_short": 0.30,
        "miss":       0.20,
        "markov":     0.25,
    }

    def __init__(self, history):
        if not history:
            raise ValueError("历史数据为空")
        # history 是从新到旧，转成从旧到新便于建模
        self.history = list(reversed(history))
        self.n = len(self.history)
        # 只取前4位作为千百十个
        self.draws = [h["nums"][:4] for h in self.history]

    def _freq_scores(self, draws):
        """计算 4 个位置的频率分布（每位返回 list[10]，已归一化为 0-1）"""
        scores = []
        total = max(len(draws), 1)
        for pos in range(4):
            cnt = Counter(d[pos] for d in draws)
            row = [cnt.get(d, 0) / total for d in range(10)]
            scores.append(self._normalize(row))
        return scores

    def _miss_scores(self):
        """每个位置每个数字的遗漏值（距今多少期没出现）"""
        scores = []
        for pos in range(4):
            row = []
            for d in range(10):
                miss = self.n  # 默认整段都没出
                for i in range(self.n - 1, -1, -1):
                    if self.draws[i][pos] == d:
                        miss = self.n - 1 - i
                        break
                row.append(miss)
            scores.append(self._normalize(row))
        return scores

    def _markov_scores(self):
        """一阶马尔可夫：基于上一期同位置数字 last_d 的转移概率 P(d | last_d)。
        使用 Laplace 平滑避免零概率。"""
        scores = []
        last_draw = self.draws[-1]
        for pos in range(4):
            transitions = [[1] * 10 for _ in range(10)]  # Laplace 平滑
            for i in range(self.n - 1):
                a = self.draws[i][pos]
                b = self.draws[i + 1][pos]
                transitions[a][b] += 1
            last_d = last_draw[pos]
            row_total = sum(transitions[last_d])
            row = [transitions[last_d][d] / row_total for d in range(10)]
            scores.append(self._normalize(row))
        return scores

    @staticmethod
    def _normalize(row):
        lo, hi = min(row), max(row)
        if hi - lo < 1e-9:
            return [0.5] * len(row)
        return [(x - lo) / (hi - lo) for x in row]

    def position_scores(self):
        """返回 4 个位置每个数字 0-9 的综合评分 list[4][10]"""
        long_s = self._freq_scores(self.draws)
        short_s = self._freq_scores(self.draws[-self.SHORT_WINDOW:])
        miss_s = self._miss_scores()
        mk_s = self._markov_scores()
        w = self.WEIGHTS
        result = []
        for pos in range(4):
            row = []
            for d in range(10):
                s = (w["freq_long"]  * long_s[pos][d] +
                     w["freq_short"] * short_s[pos][d] +
                     w["miss"]       * miss_s[pos][d] +
                     w["markov"]     * mk_s[pos][d])
                row.append(s)
            result.append(row)
        return result

    def global_digit_scores(self):
        """全局数字评分（用于"现"玩法），基于：出现频次 + 短期热度 + 整体遗漏。"""
        long_cnt = Counter()
        for d in self.draws:
            long_cnt.update(d)
        short_cnt = Counter()
        for d in self.draws[-self.SHORT_WINDOW:]:
            short_cnt.update(d)
        miss = []
        for d in range(10):
            m = self.n
            for i in range(self.n - 1, -1, -1):
                if d in self.draws[i]:
                    m = self.n - 1 - i
                    break
            miss.append(m)

        long_n = self._normalize([long_cnt.get(d, 0) for d in range(10)])
        short_n = self._normalize([short_cnt.get(d, 0) for d in range(10)])
        miss_n = self._normalize(miss)
        return [0.4 * long_n[d] + 0.4 * short_n[d] + 0.2 * miss_n[d] for d in range(10)]


# ============================================================
# 推荐生成
# ============================================================
def make_recommendations(predictor: Predictor):
    """基于评分生成 6 种玩法的推荐方案。"""
    pos_scores = predictor.position_scores()
    digit_scores = predictor.global_digit_scores()

    # 每位 top 排序
    top_per_pos = []
    for pos in range(4):
        ranked = sorted(range(10), key=lambda d: pos_scores[pos][d], reverse=True)
        top_per_pos.append(ranked)

    # 每位最佳单数字（用于定位单码）
    best_digit_each_pos = [tp[0] for tp in top_per_pos]

    # 全局数字按分排序
    digit_ranked = sorted(range(10), key=lambda d: digit_scores[d], reverse=True)

    rec = {}
    pos_names = Predictor.POS_NAMES

    # —— 二定：选评分最高的 2 个位置，每位取 top1
    pos_strength = [(pos, max(pos_scores[pos])) for pos in range(4)]
    pos_strength.sort(key=lambda x: x[1], reverse=True)
    two_def_positions = sorted([pos_strength[0][0], pos_strength[1][0]])
    rec["二定"] = {
        "单码": [(pos_names[p], best_digit_each_pos[p]) for p in two_def_positions],
        "包码": [(pos_names[p], top_per_pos[p][:3]) for p in two_def_positions],
    }

    # —— 三定：选评分最高的 3 个位置
    three_def_positions = sorted([pos_strength[i][0] for i in range(3)])
    rec["三定"] = {
        "单码": [(pos_names[p], best_digit_each_pos[p]) for p in three_def_positions],
        "包码": [(pos_names[p], top_per_pos[p][:3]) for p in three_def_positions],
    }

    # —— 四定：4 个位置全占
    rec["四定"] = {
        "单码": [(pos_names[p], best_digit_each_pos[p]) for p in range(4)],
        "包码": [(pos_names[p], top_per_pos[p][:2]) for p in range(4)],
    }

    # —— 现玩法：取全局 top N 个不同数字
    rec["二现"] = digit_ranked[:2]
    rec["三现"] = digit_ranked[:3]
    rec["四现"] = digit_ranked[:4]

    return rec, pos_scores, digit_scores


# ============================================================
# GUI
# ============================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("排列五预测器")
        self.geometry("780x640")
        self.configure(bg="#f5f5f5")

        try:
            self.iconbitmap(default="")
        except Exception:
            pass

        self._build_ui()
        self.history = None

    def _build_ui(self):
        top = tk.Frame(self, bg="#f5f5f5")
        top.pack(fill=tk.X, padx=14, pady=10)

        title = tk.Label(top, text="排列五预测器",
                         font=("微软雅黑", 16, "bold"),
                         bg="#f5f5f5", fg="#222")
        title.pack(side=tk.LEFT)

        self.btn_predict = tk.Button(
            top, text="预测", font=("微软雅黑", 11, "bold"),
            width=10, height=1, bg="#e74c3c", fg="white",
            activebackground="#c0392b", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            command=self.on_predict)
        self.btn_predict.pack(side=tk.RIGHT)

        self.status = tk.Label(self, text="就绪 — 点击右上角 [预测] 开始",
                               anchor="w", bg="#ececec", fg="#555",
                               font=("微软雅黑", 9), padx=10)
        self.status.pack(fill=tk.X)

        # 主显示区
        body = tk.Frame(self, bg="#f5f5f5")
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        self.text = scrolledtext.ScrolledText(
            body, font=("Consolas", 10), wrap=tk.WORD,
            bg="white", relief=tk.SOLID, bd=1)
        self.text.pack(fill=tk.BOTH, expand=True)
        self.text.configure(state=tk.DISABLED)

        self._tag_setup()

    def _tag_setup(self):
        self.text.tag_configure("h1", font=("微软雅黑", 13, "bold"), foreground="#c0392b", spacing3=6)
        self.text.tag_configure("h2", font=("微软雅黑", 11, "bold"), foreground="#2c3e50", spacing3=4)
        self.text.tag_configure("hint", font=("微软雅黑", 9), foreground="#888")
        self.text.tag_configure("num", font=("Consolas", 11, "bold"), foreground="#e74c3c")
        self.text.tag_configure("digit", font=("Consolas", 14, "bold"), foreground="#e74c3c")
        self.text.tag_configure("ok", foreground="#27ae60")
        self.text.tag_configure("dim", foreground="#999")

    def set_status(self, msg, color="#555"):
        self.status.config(text=msg, fg=color)

    def append(self, text, tag=None):
        self.text.configure(state=tk.NORMAL)
        if tag:
            self.text.insert(tk.END, text, tag)
        else:
            self.text.insert(tk.END, text)
        self.text.configure(state=tk.DISABLED)
        self.text.see(tk.END)

    def clear(self):
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.configure(state=tk.DISABLED)

    def on_predict(self):
        self.btn_predict.config(state=tk.DISABLED, text="爬取中...")
        self.set_status("正在从 zhcw.com 爬取最新50期开奖数据...", "#2980b9")
        self.clear()
        threading.Thread(target=self._do_predict, daemon=True).start()

    def _do_predict(self):
        try:
            history = fetch_history(50)
            self.history = history
            self.after(0, lambda: self.set_status(f"已获取 {len(history)} 期数据，正在分析...", "#2980b9"))

            predictor = Predictor(history)
            rec, pos_scores, digit_scores = make_recommendations(predictor)

            self.after(0, lambda: self._render(history, rec, pos_scores, digit_scores))
            self.after(0, lambda: self.set_status(f"完成 — 已分析 {len(history)} 期", "#27ae60"))
        except Exception as e:
            err = str(e)
            self.after(0, lambda: self.set_status(f"失败：{err}", "#c0392b"))
            self.after(0, lambda: messagebox.showerror("出错了", f"爬取或分析失败：\n{err}"))
        finally:
            self.after(0, lambda: self.btn_predict.config(state=tk.NORMAL, text="预测"))

    def _render(self, history, rec, pos_scores, digit_scores):
        self.clear()
        latest = history[0]

        self.append(f"最新一期 {latest['issue']} ({latest['date']})  开奖号码：", "h2")
        self.append("".join(str(x) for x in latest["nums"][:4]) + " ", "digit")
        self.append(f"(后位 {latest['nums'][4]})\n\n", "dim")

        # ========== 定位玩法 ==========
        self.append("【定位玩法 — 千百十个位置】\n", "h1")
        self.append("规则：选定位置上中对应数字才算中。包码会拆成多注单码。\n\n", "hint")

        for name in ["二定", "三定", "四定"]:
            data = rec[name]
            self.append(f"▶ {name}\n", "h2")
            single_str = " | ".join(f"{p}={d}" for p, d in data["单码"])
            self.append(f"  推荐单码：{single_str}\n")
            # 显示具体投注号
            self.append(f"  投注示例：", "hint")
            self.append(self._format_single_bet(data["单码"]) + "\n", "num")

            bao_parts = []
            total_combos = 1
            for p, ds in data["包码"]:
                bao_parts.append(f"{p}∈{{{','.join(str(x) for x in ds)}}}")
                total_combos *= len(ds)
            self.append(f"  推荐包码：{' , '.join(bao_parts)}  共 {total_combos} 注\n\n")

        # ========== 现玩法 ==========
        self.append("【现玩法 — 不论位置只看数字出现】\n", "h1")
        self.append("规则：选中的所有数字都要在开奖号码中出现才算中。\n\n", "hint")

        for name, count in [("二现", 2), ("三现", 3), ("四现", 4)]:
            digits = rec[name]
            self.append(f"▶ {name}：", "h2")
            self.append(" ".join(str(d) for d in digits) + "\n", "digit")
            self.append(f"  即买入数字 {{{','.join(str(d) for d in digits)}}}，"
                        f"开奖号码出现这 {count} 个数字（不论位置）即中\n\n", "hint")

        # ========== 评分明细 ==========
        self.append("\n──────────  评分明细（仅参考）  ──────────\n", "dim")
        self.append("各位置 0-9 综合评分（分越高推荐度越高）\n", "hint")
        self.append("位置  " + "   ".join(f" {d} " for d in range(10)) + "\n", "hint")
        for pos in range(4):
            row = pos_scores[pos]
            line = f"{Predictor.POS_NAMES[pos]}  "
            for d in range(10):
                line += f"{row[d]:.2f} "
            self.append(line + "\n")
        self.append("\n全局数字评分（用于现玩法）\n", "hint")
        self.append("数字  " + "   ".join(f" {d} " for d in range(10)) + "\n", "hint")
        line = "评分  "
        for d in range(10):
            line += f"{digit_scores[d]:.2f} "
        self.append(line + "\n")

        # ========== 最近开奖 ==========
        self.append("\n──────────  最近10期开奖  ──────────\n", "dim")
        for h in history[:10]:
            nums = "".join(str(x) for x in h["nums"][:4])
            self.append(f"{h['issue']} ({h['date']})  {nums} ", )
            self.append(f"[后位{h['nums'][4]}]\n", "dim")

        self.append("\n* 提示：本预测基于历史数据统计建模，仅供参考。彩票开奖本质随机，理性购彩。\n", "hint")

    @staticmethod
    def _format_single_bet(items):
        """将 [(位置名, 数字), ...] 格式化为 千百十个 形式（缺失位置打 *）"""
        order = ["千位", "百位", "十位", "个位"]
        m = {pos: d for pos, d in items}
        return " ".join(str(m[p]) if p in m else "*" for p in order)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
