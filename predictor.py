# -*- coding: utf-8 -*-
"""排列五预测器 - 爬取近50期数据并基于多特征加权评分给出预测"""
import json
import re
import math
import threading
import tkinter as tk
import unicodedata
from tkinter import ttk, scrolledtext, messagebox
from collections import Counter
from itertools import combinations


def _vwidth(s: str) -> int:
    """字符串视觉宽度（CJK 字符算 2，ASCII 算 1）。"""
    w = 0
    for c in s:
        if unicodedata.east_asian_width(c) in ("F", "W"):
            w += 2
        else:
            w += 1
    return w


def _vpad(s: str, width: int, align: str = "left") -> str:
    """按视觉宽度填充。"""
    pad = max(0, width - _vwidth(s))
    if align == "right":
        return " " * pad + s
    return s + " " * pad

import requests


API_URL = "https://jc.zhcw.com/port/client_json.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.zhcw.com/kjxx/pl5/",
}

# 每元投注的赔付倍数
PAYOUT_RATIO = {
    "二定": 96,
    "三定": 960,
    "四定": 9600,
    "二现": 9,
    "三现": 45,
    "四现": 320,
}

# 现玩法理论概率：选 N 个不同数字，全部出现在 4 位开奖号中
# 用容斥原理：P = 1 - C(N,1)(9/10)^4 + C(N,2)(8/10)^4 - ...
PROB_XIAN = {
    2: 0.0974,
    3: 0.0204,
    4: 0.0024,
}

# 风险偏好预设：三档不同的预算分配策略
# 保守：偏向高命中率、低波动（二现/二定为主）
# 平衡：六种玩法雨露均沾
# 激进：偏向高赔付、低概率（四定/四现为主，搏大奖）
RISK_PROFILES = {
    "保守": {
        "二定包码": 0.30,
        "三定包码": 0.10,
        "四定包码": 0.00,
        "二现":     0.40,
        "三现":     0.20,
        "四现":     0.00,
    },
    "平衡": {
        "二定包码": 0.15,
        "三定包码": 0.25,
        "四定包码": 0.15,
        "二现":     0.10,
        "三现":     0.25,
        "四现":     0.10,
    },
    "激进": {
        "二定包码": 0.05,
        "三定包码": 0.20,
        "四定包码": 0.40,
        "二现":     0.00,
        "三现":     0.10,
        "四现":     0.25,
    },
}

RISK_DESC = {
    "保守": "高命中率优先 — 二现(9.74%)+二定包码(9%)为主，单注小额、回报稳定",
    "平衡": "六种玩法均衡分配 — 兼顾命中率与赔付倍数",
    "激进": "高赔付搏大奖 — 四定(9600倍)+四现(320倍)为主，命中率低但单中收益高",
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
# 预算分配
# ============================================================
def calculate_budget_plans(budget: float, rec: dict, risk: str = "平衡"):
    """根据预算、推荐方案和风险偏好，给每种玩法分配投注金额。

    risk: "保守" / "平衡" / "激进"，对应不同的预算分配权重。
    返回 dict: 玩法名 -> {倍数, 组合数, 实际投入, 命中概率, 中奖金额, 单注赔付}
    特殊键 __total__ 存放合计实际投入，__risk__ 存放使用的风险档位。
    """
    if budget <= 0:
        return {"__total__": 0.0, "__risk__": risk}
    weights = RISK_PROFILES.get(risk, RISK_PROFILES["平衡"])

    # 各定位包码的组合数
    bao_combos = {}
    for name in ["二定", "三定", "四定"]:
        n = 1
        for _, ds in rec[name]["包码"]:
            n *= len(ds)
        bao_combos[name] = n

    # 每种"投注单元"的元数据：单份成本 / 单注赔付 / 命中概率（命中即至少一注中）
    schemes = {
        "二定包码": {
            "组合数": bao_combos["二定"],
            "单份成本": round(bao_combos["二定"] * 0.1, 2),
            "单注赔付": 0.1 * PAYOUT_RATIO["二定"],
            "命中概率": bao_combos["二定"] / 100.0,
        },
        "三定包码": {
            "组合数": bao_combos["三定"],
            "单份成本": round(bao_combos["三定"] * 0.1, 2),
            "单注赔付": 0.1 * PAYOUT_RATIO["三定"],
            "命中概率": bao_combos["三定"] / 1000.0,
        },
        "四定包码": {
            "组合数": bao_combos["四定"],
            "单份成本": round(bao_combos["四定"] * 0.1, 2),
            "单注赔付": 0.1 * PAYOUT_RATIO["四定"],
            "命中概率": bao_combos["四定"] / 10000.0,
        },
        "二现": {
            "组合数": 1,
            "单份成本": 1.0,
            "单注赔付": float(PAYOUT_RATIO["二现"]),
            "命中概率": PROB_XIAN[2],
        },
        "三现": {
            "组合数": 1,
            "单份成本": 1.0,
            "单注赔付": float(PAYOUT_RATIO["三现"]),
            "命中概率": PROB_XIAN[3],
        },
        "四现": {
            "组合数": 1,
            "单份成本": 1.0,
            "单注赔付": float(PAYOUT_RATIO["四现"]),
            "命中概率": PROB_XIAN[4],
        },
    }

    plans = {}
    total = 0.0
    for play, weight in weights.items():
        if weight <= 0:
            continue
        s = schemes[play]
        target = budget * weight
        multiples = int(target / s["单份成本"])
        if multiples < 1:
            continue
        cost = round(multiples * s["单份成本"], 2)
        payout = round(multiples * s["单注赔付"], 2)
        plans[play] = {
            "倍数": multiples,
            "组合数": s["组合数"],
            "单份成本": s["单份成本"],
            "实际投入": cost,
            "命中概率": s["命中概率"],
            "单注赔付": s["单注赔付"],
            "中奖金额": payout,
            "净收益": round(payout - cost, 2),
        }
        total += cost

    # 预算太小时的兜底：保证至少有一份最便宜的玩法
    if not plans:
        candidates = sorted(schemes.items(), key=lambda x: x[1]["单份成本"])
        for play, s in candidates:
            multiples = int(budget / s["单份成本"])
            if multiples >= 1:
                cost = round(multiples * s["单份成本"], 2)
                payout = round(multiples * s["单注赔付"], 2)
                plans[play] = {
                    "倍数": multiples,
                    "组合数": s["组合数"],
                    "单份成本": s["单份成本"],
                    "实际投入": cost,
                    "命中概率": s["命中概率"],
                    "单注赔付": s["单注赔付"],
                    "中奖金额": payout,
                    "净收益": round(payout - cost, 2),
                }
                total += cost
                break

    plans["__total__"] = round(total, 2)
    plans["__risk__"] = risk
    return plans


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

        self.risk_var = tk.StringVar(value="平衡")
        self.combo_risk = ttk.Combobox(
            top, textvariable=self.risk_var,
            values=list(RISK_PROFILES.keys()),
            state="readonly", width=6,
            font=("微软雅黑", 10))
        self.combo_risk.pack(side=tk.RIGHT, padx=(4, 12))
        tk.Label(top, text="风险偏好：", font=("微软雅黑", 10),
                 bg="#f5f5f5", fg="#555").pack(side=tk.RIGHT)

        tk.Label(top, text="元", font=("微软雅黑", 10),
                 bg="#f5f5f5", fg="#555").pack(side=tk.RIGHT, padx=(2, 12))
        self.budget_var = tk.StringVar(value="100")
        self.entry_budget = tk.Entry(
            top, textvariable=self.budget_var,
            font=("Consolas", 11), width=8, justify="right",
            relief=tk.SOLID, bd=1)
        self.entry_budget.pack(side=tk.RIGHT, padx=(4, 2))
        tk.Label(top, text="本期投入：", font=("微软雅黑", 10),
                 bg="#f5f5f5", fg="#555").pack(side=tk.RIGHT)

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

    def _read_budget(self):
        raw = self.budget_var.get().strip()
        if not raw:
            return 0.0
        try:
            v = float(raw)
            if v < 0:
                raise ValueError
            return v
        except ValueError:
            messagebox.showwarning("金额无效", "请输入大于等于 0 的数字作为本期投入金额。")
            return None

    def on_predict(self):
        budget = self._read_budget()
        if budget is None:
            return
        self.budget = budget
        self.risk = self.risk_var.get() or "平衡"
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
            budget_plans = calculate_budget_plans(self.budget, rec, self.risk)

            self.after(0, lambda: self._render(history, rec, pos_scores, digit_scores, budget_plans))
            self.after(0, lambda: self.set_status(f"完成 — 已分析 {len(history)} 期", "#27ae60"))
        except Exception as e:
            err = str(e)
            self.after(0, lambda: self.set_status(f"失败：{err}", "#c0392b"))
            self.after(0, lambda: messagebox.showerror("出错了", f"爬取或分析失败：\n{err}"))
        finally:
            self.after(0, lambda: self.btn_predict.config(state=tk.NORMAL, text="预测"))

    def _render(self, history, rec, pos_scores, digit_scores, budget_plans):
        self.clear()
        latest = history[0]

        self.append(f"最新一期 {latest['issue']} ({latest['date']})  开奖号码：", "h2")
        self.append("".join(str(x) for x in latest["nums"][:4]) + " ", "digit")
        self.append(f"(后位 {latest['nums'][4]})\n\n", "dim")

        # ========== 预算分配方案 ==========
        budget_total = budget_plans.get("__total__", 0.0)
        risk = budget_plans.get("__risk__", "平衡")
        if budget_total > 0:
            self.append(f"【预算分配方案 — 本期投入 ¥{self.budget:.2f}  风险偏好：{risk}】\n", "h1")
            self.append(f"{RISK_DESC.get(risk, '')}\n", "hint")
            self.append(f"实际投注合计 ¥{budget_total:.2f}（剩余 ¥{self.budget - budget_total:.2f}，因最小注金限制无法整除）\n\n", "hint")

            cols = [
                ("玩法",         8, "left"),
                ("投注金额",     9, "right"),
                ("倍数×组合数", 10, "left"),
                ("命中概率",     8, "right"),
                ("若中可得",    10, "right"),
                ("净收益",      10, "right"),
            ]
            sep = "  "
            header = sep.join(_vpad(name, w, a) for name, w, a in cols)
            self.append(header + "\n", "hint")
            self.append("─" * _vwidth(header) + "\n", "dim")

            order = ["二定包码", "三定包码", "四定包码", "二现", "三现", "四现"]
            for play in order:
                if play not in budget_plans:
                    continue
                p = budget_plans[play]
                cells = [
                    play,
                    f"¥{p['实际投入']:.2f}",
                    f"{p['倍数']}×{p['组合数']}",
                    f"{p['命中概率']*100:.3f}%",
                    f"¥{p['中奖金额']:.2f}",
                    f"¥{p['净收益']:+.2f}",
                ]
                line = sep.join(_vpad(v, cols[i][1], cols[i][2]) for i, v in enumerate(cells))
                self.append(line + "\n")
            self.append(
                "\n说明：定位包码每注 0.1 元，赔率同比降10倍；现玩法每注 1 元。"
                "命中概率指至少有一注命中的概率（基于推荐号选中真实号的假设上限）。\n\n",
                "hint")
        elif self.budget > 0:
            self.append(f"【预算 ¥{self.budget:.2f} 过小，无法分配任何玩法】\n\n", "h1")

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
