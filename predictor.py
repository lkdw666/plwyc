# -*- coding: utf-8 -*-
"""排列五预测器 - 爬取近50期数据并基于多特征加权评分给出预测"""
import json
import os
import re
import math
import random
import threading
import tkinter as tk
import unicodedata
from tkinter import ttk, scrolledtext, messagebox
from collections import Counter
from itertools import combinations

import pymysql
import requests


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
        "二定单码": 0.00,
        "三定单码": 0.00,
        "四定单码": 0.00,
        "二定包码": 0.30,
        "三定包码": 0.10,
        "四定包码": 0.00,
        "二现":     0.40,
        "三现":     0.20,
        "四现":     0.00,
    },
    "平衡": {
        "二定单码": 0.05,
        "三定单码": 0.05,
        "四定单码": 0.05,
        "二定包码": 0.10,
        "三定包码": 0.20,
        "四定包码": 0.10,
        "二现":     0.10,
        "三现":     0.25,
        "四现":     0.10,
    },
    "激进": {
        "二定单码": 0.00,
        "三定单码": 0.10,
        "四定单码": 0.25,
        "二定包码": 0.05,
        "三定包码": 0.10,
        "四定包码": 0.20,
        "二现":     0.00,
        "三现":     0.10,
        "四现":     0.20,
    },
}

RISK_DESC = {
    "保守": "高命中率优先 — 二现(9.74%)+二定包码(9%)为主，单注小额、回报稳定",
    "平衡": "六种玩法均衡分配 — 兼顾命中率与赔付倍数",
    "激进": "高赔付搏大奖 — 四定(9600倍)+四现(320倍)为主，命中率低但单中收益高",
}

DB_CONFIG = dict(host="localhost", user="root", password="root",
                 database="pl5_predictor", charset="utf8mb4")


# ============================================================
# 数据库
# ============================================================
def db_connect():
    return pymysql.connect(**DB_CONFIG)


def db_save_draws(history):
    """缓存开奖数据（newest-first list），重复期号忽略。"""
    if not history:
        return
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            for h in history:
                nums = h["nums"]
                cur.execute(
                    "INSERT IGNORE INTO draws (issue, open_date, d1, d2, d3, d4, d5) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (h["issue"], h["date"], nums[0], nums[1], nums[2], nums[3], nums[4])
                )
        conn.commit()
    finally:
        conn.close()


def db_save_prediction(target_issue, budget, risk, rec, plans, pos_scores, digit_scores):
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO predictions (target_issue, budget, risk, recommendations, "
                "budget_plans, pos_scores, digit_scores) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (target_issue, budget, risk,
                 json.dumps(rec, ensure_ascii=False),
                 json.dumps(plans, ensure_ascii=False),
                 json.dumps(pos_scores), json.dumps(digit_scores))
            )
        conn.commit()
    finally:
        conn.close()


def db_save_backtest(result):
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            t = result["totals"]
            algo_hits = {p: result["play_stats"][p]["algo_hits"] for p in result["play_stats"]}
            random_hits = {p: result["play_stats"][p]["random_hits"] for p in result["play_stats"]}
            cur.execute(
                "INSERT INTO backtests (train_window, test_periods, budget, risk, "
                "total_cost, total_payout, net_return, roi, algo_hit_rate, random_hit_rate, summary) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (result["train_window"], result["n_test"], result["budget"], result["risk"],
                 t["algo_cost"], t["algo_payout"], t["algo_net"], t["algo_roi"],
                 json.dumps(algo_hits, ensure_ascii=False),
                 json.dumps(random_hits, ensure_ascii=False),
                 json.dumps(result["play_stats"], ensure_ascii=False))
            )
            backtest_id = cur.lastrowid
            for d in result["details"]:
                cur.execute(
                    "INSERT INTO backtest_details (backtest_id, test_issue, actual_nums, "
                    "algo_recommendations, random_recommendations, algo_hits, random_hits, "
                    "cost, payout) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (backtest_id, d["issue"], d["actual"],
                     json.dumps(d["algo_rec"], ensure_ascii=False),
                     json.dumps(d["random_rec"], ensure_ascii=False),
                     json.dumps({k: v["hit"] for k, v in d["algo_eval"]["results"].items()}, ensure_ascii=False),
                     json.dumps({k: v["hit"] for k, v in d["random_eval"]["results"].items()}, ensure_ascii=False),
                     d["algo_eval"]["total_cost"], d["algo_eval"]["total_payout"])
                )
        conn.commit()
        return backtest_id
    finally:
        conn.close()


# ============================================================
# 数据爬取
# ============================================================
def _fetch_api(count: int = 50):
    """API 爬取（fallback 用）。"""
    params = {
        "transactionType": "10001001",
        "lotteryId": "284",
        "issueCount": str(count),
        "startIssue": "", "endIssue": "",
        "startDate": "", "endDate": "",
        "type": "0",
        "pageNum": "1", "pageSize": str(count),
        "tt": "0.123", "callback": "cb",
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


def fetch_history(count: int = 50):
    """读取历史数据。优先从 MySQL 数据库读，失败时 fallback 到 API 爬取。"""
    try:
        from db_utils import load_history
        data = load_history(count)
        if data and len(data) >= min(count, 10):
            return data
    except Exception:
        pass
    return _fetch_api(count)


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
def _select_dynamic(scores, min_n, max_n, threshold):
    """按评分动态选择数字：分数 >= threshold 的全选，限制在 [min_n, max_n] 区间。
    评分平 → 多选（信号弱，分散风险）；评分尖锐 → 少选（信号强，集中）。"""
    ranked = sorted(range(10), key=lambda d: scores[d], reverse=True)
    selected = [d for d in ranked if scores[d] >= threshold]
    if len(selected) < min_n:
        selected = ranked[:min_n]
    elif len(selected) > max_n:
        selected = selected[:max_n]
    return selected


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
        "包码": [(pos_names[p], _select_dynamic(pos_scores[p], 3, 6, 0.55))
                 for p in two_def_positions],
    }

    # —— 三定：选评分最高的 3 个位置
    three_def_positions = sorted([pos_strength[i][0] for i in range(3)])
    rec["三定"] = {
        "单码": [(pos_names[p], best_digit_each_pos[p]) for p in three_def_positions],
        "包码": [(pos_names[p], _select_dynamic(pos_scores[p], 3, 6, 0.55))
                 for p in three_def_positions],
    }

    # —— 四定：4 个位置全占（成本对位置数敏感，阈值更严、上限更低）
    rec["四定"] = {
        "单码": [(pos_names[p], best_digit_each_pos[p]) for p in range(4)],
        "包码": [(pos_names[p], _select_dynamic(pos_scores[p], 2, 4, 0.6))
                 for p in range(4)],
    }

    # —— 现玩法：取全局 top N 个不同数字
    rec["二现"] = digit_ranked[:2]
    rec["三现"] = digit_ranked[:3]
    rec["四现"] = digit_ranked[:4]

    return rec, pos_scores, digit_scores


def make_custom_recommendations(predictor: Predictor, config: dict):
    """按用户自定义配置生成推荐方案。

    config 结构:
        enabled: set[str] — 启用的玩法名 (如 {"二定包码", "三定单码", "二现"})
        bao_pos: dict[str, list[int]] — 各定位包码的位置选择数量
            {"二定": [3,3,0,0], "三定": [3,3,3,0], "四定": [2,2,2,2]}
            0 表示该位置不参与，1-9 表示该位置取 top-N
        xian_manual: dict[str, list[int]] — 现玩法手动指定数字（可选）
            {"二现": [3, 7]}; 未指定则用算法 top-N
    """
    pos_scores = predictor.position_scores()
    digit_scores = predictor.global_digit_scores()
    pos_names = Predictor.POS_NAMES

    top_per_pos = []
    for pos in range(4):
        ranked = sorted(range(10), key=lambda d: pos_scores[pos][d], reverse=True)
        top_per_pos.append(ranked)
    best_digit_each_pos = [tp[0] for tp in top_per_pos]
    digit_ranked = sorted(range(10), key=lambda d: digit_scores[d], reverse=True)

    enabled = config.get("enabled", set())
    bao_pos = config.get("bao_pos", {})
    xian_manual = config.get("xian_manual", {})

    rec = {}
    for name in ["二定", "三定", "四定"]:
        counts = bao_pos.get(name, [0, 0, 0, 0])
        user_active = [p for p in range(4) if counts[p] > 0]

        # 包码：用用户配置的活跃位置
        rec[name] = {
            "包码": [(pos_names[p], top_per_pos[p][:counts[p]]) for p in user_active],
        }

    # 单码：按整体位置强度选 top-N 个位置（不受 bao_pos 影响）
    # 先按综合评分排位置强弱
    pos_avg = [sum(pos_scores[p]) / 10 for p in range(4)]
    pos_ranked = sorted(range(4), key=lambda p: pos_avg[p], reverse=True)
    need_positions = {"二定": 2, "三定": 3, "四定": 4}
    for name in ["二定", "三定", "四定"]:
        n = need_positions[name]
        selected = pos_ranked[:n]
        rec[name]["单码"] = [(pos_names[p], best_digit_each_pos[p]) for p in selected]

    for name, default_n in [("二现", 2), ("三现", 3), ("四现", 4)]:
        manual = xian_manual.get(name)
        if manual and len(manual) == default_n:
            rec[name] = list(manual)
        else:
            rec[name] = digit_ranked[:default_n]

    return rec, pos_scores, digit_scores, enabled


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
        "二定单码": {
            "组合数": 1,
            "单份成本": 1.0,
            "单注赔付": float(PAYOUT_RATIO["二定"]),
            "命中概率": 1 / 100.0,
        },
        "三定单码": {
            "组合数": 1,
            "单份成本": 1.0,
            "单注赔付": float(PAYOUT_RATIO["三定"]),
            "命中概率": 1 / 1000.0,
        },
        "四定单码": {
            "组合数": 1,
            "单份成本": 1.0,
            "单注赔付": float(PAYOUT_RATIO["四定"]),
            "命中概率": 1 / 10000.0,
        },
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


def calculate_custom_budget_plans(budget: float, rec: dict, enabled: set):
    """自定义模式：等分预算，但确保每个已勾选的玩至少得到一个正额度，
    并在剩余预算中再次分配以避免遗漏。
    """
    if budget <= 0 or not enabled:
        return {"__total__": 0.0, "__risk__": "自定义"}

    # ---------- 1️⃣ 计算每个定位包码的组合数 ----------
    bao_combos = {}
    for name in ["二定", "三定", "四定"]:
        n = 1
        for _, ds in rec[name]["包码"]:
            n *= len(ds)
        bao_combos[name] = n if rec[name]["包码"] else 0

    # ---------- 2️⃣ 构造每种“投注单元”的基础信息 ----------
    schemes = {
        "二定单码": {"组合数": 1, "单份成本": 1.0,
                    "单注赔付": float(PAYOUT_RATIO["二定"]),
                    "命中概率": 1 / 100.0},
        "三定单码": {"组合数": 1, "单份成本": 1.0,
                    "单注赔付": float(PAYOUT_RATIO["三定"]),
                    "命中概率": 1 / 1000.0},
        "四定单码": {"组合数": 1, "单份成本": 1.0,
                    "单注赔付": float(PAYOUT_RATIO["四定"]),
                    "命中概率": 1 / 10000.0},
        "二定包码": {"组合数": bao_combos["二定"],
                    "单份成本": round(bao_combos["二定"] * 0.1, 2) if bao_combos["二定"] else 0,
                    "单注赔付": 0.1 * PAYOUT_RATIO["二定"],
                    "命中概率": bao_combos["二定"] / 100.0 if bao_combos["二定"] else 0},
        "三定包码": {"组合数": bao_combos["三定"],
                    "单份成本": round(bao_combos["三定"] * 0.1, 2) if bao_combos["三定"] else 0,
                    "单注赔付": 0.1 * PAYOUT_RATIO["三定"],
                    "命中概率": bao_combos["三定"] / 1000.0 if bao_combos["三定"] else 0},
        "四定包码": {"组合数": bao_combos["四定"],
                    "单份成本": round(bao_combos["四定"] * 0.1, 2) if bao_combos["四定"] else 0,
                    "单注赔付": 0.1 * PAYOUT_RATIO["四定"],
                    "命中概率": bao_combos["四定"] / 10000.0 if bao_combos["四定"] else 0},
        "二现": {"组合数": 1, "单份成本": 1.0,
                "单注赔付": float(PAYOUT_RATIO["二现"]),
                "命中概率": PROB_XIAN[2]},
        "三现": {"组合数": 1, "单份成本": 1.0,
                "单注赔付": float(PAYOUT_RATIO["三现"]),
                "命中概率": PROB_XIAN[3]},
        "四现": {"组合数": 1, "单份成本": 1.0,
                "单注赔付": float(PAYOUT_RATIO["四现"]),
                "命中概率": PROB_XIAN[4]},
    }

    valid = [p for p in enabled if p in schemes and schemes[p]["单份成本"] > 0]
    if not valid:
        return {"__total__": 0.0, "__risk__": "自定义"}

    # ---------- 4️⃣ 初始等分配 ----------
    each = budget / len(valid)                # 每个有效玩法的理论份额
    plans = {}
    total_spent = 0.0

    # 首轮：对每个玩法尝试分配 “multiples” >= 1
    for play in valid:
        s = schemes[play]
        multiples = int(each / s["单份成本"])
        if multiples >= 1:                       # 能买到完整的几倍
            cost = round(multiples * s["单份成本"], 2)
        else:                                    # 不到 1 倍，仍要保证最小参与
            # 这里 **不直接跳过**，而是使用单份成本的最小费用
            cost = s["单份成本"]
        plans[play] = {
            "倍数": multiples,
            "组合数": s["组合数"],
            "单份成本": s["单份成本"],
            "实际投入": cost,
            "命中概率": s["命中概率"],
            "单注赔付": s["单注赔付"],
            "中奖金额": round(multiples * s["单注赔付"], 2) if multiples > 0 else 0,
            "净收益": round((multiples * s["单注赔付"]) - cost, 2) if multiples > 0 else 0,
        }
        total_spent += cost

    # ---------- 5️⃣ 剩余预算重新分配给仍未出现的已勾选玩法 ----------
    # （有可能因为上一步的 cost 为单份成本，导致还有剩余预算没有被使用）
    remaining = budget - total_spent
    if remaining > 0:
        # 重新遍历一次 valid，把剩余金额分配给那些仍未加入 plans 的玩法
        # （即上一步因为 `multiples == 0` 被跳过的玩法）
        for play in valid:
            if play not in plans:                     # 仍未得到预算
                s = schemes[play]
                # 给它分配最多剩余金额，但也不能低于它的单份成本（否则还是 0）
                allocate = min(s["单份成本"], remaining)
                if allocate > 0:
                    plans[play] = {
                        "倍数": 0,                         # 因为只分配了一次，倍数设为 0 代表 “1 注”
                        "组合数": s["组合数"],
                        "单份成本": s["单份成本"],
                        "实际投入": allocate,
                        "命中概率": s["命中概率"],
                        "单注赔付": s["单注赔付"],
                        "中奖金额": 0,                       # 只买 1 注，理论上中奖金额为 0
                        "净收益": -allocate,
                    }
                    total_spent += allocate
                    remaining -= allocate
                    if remaining <= 0:
                        break

    # ---------- 6️⃣ 最终收尾 ----------
    plans["__total__"] = round(total_spent, 2)
    plans["__risk__"] = "自定义"
    return plans


# ============================================================
# 回测引擎
# ============================================================
PLAY_TO_DEF_NAME = {
    "二定单码": ("二定", "单码"),
    "三定单码": ("三定", "单码"),
    "四定单码": ("四定", "单码"),
    "二定包码": ("二定", "包码"),
    "三定包码": ("三定", "包码"),
    "四定包码": ("四定", "包码"),
}
POS_NAME_TO_IDX = {"千位": 0, "百位": 1, "十位": 2, "个位": 3}


def make_random_recommendations():
    """生成随机基准的推荐方案，结构与 make_recommendations 输出一致。"""
    pos_names = Predictor.POS_NAMES
    rec = {}
    pos_strength = list(range(4))
    random.shuffle(pos_strength)

    def rand_digits(k):
        return random.sample(range(10), k)

    two_pos = sorted(pos_strength[:2])
    rec["二定"] = {
        "单码": [(pos_names[p], random.randint(0, 9)) for p in two_pos],
        "包码": [(pos_names[p], rand_digits(3)) for p in two_pos],
    }
    three_pos = sorted(pos_strength[:3])
    rec["三定"] = {
        "单码": [(pos_names[p], random.randint(0, 9)) for p in three_pos],
        "包码": [(pos_names[p], rand_digits(3)) for p in three_pos],
    }
    rec["四定"] = {
        "单码": [(pos_names[p], random.randint(0, 9)) for p in range(4)],
        "包码": [(pos_names[p], rand_digits(2)) for p in range(4)],
    }
    rec["二现"] = rand_digits(2)
    rec["三现"] = rand_digits(3)
    rec["四现"] = rand_digits(4)
    return rec


def evaluate_bet(play, rec, plan, actual):
    """判断一注是否命中，返回 (hit, cost, payout)。
    actual: 4位开奖号 list[int]"""
    cost = plan["实际投入"]
    multiples = plan["倍数"]

    if play in PLAY_TO_DEF_NAME:
        def_name, kind = PLAY_TO_DEF_NAME[play]
        if kind == "单码":
            single = rec[def_name]["单码"]
            hit = all(actual[POS_NAME_TO_IDX[pos]] == d for pos, d in single)
            payout = round(multiples * PAYOUT_RATIO[def_name], 2) if hit else 0.0
        else:
            bao = rec[def_name]["包码"]
            hit = all(actual[POS_NAME_TO_IDX[pos]] in digits for pos, digits in bao)
            payout = round(0.1 * PAYOUT_RATIO[def_name] * multiples, 2) if hit else 0.0
    else:
        digits = rec[play]
        hit = all(d in actual for d in digits)
        payout = round(multiples * PAYOUT_RATIO[play], 2) if hit else 0.0

    return hit, cost, payout


def run_backtest(history, train_window=50, budget=100.0, risk="平衡"):
    """滚动回测：history 为 newest-first，对每一期 t (从最旧的可测期到最新)
    用其前 train_window 期作训练数据预测，对比算法 vs 随机基准。

    返回包含 totals, play_stats, details 的 dict。"""
    chrono = list(reversed(history))  # 从旧到新
    n = len(chrono)
    if n < train_window + 1:
        raise ValueError(f"数据不足，需要至少 {train_window + 1} 期，当前 {n} 期")

    play_stats = {}
    play_names = ["二定单码", "二定包码", "三定单码", "三定包码",
                  "四定单码", "四定包码", "二现", "三现", "四现"]
    for p in play_names:
        play_stats[p] = {
            "algo_bets": 0, "algo_hits": 0, "algo_cost": 0.0, "algo_payout": 0.0,
            "random_bets": 0, "random_hits": 0, "random_cost": 0.0, "random_payout": 0.0,
        }

    details = []
    algo_total_cost = algo_total_payout = 0.0
    rand_total_cost = rand_total_payout = 0.0

    for t in range(train_window, n):
        train_newest_first = list(reversed(chrono[:t]))
        target = chrono[t]
        actual = target["nums"][:4]

        predictor = Predictor(train_newest_first)
        algo_rec, _, _ = make_recommendations(predictor)
        algo_plans = calculate_budget_plans(budget, algo_rec, risk)

        random_rec = make_random_recommendations()
        random_plans = calculate_budget_plans(budget, random_rec, risk)

        algo_eval = {"results": {}, "total_cost": 0.0, "total_payout": 0.0}
        random_eval = {"results": {}, "total_cost": 0.0, "total_payout": 0.0}

        for play in play_names:
            if play in algo_plans and not play.startswith("__"):
                hit, cost, payout = evaluate_bet(play, algo_rec, algo_plans[play], actual)
                algo_eval["results"][play] = {"hit": hit, "cost": cost, "payout": payout}
                algo_eval["total_cost"] += cost
                algo_eval["total_payout"] += payout
                play_stats[play]["algo_bets"] += 1
                play_stats[play]["algo_hits"] += int(hit)
                play_stats[play]["algo_cost"] += cost
                play_stats[play]["algo_payout"] += payout

            if play in random_plans and not play.startswith("__"):
                hit, cost, payout = evaluate_bet(play, random_rec, random_plans[play], actual)
                random_eval["results"][play] = {"hit": hit, "cost": cost, "payout": payout}
                random_eval["total_cost"] += cost
                random_eval["total_payout"] += payout
                play_stats[play]["random_bets"] += 1
                play_stats[play]["random_hits"] += int(hit)
                play_stats[play]["random_cost"] += cost
                play_stats[play]["random_payout"] += payout

        algo_total_cost += algo_eval["total_cost"]
        algo_total_payout += algo_eval["total_payout"]
        rand_total_cost += random_eval["total_cost"]
        rand_total_payout += random_eval["total_payout"]

        details.append({
            "issue": target["issue"], "date": target["date"],
            "actual": "".join(str(x) for x in actual),
            "algo_rec": algo_rec, "random_rec": random_rec,
            "algo_eval": algo_eval, "random_eval": random_eval,
        })

    totals = {
        "algo_cost": round(algo_total_cost, 2),
        "algo_payout": round(algo_total_payout, 2),
        "algo_net": round(algo_total_payout - algo_total_cost, 2),
        "algo_roi": round((algo_total_payout - algo_total_cost) / algo_total_cost, 4) if algo_total_cost > 0 else 0.0,
        "random_cost": round(rand_total_cost, 2),
        "random_payout": round(rand_total_payout, 2),
        "random_net": round(rand_total_payout - rand_total_cost, 2),
        "random_roi": round((rand_total_payout - rand_total_cost) / rand_total_cost, 4) if rand_total_cost > 0 else 0.0,
    }

    for p, s in play_stats.items():
        s["algo_hit_rate"] = round(s["algo_hits"] / s["algo_bets"], 4) if s["algo_bets"] > 0 else 0.0
        s["random_hit_rate"] = round(s["random_hits"] / s["random_bets"], 4) if s["random_bets"] > 0 else 0.0
        s["algo_roi"] = round((s["algo_payout"] - s["algo_cost"]) / s["algo_cost"], 4) if s["algo_cost"] > 0 else 0.0
        s["random_roi"] = round((s["random_payout"] - s["random_cost"]) / s["random_cost"], 4) if s["random_cost"] > 0 else 0.0

    return {
        "train_window": train_window,
        "n_test": len(details),
        "budget": budget,
        "risk": risk,
        "totals": totals,
        "play_stats": play_stats,
        "details": details,
    }


# ============================================================
# GUI
# ============================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("排列五预测器")
        self.geometry("840x660")
        self.configure(bg="#f5f5f5")

        try:
            # 尝试加载图标
            import os
            icon_path = os.path.join(os.path.dirname(__file__), "app.ico")
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
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

        self.btn_quant = tk.Button(
            top, text="量化", font=("微软雅黑", 11, "bold"),
            width=10, height=1, bg="#27ae60", fg="white",
            activebackground="#1e8449", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            command=self.on_load_quant)
        self.btn_quant.pack(side=tk.RIGHT)

        self.btn_backtest = tk.Button(
            top, text="回测", font=("微软雅黑", 11, "bold"),
            width=10, height=1, bg="#2980b9", fg="white",
            activebackground="#1f618d", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            command=self.on_backtest)
        self.btn_backtest.pack(side=tk.RIGHT, padx=(0, 6))

        self.btn_predict = tk.Button(
            top, text="预测", font=("微软雅黑", 11, "bold"),
            width=10, height=1, bg="#e74c3c", fg="white",
            activebackground="#c0392b", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            command=self.on_predict)
        self.btn_predict.pack(side=tk.RIGHT, padx=(0, 6))

        self.dropdown_slot = tk.Frame(top, bg="#f5f5f5", width=130, height=28)
        self.dropdown_slot.pack(side=tk.RIGHT, padx=(12, 6))
        self.dropdown_slot.pack_propagate(False)

        self.risk_frame = tk.Frame(self.dropdown_slot, bg="#f5f5f5")
        tk.Label(self.risk_frame, text="风险偏好：", font=("微软雅黑", 10),
                 bg="#f5f5f5", fg="#555").pack(side=tk.LEFT)
        self.risk_var = tk.StringVar(value="平衡")
        self.combo_risk = ttk.Combobox(
            self.risk_frame, textvariable=self.risk_var,
            values=list(RISK_PROFILES.keys()),
            state="readonly", width=6,
            font=("微软雅黑", 10))
        self.combo_risk.pack(side=tk.LEFT, padx=(4, 0))

        self.quant_frame = tk.Frame(self.dropdown_slot, bg="#f5f5f5")
        tk.Label(self.quant_frame, text="量化数据：", font=("微软雅黑", 10),
                 bg="#f5f5f5", fg="#555").pack(side=tk.LEFT)
        self.quant_periods_var = tk.StringVar(value="2000")
        self.combo_quant_periods = ttk.Combobox(
            self.quant_frame, textvariable=self.quant_periods_var,
            values=["500", "1000", "2000", "5000", "全部"],
            state="readonly", width=6,
            font=("微软雅黑", 10))
        self.combo_quant_periods.pack(side=tk.LEFT, padx=(4, 0))

        self._show_risk_dropdown()

        for w in (self.btn_predict, self.btn_backtest):
            w.bind("<Enter>", lambda e: self._show_risk_dropdown())
        self.btn_quant.bind("<Enter>", lambda e: self._show_quant_dropdown())

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

        self._build_custom_panel()

        # 主显示区
        body = tk.Frame(self, bg="#f5f5f5")
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        self.text = scrolledtext.ScrolledText(
            body, font=("Consolas", 10), wrap=tk.WORD,
            bg="white", relief=tk.SOLID, bd=1)
        self.text.pack(fill=tk.BOTH, expand=True)
        self.text.configure(state=tk.DISABLED)

        # 底部"凌枯网络制作"文字
        bottom_frame = tk.Frame(self, bg="#f5f5f5")
        bottom_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        tk.Label(bottom_frame, text="凌枯网络制作", 
                 font=("微软雅黑", 9), 
                 bg="#f5f5f5", 
                 fg="#999").pack(side=tk.RIGHT)

        self._tag_setup()

    def _tag_setup(self):
        self.text.tag_configure("h1", font=("微软雅黑", 13, "bold"), foreground="#c0392b", spacing3=6)
        self.text.tag_configure("h2", font=("微软雅黑", 11, "bold"), foreground="#2c3e50", spacing3=4)
        self.text.tag_configure("hint", font=("微软雅黑", 9), foreground="#888")
        self.text.tag_configure("num", font=("Consolas", 11, "bold"), foreground="#e74c3c")
        self.text.tag_configure("digit", font=("Consolas", 14, "bold"), foreground="#e74c3c")
        self.text.tag_configure("ok", foreground="#27ae60")
        self.text.tag_configure("dim", foreground="#999")

    def _build_custom_panel(self):
        """状态栏下方的自定义策略入口栏。"""
        bar = tk.Frame(self, bg="#fff8e7", height=32)
        bar.pack(fill=tk.X)

        self.custom_enabled_var = tk.BooleanVar(value=False)
        chk = tk.Checkbutton(
            bar, text="启用自定义策略", variable=self.custom_enabled_var,
            font=("微软雅黑", 9), bg="#fff8e7", fg="#7d5a00",
            activebackground="#fff8e7", selectcolor="white",
            command=self._save_custom_config)
        chk.pack(side=tk.LEFT, padx=(14, 6), pady=4)

        btn_cfg = tk.Button(
            bar, text="配置...", font=("微软雅黑", 9),
            bg="#f39c12", fg="white",
            activebackground="#d68910", activeforeground="white",
            relief=tk.FLAT, cursor="hand2", padx=10,
            command=self.open_custom_dialog)
        btn_cfg.pack(side=tk.LEFT, pady=4)

        self.custom_summary = tk.Label(
            bar, text="（启用后，预测将按你的设置生成方案）",
            font=("微软雅黑", 9), bg="#fff8e7", fg="#a89968", anchor="w")
        self.custom_summary.pack(side=tk.LEFT, padx=10, pady=4)

        self.custom_config = self._load_custom_config()
        active = self.custom_config.pop("__active__", False)
        self.custom_enabled_var.set(active)
        self._refresh_custom_summary()

    @staticmethod
    def _custom_config_path():
        return os.path.join(os.path.dirname(__file__), "custom_config.json")

    def _load_custom_config(self):
        default = {
            "enabled": {"二定包码", "三定包码", "二现", "三现"},
            "bao_pos": {
                "二定": [0, 3, 0, 3],
                "三定": [3, 3, 3, 0],
                "四定": [2, 2, 2, 2],
            },
            "xian_manual": {},
            "__active__": False,
        }
        path = self._custom_config_path()
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "enabled": set(data.get("enabled", [])),
                "bao_pos": data.get("bao_pos", default["bao_pos"]),
                "xian_manual": data.get("xian_manual", {}),
                "__active__": bool(data.get("__active__", False)),
            }
        except Exception:
            return default

    def _save_custom_config(self):
        try:
            cfg = self.custom_config
            data = {
                "__active__": bool(self.custom_enabled_var.get()),
                "enabled": sorted(cfg.get("enabled", set())),
                "bao_pos": cfg.get("bao_pos", {}),
                "xian_manual": cfg.get("xian_manual", {}),
            }
            with open(self._custom_config_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[warn] save custom config failed: {e}")

    def _refresh_custom_summary(self):
        cfg = self.custom_config
        en = cfg.get("enabled", set())
        if not en:
            self.custom_summary.config(text="（未勾选任何玩法）")
            return
        parts = []
        for name in ["二定单码", "二定包码", "三定单码", "三定包码",
                     "四定单码", "四定包码", "二现", "三现", "四现"]:
            if name in en:
                parts.append(name)
        self.custom_summary.config(
            text=f"已选: {' / '.join(parts)}（共 {len(parts)} 项）")

    def open_custom_dialog(self):
        """弹出自定义策略配置对话框。"""
        dlg = tk.Toplevel(self)
        dlg.title("自定义策略配置")
        dlg.geometry("580x640")
        dlg.minsize(560, 600)
        dlg.configure(bg="#f5f5f5")
        dlg.transient(self)
        dlg.grab_set()

        cfg = self.custom_config

        play_vars = {}
        play_frame = tk.LabelFrame(
            dlg, text=" 玩法选择（勾选要参与的玩法） ",
            font=("微软雅黑", 10, "bold"),
            bg="#f5f5f5", fg="#2c3e50", padx=10, pady=8)
        play_frame.pack(fill=tk.X, padx=14, pady=(12, 6))

        plays_layout = [
            ["二定单码", "二定包码"],
            ["三定单码", "三定包码"],
            ["四定单码", "四定包码"],
            ["二现", "三现", "四现"],
        ]
        for r, row in enumerate(plays_layout):
            for c, name in enumerate(row):
                v = tk.BooleanVar(value=name in cfg.get("enabled", set()))
                play_vars[name] = v
                tk.Checkbutton(
                    play_frame, text=name, variable=v,
                    font=("微软雅黑", 10), bg="#f5f5f5", anchor="w", width=12
                ).grid(row=r, column=c, sticky="w", padx=4, pady=2)

        bao_frame = tk.LabelFrame(
            dlg, text=" 定位包码：每位取多少个数（0=不选这位，1-9=取算法 top-N） ",
            font=("微软雅黑", 10, "bold"),
            bg="#f5f5f5", fg="#2c3e50", padx=10, pady=8)
        bao_frame.pack(fill=tk.X, padx=14, pady=6)

        pos_labels = ["千位", "百位", "十位", "个位"]
        bao_vars = {}
        tk.Label(bao_frame, text="", bg="#f5f5f5", width=8).grid(row=0, column=0)
        for c, lab in enumerate(pos_labels):
            tk.Label(bao_frame, text=lab, font=("微软雅黑", 9, "bold"),
                     bg="#f5f5f5", fg="#555").grid(row=0, column=c + 1, padx=8)

        for r, def_name in enumerate(["二定", "三定", "四定"]):
            tk.Label(bao_frame, text=f"{def_name}包码",
                     font=("微软雅黑", 10), bg="#f5f5f5",
                     fg="#2c3e50").grid(row=r + 1, column=0, padx=4, pady=4, sticky="w")
            counts = cfg.get("bao_pos", {}).get(def_name, [0, 0, 0, 0])
            row_vars = []
            for c in range(4):
                sv = tk.StringVar(value=str(counts[c] if c < len(counts) else 0))
                ttk.Combobox(
                    bao_frame, textvariable=sv,
                    values=[str(i) for i in range(10)],
                    state="readonly", width=4,
                    font=("Consolas", 10)
                ).grid(row=r + 1, column=c + 1, padx=8, pady=4)
                row_vars.append(sv)
            bao_vars[def_name] = row_vars

        xian_frame = tk.LabelFrame(
            dlg, text=" 现玩法手动指定数字（留空则用算法 top-N） ",
            font=("微软雅黑", 10, "bold"),
            bg="#f5f5f5", fg="#2c3e50", padx=10, pady=8)
        xian_frame.pack(fill=tk.X, padx=14, pady=6)

        xian_vars = {}
        for r, (name, n) in enumerate([("二现", 2), ("三现", 3), ("四现", 4)]):
            tk.Label(xian_frame, text=f"{name}（{n}个数字，用空格分隔）",
                     font=("微软雅黑", 9), bg="#f5f5f5",
                     fg="#555").grid(row=r, column=0, sticky="w", padx=4, pady=3)
            cur = cfg.get("xian_manual", {}).get(name, [])
            sv = tk.StringVar(value=" ".join(str(x) for x in cur))
            tk.Entry(xian_frame, textvariable=sv,
                     font=("Consolas", 10), width=20).grid(
                row=r, column=1, padx=8, pady=3, sticky="w")
            xian_vars[name] = (sv, n)

        btn_bar = tk.Frame(dlg, bg="#f5f5f5")
        btn_bar.pack(fill=tk.X, padx=14, pady=10)

        def on_save():
            new_enabled = {n for n, v in play_vars.items() if v.get()}
            new_bao = {}
            for def_name, vars_ in bao_vars.items():
                vals = []
                for sv in vars_:
                    try:
                        vals.append(max(0, min(9, int(sv.get()))))
                    except ValueError:
                        vals.append(0)
                new_bao[def_name] = vals
            new_xian = {}
            for name, (sv, n) in xian_vars.items():
                raw = sv.get().strip()
                if not raw:
                    continue
                parts = raw.replace(",", " ").split()
                try:
                    digits = [int(x) for x in parts]
                except ValueError:
                    messagebox.showerror("格式错误",
                                          f"{name} 的数字格式不正确：{raw}")
                    return
                if len(digits) != n or not all(0 <= d <= 9 for d in digits):
                    messagebox.showerror("格式错误",
                                          f"{name} 需要 {n} 个 0-9 的数字")
                    return
                if len(set(digits)) != n:
                    messagebox.showerror("格式错误",
                                          f"{name} 的 {n} 个数字不能重复")
                    return
                new_xian[name] = digits

            self.custom_config = {
                "enabled": new_enabled,
                "bao_pos": new_bao,
                "xian_manual": new_xian,
            }
            self.custom_enabled_var.set(True)
            self._save_custom_config()
            self._refresh_custom_summary()
            dlg.destroy()

        tk.Button(btn_bar, text="确定",
                  font=("微软雅黑", 10, "bold"),
                  bg="#27ae60", fg="white",
                  activebackground="#1e8449", activeforeground="white",
                  relief=tk.FLAT, cursor="hand2", padx=20, pady=4,
                  command=on_save).pack(side=tk.RIGHT, padx=(6, 0))
        tk.Button(btn_bar, text="取消",
                  font=("微软雅黑", 10),
                  bg="#bdc3c7", fg="#333",
                  activebackground="#95a5a6", activeforeground="#333",
                  relief=tk.FLAT, cursor="hand2", padx=14, pady=4,
                  command=dlg.destroy).pack(side=tk.RIGHT)

    def _show_risk_dropdown(self):
        self.quant_frame.pack_forget()
        self.risk_frame.pack(fill=tk.BOTH, expand=True)

    def _show_quant_dropdown(self):
        self.risk_frame.pack_forget()
        self.quant_frame.pack(fill=tk.BOTH, expand=True)

    def set_status(self, msg, color="#555"):
        self.status.config(text=msg, fg=color)

    def append(self, text, tag=None):
        self.text.configure(state=tk.NORMAL)
        if tag:
            self.text.insert(tk.END, text, tag)
        else:
            self.text.insert(tk.END, text)
        self.text.configure(state=tk.DISABLED)

    def _scroll_top(self):
        self.text.yview_moveto(0)

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
            try:
                db_save_draws(history)
            except Exception as db_err:
                print(f"[DB warn] save draws failed: {db_err}")
            self.after(0, lambda: self.set_status(f"已获取 {len(history)} 期数据，正在分析...", "#2980b9"))

            predictor = Predictor(history)
            use_custom = self.custom_enabled_var.get()
            print(f"[预测] use_custom={use_custom} config.enabled={self.custom_config.get('enabled')}")
            if use_custom:
                rec, pos_scores, digit_scores, enabled = make_custom_recommendations(
                    predictor, self.custom_config)
                budget_plans = calculate_custom_budget_plans(self.budget, rec, enabled)
                print(f"[预测] 自定义路径 plans={[k for k in budget_plans if not k.startswith('__')]}")
            else:
                rec, pos_scores, digit_scores = make_recommendations(predictor)
                budget_plans = calculate_budget_plans(self.budget, rec, self.risk)
                print(f"[预测] 标准路径 risk={self.risk}")

            try:
                next_issue = self._next_issue(history[0]["issue"])
                db_save_prediction(next_issue, self.budget,
                                   "自定义" if use_custom else self.risk, rec,
                                   budget_plans, pos_scores, digit_scores)
            except Exception as db_err:
                print(f"[DB warn] save prediction failed: {db_err}")

            self.after(0, lambda: self._render(history, rec, pos_scores, digit_scores, budget_plans))
            mode_label = f"自定义模式 ({len([k for k in budget_plans if not k.startswith('__')])} 项)" if use_custom else f"标准模式 ({self.risk})"
            self.after(0, lambda: self.set_status(f"完成 — {mode_label}", "#27ae60"))
        except Exception as e:
            err = str(e)
            self.after(0, lambda: self.set_status(f"失败：{err}", "#c0392b"))
            self.after(0, lambda: messagebox.showerror("出错了", f"爬取或分析失败：\n{err}"))
        finally:
            self.after(0, lambda: self.btn_predict.config(state=tk.NORMAL, text="预测"))

    @staticmethod
    def _next_issue(issue):
        try:
            return str(int(issue) + 1)
        except (ValueError, TypeError):
            return f"after_{issue}"

    def on_backtest(self):
        budget = self._read_budget()
        if budget is None:
            return
        self.budget = budget
        self.risk = self.risk_var.get() or "平衡"
        self.btn_backtest.config(state=tk.DISABLED, text="回测中...")
        self.btn_predict.config(state=tk.DISABLED)
        self.set_status("正在爬取100期数据用于滚动回测（前50期训练，后50期测试）...", "#2980b9")
        self.clear()
        threading.Thread(target=self._do_backtest, daemon=True).start()

    def _do_backtest(self):
        try:
            history = fetch_history(100)
            try:
                db_save_draws(history)
            except Exception as db_err:
                print(f"[DB warn] save draws failed: {db_err}")
            self.after(0, lambda: self.set_status(
                f"已获取 {len(history)} 期，正在滚动回测...", "#2980b9"))

            result = run_backtest(history, train_window=50,
                                  budget=self.budget, risk=self.risk)
            try:
                bt_id = db_save_backtest(result)
                result["__db_id__"] = bt_id
            except Exception as db_err:
                print(f"[DB warn] save backtest failed: {db_err}")
                result["__db_id__"] = None

            self.after(0, lambda: self._render_backtest(result))
            self.after(0, lambda: self.set_status(
                f"回测完成 — {result['n_test']} 期测试", "#27ae60"))
        except Exception as e:
            err = str(e)
            self.after(0, lambda: self.set_status(f"失败：{err}", "#c0392b"))
            self.after(0, lambda: messagebox.showerror("出错了", f"回测失败：\n{err}"))
        finally:
            self.after(0, lambda: self.btn_backtest.config(state=tk.NORMAL, text="回测"))
            self.after(0, lambda: self.btn_predict.config(state=tk.NORMAL))

    def _render_backtest(self, result):
        self.clear()
        t = result["totals"]
        n = result["n_test"]

        self.append(f"【滚动回测报告 — {n} 期测试】\n", "h1")
        self.append(
            f"训练窗口：{result['train_window']} 期    每期预算：¥{result['budget']:.2f}    "
            f"风险偏好：{result['risk']}", "hint")
        if result.get("__db_id__"):
            self.append(f"    DB记录ID：{result['__db_id__']}\n\n", "hint")
        else:
            self.append("\n\n", "hint")

        self.append("──────  总体收益对比  ──────\n", "h2")
        sep = "  "
        cols = [("策略", 8, "left"), ("总投入", 12, "right"),
                ("总回报", 12, "right"), ("净收益", 12, "right"),
                ("ROI", 10, "right")]
        self.append(sep.join(_vpad(n_, w, a) for n_, w, a in cols) + "\n", "hint")
        self.append("─" * _vwidth(sep.join(_vpad(n_, w, a) for n_, w, a in cols)) + "\n", "dim")

        for label, prefix in [("算法选号", "algo"), ("随机选号", "random")]:
            cells = [
                label,
                f"¥{t[prefix+'_cost']:.2f}",
                f"¥{t[prefix+'_payout']:.2f}",
                f"¥{t[prefix+'_net']:+.2f}",
                f"{t[prefix+'_roi']*100:+.2f}%",
            ]
            line = sep.join(_vpad(v, cols[i][1], cols[i][2]) for i, v in enumerate(cells))
            self.append(line + "\n")

        diff_net = t["algo_net"] - t["random_net"]
        diff_roi = t["algo_roi"] - t["random_roi"]
        self.append(
            f"\n算法相对随机：净收益差 ¥{diff_net:+.2f}，ROI 差 {diff_roi*100:+.2f} 个百分点\n\n",
            "ok" if diff_net > 0 else "dim")

        self.append("──────  各玩法命中率与 ROI  ──────\n", "h2")
        cols2 = [("玩法", 8, "left"),
                 ("算法命中", 10, "right"), ("算法ROI", 10, "right"),
                 ("随机命中", 10, "right"), ("随机ROI", 10, "right"),
                 ("理论命中", 10, "right")]
        self.append(sep.join(_vpad(n_, w, a) for n_, w, a in cols2) + "\n", "hint")
        self.append("─" * _vwidth(sep.join(_vpad(n_, w, a) for n_, w, a in cols2)) + "\n", "dim")

        theoretical = {
            "二定单码": 0.01, "三定单码": 0.001, "四定单码": 0.0001,
            "二定包码": 0.09, "三定包码": 0.027, "四定包码": 0.0016,
            "二现": PROB_XIAN[2], "三现": PROB_XIAN[3], "四现": PROB_XIAN[4],
        }
        for play in ["二定单码", "二定包码", "三定单码", "三定包码",
                     "四定单码", "四定包码", "二现", "三现", "四现"]:
            s = result["play_stats"].get(play, {})
            if not s.get("algo_bets") and not s.get("random_bets"):
                continue
            cells = [
                play,
                f"{s['algo_hits']}/{s['algo_bets']}={s['algo_hit_rate']*100:.1f}%",
                f"{s['algo_roi']*100:+.2f}%",
                f"{s['random_hits']}/{s['random_bets']}={s['random_hit_rate']*100:.1f}%",
                f"{s['random_roi']*100:+.2f}%",
                f"{theoretical.get(play, 0)*100:.2f}%",
            ]
            line = sep.join(_vpad(v, cols2[i][1], cols2[i][2]) for i, v in enumerate(cells))
            self.append(line + "\n")

        self.append("\n──────  逐期明细  ──────\n", "h2")
        self.append("期号       开奖    算法投入  算法回报  随机投入  随机回报\n", "hint")
        self.append("─" * 60 + "\n", "dim")
        for d in result["details"]:
            ae = d["algo_eval"]
            re_ = d["random_eval"]
            line = (f"{d['issue']:<10} {d['actual']:<7}"
                    f"¥{ae['total_cost']:>7.2f} ¥{ae['total_payout']:>8.2f}"
                    f"  ¥{re_['total_cost']:>7.2f} ¥{re_['total_payout']:>8.2f}\n")
            tag = "ok" if ae["total_payout"] > ae["total_cost"] else None
            self.append(line, tag)

        self.append(
            "\n* 结论参考：理论上排列五独立同分布，长期算法 vs 随机的命中率应趋同。"
            "样本越大越能说明问题。50期波动较大属正常。\n",
            "hint")
        self._scroll_top()

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
                ("推荐组合",    24, "left"),
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

            order = ["二定单码", "二定包码", "三定单码", "三定包码",
                     "四定单码", "四定包码", "二现", "三现", "四现"]
            # 决定在当前模式下应该显示的玩法列表
            # 如果是自定义模式，则使用用户勾选的 enabled；否则使用完整顺序，但仍可能因预算缺失而过滤
            is_custom = budget_plans.get("__risk__", "平衡") == "自定义"
            # 在自定义模式下显示用户勾选的玩法；在标准模式下只显示预算中已有条目的项
            if is_custom:
                display_plays = [p for p in order if p in self.custom_config.get("enabled", set())]
            else:
                display_plays = order

            for play in display_plays:
                # 自定义模式下即使预算没有条目，也应显示占位行
                if is_custom:
                    if play not in budget_plans:
                        p = {
                            "实际投入": 0.0,
                            "倍数": 0,
                            "组合数": 0,
                            "命中概率": 0,
                            "中奖金额": 0,
                            "净收益": 0,
                        }
                    else:
                        p = budget_plans[play]
                else:
                    # 标准模式下若没有预算条目则直接跳过
                    if play not in budget_plans:
                        continue
                    p = budget_plans[play]

                cells = [
                    play,
                    self._format_play_digits(play, rec),
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

        is_custom = risk == "自定义"

        # ========== 定位玩法 ==========
        any_def_play = any(f"{n}{k}" in budget_plans
                           for n in ["二定", "三定", "四定"]
                           for k in ["单码", "包码"]) if is_custom else True
        if any_def_play:
            self.append("【定位玩法 — 千百十个位置】\n", "h1")
            self.append("规则：选定位置上中对应数字才算中。包码会拆成多注单码。\n\n", "hint")

            for name in ["二定", "三定", "四定"]:
                data = rec[name]
                show_single = (not is_custom) or (f"{name}单码" in budget_plans)
                show_bao = (not is_custom) or (f"{name}包码" in budget_plans)
                if not (show_single or show_bao) or not data["单码"]:
                    continue
                self.append(f"▶ {name}\n", "h2")

                if show_single:
                    single_str = " | ".join(f"{p}={d}" for p, d in data["单码"])
                    self.append(f"  推荐单码：{single_str}\n")
                    self.append(f"  投注示例：", "hint")
                    self.append(self._format_single_bet(data["单码"]) + "\n", "num")

                if show_bao and data["包码"]:
                    bao_parts = []
                    total_combos = 1
                    for p, ds in data["包码"]:
                        bao_parts.append(f"{p}∈{{{','.join(str(x) for x in ds)}}}")
                        total_combos *= len(ds)
                    self.append(f"  推荐包码：{' , '.join(bao_parts)}  共 {total_combos} 注\n")
                self.append("\n")

        # ========== 现玩法 ==========
        any_xian_play = any(n in budget_plans for n in ["二现", "三现", "四现"]) if is_custom else True
        if any_xian_play:
            self.append("【现玩法 — 不论位置只看数字出现】\n", "h1")
            self.append("规则：选中的所有数字都要在开奖号码中出现才算中。\n\n", "hint")

            for name, count in [("二现", 2), ("三现", 3), ("四现", 4)]:
                if is_custom and name not in budget_plans:
                    continue
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
        self._scroll_top()

    @staticmethod
    def _format_single_bet(items):
        """将 [(位置名, 数字), ...] 格式化为 千百十个 形式（缺失位置打 *）"""
        order = ["千位", "百位", "十位", "个位"]
        m = {pos: d for pos, d in items}
        return " ".join(str(m[p]) if p in m else "*" for p in order)

    @staticmethod
    def _format_play_digits(play, rec):
        """把推荐组合压成一行，给预算表的"推荐组合"列用。"""
        order = ["千位", "百位", "十位", "个位"]
        if play in ("二现", "三现", "四现"):
            return "".join(str(d) for d in rec[play])
        def_name = play[:2]
        kind = "单码" if "单码" in play else "包码"
        items = rec[def_name][kind]
        if not items:
            return "-"
        m = {pos: ds for pos, ds in items}
        if kind == "单码":
            return " ".join(str(m[p]) if p in m else "*" for p in order)
        parts = []
        for p in order:
            if p in m:
                parts.append("".join(str(d) for d in m[p]))
            else:
                parts.append("-")
        return " ".join(parts)

    def on_load_quant(self):
        """根据下拉框选择的数据量重跑量化模型并展示报告。"""
        sel = self.quant_periods_var.get().strip()
        if sel in ("全部", "all", "0"):
            periods = 0
            label = "全部"
        else:
            try:
                periods = int(sel)
                label = f"{periods}期"
            except ValueError:
                messagebox.showwarning("参数错误", f"无法识别数据量：{sel}")
                return

        self.btn_quant.config(state=tk.DISABLED, text="计算中...")
        self.set_status(f"量化计算中（使用 {label} 数据，可能需要 1-3 分钟）...", "#2980b9")
        self.clear()
        self.append(f"【量化计算进行中】使用 {label} 数据\n", "h1")
        self.append("正在加载历史数据 → 学习因子权重 → 滚动回测...\n请稍候\n", "hint")
        threading.Thread(target=self._do_quant, args=(periods,), daemon=True).start()

    def _do_quant(self, periods):
        json_path = os.path.join(os.path.dirname(__file__), "quant_output.json")
        try:
            import quant
            quant.run_quant(periods=periods, budget=100.0,
                            output_file=json_path, verbose=False)
            with open(json_path, "r", encoding="utf-8") as f:
                self.quant_data = json.load(f)
            self.after(0, self._render_quant)
            n = self.quant_data["meta"].get("periods_used", "?")
            self.after(0, lambda: self.set_status(
                f"量化完成 — 使用 {n} 期数据", "#27ae60"))
        except Exception as e:
            err = str(e)
            self.after(0, lambda: self.set_status(f"量化失败：{err}", "#c0392b"))
            self.after(0, lambda: messagebox.showerror("量化失败", err))
        finally:
            self.after(0, lambda: self.btn_quant.config(state=tk.NORMAL, text="量化"))

    def _render_quant(self):
        self.clear()
        d = self.quant_data
        meta = d.get("meta", {})
        bt = d.get("回测报告", {})
        fw = d.get("因子权重", {})
        rec = d.get("预测推荐", {})

        sep = "  "

        # ── 头部 ──
        self.append("【量化分析报告 — 多因子选号模型】\n", "h1")
        self.append(
            f"训练窗口：{meta.get('train_window', '?')} 期    "
            f"测试期数：{meta.get('test_periods', '?')} 期    "
            f"每期预算：¥{meta.get('budget', '?')}    "
            f"生成日期：{meta.get('generated_at', '?')}\n\n",
            "hint")

        # ── 回测概览 ──
        self.append("──────  回测表现  ──────\n", "h2")
        metrics = [
            ("测试期数",  bt.get("n_test", 0), ""),
            ("总投入",    f"¥{bt.get('total_cost', 0):.2f}", ""),
            ("总回报",    f"¥{bt.get('total_payout', 0):.2f}", ""),
            ("净收益",    f"¥{bt.get('net', 0):+.2f}", "ok" if bt.get("net", 0) > 0 else "dim"),
            ("ROI",      f"{bt.get('roi', 0)*100:+.2f}%", ""),
            ("Sharpe",   f"{bt.get('sharpe', 0):.4f}", ""),
            ("Calmar",   f"{bt.get('calmar', 0):.4f}", ""),
            ("最大回撤",  f"¥{bt.get('max_drawdown', 0):.2f}", ""),
            ("盈亏比",    f"{bt.get('profit_factor', 0):.4f}", ""),
            ("胜率",     f"{bt.get('win_rate', 0)*100:.2f}%", ""),
        ]
        cols3 = [("指标", 8, "left"), ("数值", 16, "right")]
        self.append(sep.join(_vpad(n, w, a) for n, w, a in cols3) + "\n", "hint")
        self.append("─" * _vwidth(sep.join(_vpad(n, w, a) for n, w, a in cols3)) + "\n", "dim")
        for name, val, tag in metrics:
            line = sep.join([_vpad(name, 8, "left"), _vpad(str(val), 16, "right")])
            self.append(line + "\n", tag if tag else None)
        diff = bt.get("algo_vs_random_net_diff", 0)
        self.append(
            f"\n算法相对随机基准净收益差：¥{diff:+.2f}\n",
            "ok" if diff > 0 else "dim")

        # ── 因子权重 ──
        self.append("\n──────  因子权重（IC学习）  ──────\n", "h2")
        cols4 = [("因子名", 16, "left"), ("IC", 8, "right"), ("权重", 8, "right"), ("强度", 15, "left")]
        self.append(sep.join(_vpad(n, w, a) for n, w, a in cols4) + "\n", "hint")
        self.append("─" * _vwidth(sep.join(_vpad(n, w, a) for n, w, a in cols4)) + "\n", "dim")
        for name in sorted(fw, key=lambda n: fw[n]["weight"], reverse=True):
            info = fw[name]
            bar = "█" * int(info["weight"] * 50)
            cells = [name, f"{info['ic']:.4f}", f"{info['weight']:.4f}", bar]
            line = sep.join(_vpad(v, cols4[i][1], cols4[i][2]) for i, v in enumerate(cells))
            self.append(line + "\n")

        # ── 量化推荐 ──
        self.append("\n──────  最新推荐组合  ──────\n", "h2")
        pos_names = Predictor.POS_NAMES
        for play_name in ["二定", "三定", "四定"]:
            data = rec.get(play_name, {})
            if not data:
                continue
            single = [(p, d) for p, d in data.get("单码", [])]
            bao = [(p, ds) for p, ds in data.get("包码", [])]
            self.append(f"▶ {play_name}\n", "num")
            self.append(f"  单码：{self._format_single_bet(single)}\n")
            bao_str = " , ".join(f"{p}∈{{{','.join(str(x) for x in ds)}}}" for p, ds in bao)
            n_combo = 1
            for _, ds in bao:
                n_combo *= len(ds)
            self.append(f"  包码：{bao_str}  共 {n_combo} 注 = ¥{n_combo*0.1:.2f}\n\n")

        for play_name in ["二现", "三现", "四现"]:
            digits = rec.get(play_name, [])
            if digits:
                self.append(f"▶ {play_name}：{' '.join(str(d) for d in digits)}\n", "digit")
                self.append(f"  即买入数字 {{{','.join(str(d) for d in digits)}}}\n\n", "hint")

        # ── 位置评分 ──
        self.append("──────  各位置数字评分  ──────\n", "dim")
        pos_scores = d.get("位置评分", [])
        if pos_scores:
            self.append("位置  " + "   ".join(f" {d} " for d in range(10)) + "\n", "hint")
            for pos in range(4):
                row = pos_scores[pos]
                line = f"{pos_names[pos]}  "
                for d_idx in range(10):
                    line += f"{row[d_idx]:.2f} "
                self.append(line + "\n")

        self._scroll_top()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
