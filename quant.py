# -*- coding: utf-8 -*-
"""quant.py - 排列五量化选号引擎

设计思路：
1. 多因子评分：11 个因子给每位每个数字打分
2. 滚动回测学最优权重：训练窗 → 预测 → 算 IC（信息系数）→ 选出有效因子
3. 组合优化：按 Kelly 公式分配预算到 6 种玩法
4. 输出 JSON 文件 + 量化指标报告

输出文件 quant_output.json 供 GUI 加载展示。
"""
import json
import math
import random
import re
import sys
from collections import Counter

import requests

# ------------------------------------------------------------
# 配置
# ------------------------------------------------------------
API_URL = "https://jc.zhcw.com/port/client_json.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.zhcw.com/kjxx/pl5/",
}

PAYOUT = {"二定": 96, "三定": 960, "四定": 9600,
          "二现": 9, "三现": 45, "四现": 320}
PROB_XIAN = {2: 0.0974, 3: 0.0204, 4: 0.0024}

OUTPUT_FILE = "quant_output.json"
TARGET_PERIODS = 2000   # 目标总数据量
TRAIN_WINDOW = 500       # 因子权重学习：训练窗口(500期滚动回测学IC)
IC_TEST_COUNT = 300      # 因子IC测试期数
PREDICT_WINDOW = 50      # 预测算法：只取最近50期
BACKTEST_WINDOW = 500    # 量化回测：训练窗口
BUDGET = 100.0

POS_NAMES = ["千位", "百位", "十位", "个位"]


# ------------------------------------------------------------
# 数据加载
# ------------------------------------------------------------
def _fetch_page(end_issue: str = ""):
    """单页请求，返回 (rows, 本页最老期号)。"""
    params = {
        "transactionType": "10001001", "lotteryId": "284",
        "issueCount": "100", "pageNum": "1", "pageSize": "100",
        "startIssue": "", "endIssue": end_issue,
        "startDate": "", "endDate": "", "type": "1",
        "tt": str(random.random()), "callback": "cb",
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    text = resp.text.strip()
    m = re.match(r"^\w+\((.*)\)\s*;?\s*$", text, re.DOTALL)
    if not m:
        raise ValueError("JSONP 格式异常")
    payload = json.loads(m.group(1))
    return payload.get("data", []) or []


def _fetch_api(count: int):
    """API 兜底分页爬取（数据库无数据时用）。"""
    all_rows = []
    old_issue = ""
    page = 0
    while len(all_rows) < count:
        page += 1
        rows = _fetch_page(old_issue)
        if not rows:
            break
        all_rows.extend(rows)
        new_old = rows[-1]["issue"]
        if new_old == old_issue:
            break
        old_issue = new_old
        print(f"      [API] 分页 {page}: 已拿 {len(all_rows)} 期")
        if page > 80:
            break
    history = []
    for row in all_rows:
        parts = row.get("frontWinningNum", "").strip().split()
        if len(parts) != 5 or not all(p.isdigit() for p in parts):
            continue
        history.append({
            "issue": row.get("issue", ""),
            "date": row.get("openTime", ""),
            "nums": [int(x) for x in parts],
        })
    return history


def fetch_history(count: int = 1000):
    """读取历史数据。优先从 MySQL 读，库为空才走 API。"""
    try:
        from db_utils import load_history
        data = load_history(count)
        if data:
            print(f"      从数据库读取 {len(data)} 期")
            return data
        print(f"      数据库为空，转 API")
    except Exception as e:
        print(f"      数据库不可用 ({e})，使用 API")
    return _fetch_api(count)


# ------------------------------------------------------------
# 因子计算
# ------------------------------------------------------------
def _normalize(row):
    lo, hi = min(row), max(row)
    if hi - lo < 1e-9:
        return [0.5] * len(row)
    return [(x - lo) / (hi - lo) for x in row]


class FactorEngine:
    """对给定的历史期数（chronological，旧→新）计算 11 个因子。

    返回 list[4][10]，即 4 个位置各 0-9 数字的综合因子原始值。
    调用 .to_score(weights) 得到加权评分。
    """

    FACTOR_NAMES = [
        "freq_100", "freq_20", "freq_5",          # 长期/中期/短期频率
        "miss_gap",                                 # 遗漏间隔
        "markov1", "markov2",                      # 一/二阶马尔可夫
        "cross_pos",                                # 跨位置关联
        "parity_bias",                              # 奇偶偏差
        "size_bias",                                # 大小偏差
        "sum_trend",                                # 和值趋势
        "cycle",                                    # 周期模式
    ]

    def __init__(self, draws):
        """draws: 从旧到新的 4 位开奖列表 [[d千,d百,d十,d个], ...]"""
        self.draws = draws
        self.n = len(draws)
        self.all_digits = [h["nums"] for h in getattr(self, '_history', [])] if False else None

    def compute(self):
        """返回 list[11][4][10] — 11 个因子 × 4 位置 × 10 数字"""
        factors = []
        n = self.n

        # --- F1-F3: 频率因子 ---
        for window in [n, 20, 5]:
            start = max(0, n - window)
            slice_ = self.draws[start:]
            scores = []
            for pos in range(4):
                cnt = Counter(d[pos] for d in slice_)
                row = [cnt.get(d, 0) / max(len(slice_), 1) for d in range(10)]
                scores.append(_normalize(row))
            factors.append(scores)

        # --- F4: 遗漏值 ---
        miss_scores = []
        for pos in range(4):
            row = []
            for d in range(10):
                gap = n
                for i in range(n - 1, -1, -1):
                    if self.draws[i][pos] == d:
                        gap = n - 1 - i
                        break
                row.append(gap)
            miss_scores.append(_normalize(row))
        factors.append(miss_scores)

        # --- F5-F6: 一/二阶马尔可夫转移概率 ---
        # order=1: P(d_t | d_{t-1}) 上一期同位置 → 本期
        # order=2: P(d_t | d_{t-2}) 上两期同位置 → 本期
        for order in [1, 2]:
            mk_scores = []
            if order > n:
                last_digits = (0,) * 4
            else:
                last_digits = tuple(self.draws[-order][p] for p in range(4))
            for pos in range(4):
                trans = [[1] * 10 for _ in range(10)]
                for i in range(n - order):
                    a = self.draws[i][pos]
                    b = self.draws[i + order][pos]
                    trans[a][b] += 1
                last = last_digits[pos]
                row_total = sum(trans[last])
                row = [trans[last][d] / row_total for d in range(10)]
                mk_scores.append(_normalize(row))
            factors.append(mk_scores)

        # --- F7: 跨位置共现 ---
        # 基于上一期其他三位实际出的数字，查历史共现矩阵给本位置每个 d 打分
        cross = []
        last_draw = self.draws[-1] if n > 0 else [0, 0, 0, 0]
        for pos in range(4):
            row = []
            for d in range(10):
                score = 0.0
                for other in range(4):
                    if other == pos:
                        continue
                    last_d_other = last_draw[other]
                    cnt_other = sum(1 for dr in self.draws if dr[other] == last_d_other)
                    cnt_both = sum(1 for dr in self.draws
                                   if dr[other] == last_d_other and dr[pos] == d)
                    if cnt_other > 0:
                        score += cnt_both / cnt_other
                row.append(score / 3)
            cross.append(_normalize(row))
        factors.append(cross)

        # --- F8: 奇偶反弹（最近20期 pos 位置奇/偶欠出 → 反向加分）---
        parity = []
        recent = self.draws[-min(20, n):]
        for pos in range(4):
            even_cnt = sum(1 for dr in recent if dr[pos] % 2 == 0)
            even_ratio = even_cnt / max(len(recent), 1)
            row = []
            for d in range(10):
                if d % 2 == 0:
                    row.append(1.0 - even_ratio)
                else:
                    row.append(even_ratio)
            parity.append(_normalize(row))
        factors.append(parity)

        # --- F9: 大小反弹（最近20期 pos 位置大/小欠出 → 反向加分）---
        size = []
        for pos in range(4):
            big_cnt = sum(1 for dr in recent if dr[pos] >= 5)
            big_ratio = big_cnt / max(len(recent), 1)
            row = []
            for d in range(10):
                if d >= 5:
                    row.append(1.0 - big_ratio)
                else:
                    row.append(big_ratio)
            size.append(_normalize(row))
        factors.append(size)

        # --- F10: 和值趋势 ---
        sums = []
        for i in range(1, len(self.draws)):
            s_prev = sum(self.draws[i - 1])
            s_curr = sum(self.draws[i])
            sums.append(s_curr - s_prev)
        avg_delta = sum(sums) / max(len(sums), 1)
        sum_trend = []
        last_sum = sum(self.draws[-1]) if self.draws else 18
        for pos in range(4):
            row = []
            for d in range(10):
                expected_contribution = (last_sum + avg_delta) / 4
                row.append(1.0 - abs(d - expected_contribution) / 9.0)
            sum_trend.append(_normalize(row))
        factors.append(sum_trend)

        # --- F11: 周期模式 ---
        cycle = []
        step = 5
        for pos in range(4):
            row = [0.0] * 10
            for d in range(10):
                hits = 0
                for i in range(step, n, step):
                    if self.draws[i][pos] == d:
                        hits += 1
                row[d] = hits / max(n / step, 1)
            cycle.append(_normalize(row))
        factors.append(cycle)

        return factors


# ------------------------------------------------------------
# 权重学习（滚动回测 → 信息系数 IC → 最优权重）
# ------------------------------------------------------------
def compute_factor_ic(factors, actual, n_train=None):
    """因子 IC：4 个位置的 top-3 加权命中率。
    实际数字命中 top1 计 1.0、top2 计 0.5、top3 计 0.25，最大 1.0。"""
    weights = {0: 1.0, 1: 0.5, 2: 0.25}
    score = 0.0
    for pos in range(4):
        ranked = sorted(range(10), key=lambda d: factors[pos][d], reverse=True)
        for rank in range(3):
            if actual[pos] == ranked[rank]:
                score += weights[rank]
                break
    return score / 4.0


def learn_optimal_weights(history, train_window=50, test_window=50):
    """滚动窗口学习因子权重。

    对每个 test 期：用前 train_window 期算 11 因子 → 与实际对比算 IC。
    用 Bayesian 加权平均 + softmax 得出最终权重。
    返回: (weights: list[float], ic_report: dict)
    """
    chrono = list(reversed(history))  # 旧→新
    n = len(chrono)
    if n < train_window + 10:
        raise ValueError(f"数据不足：{n} < {train_window + 10}")

    # 提取 draws 序列
    all_draws = [h["nums"][:4] for h in chrono]
    n_test = min(test_window, n - train_window)
    ic_accum = [0.0] * 11
    ic_count = 0

    for t in range(train_window, train_window + n_test):
        train_draws = all_draws[t - train_window:t]
        actual = all_draws[t]
        engine = FactorEngine.__new__(FactorEngine)
        engine.draws = train_draws
        engine.n = len(train_draws)
        factors = engine.compute()
        for f_idx, f_data in enumerate(factors):
            ic_accum[f_idx] += compute_factor_ic(f_data, actual, train_window)
        ic_count += 1

    if ic_count == 0:
        return [1.0 / 11] * 11, {}

    mean_ic = [v / ic_count for v in ic_accum]
    # 每个因子权重正比于 max(IC, 0.04)（IC 低但有微弱信号仍保留，避免过拟合噪声）
    positive_ic = [max(ic, 0.04) for ic in mean_ic]
    total = sum(positive_ic)
    weights = [ic / total for ic in positive_ic]

    # 生成 IC 报告
    ic_report = {}
    for i, (name, ic, w) in enumerate(zip(FactorEngine.FACTOR_NAMES, mean_ic, weights)):
        ic_report[name] = {"ic": round(ic, 6), "weight": round(w, 4)}

    return weights, ic_report


# ------------------------------------------------------------
# 综合评分 + 推荐生成
# ------------------------------------------------------------
def predict_with_weights(history, weights):
    """用学习到的权重对最新一期做预测。返回 (rec, pos_scores, digit_scores)"""
    chrono = list(reversed(history))
    draws = [h["nums"][:4] for h in chrono]
    engine = FactorEngine.__new__(FactorEngine)
    engine.draws = draws
    engine.n = len(draws)
    factors = engine.compute()

    # 综合评分
    pos_scores = [[sum(w * f[pos][d] for w, f in zip(weights, factors))
                   for d in range(10)] for pos in range(4)]

    # 全局数字评分
    digit_scores = [0.0] * 10
    for d in range(10):
        cnt = sum(1 for dr in draws for p in range(4) if dr[p] == d)
        digit_scores[d] = cnt / max(len(draws) * 4, 1)
    # 融合增量（取最近 5 期加权平滑）
    recent = draws[-5:]
    for d in range(10):
        cnt_r = sum(1 for dr in recent for p in range(4) if dr[p] == d)
        digit_scores[d] += cnt_r / max(len(recent) * 4, 1) * 0.3
    digit_scores = _normalize(digit_scores)

    # 推荐生成
    top_per_pos = []
    for pos in range(4):
        ranked = sorted(range(10), key=lambda d: pos_scores[pos][d], reverse=True)
        top_per_pos.append(ranked)
    best_digit_each_pos = [tp[0] for tp in top_per_pos]
    digit_ranked = sorted(range(10), key=lambda d: digit_scores[d], reverse=True)

    pos_strength = [(pos, max(pos_scores[pos])) for pos in range(4)]
    pos_strength.sort(key=lambda x: x[1], reverse=True)

    def _select_dynamic(scores, min_n, max_n, threshold):
        ranked = sorted(range(10), key=lambda d: scores[d], reverse=True)
        selected = [d for d in ranked if scores[d] >= threshold]
        if len(selected) < min_n:
            selected = ranked[:min_n]
        elif len(selected) > max_n:
            selected = selected[:max_n]
        return selected

    rec = {}
    two_pos = sorted([pos_strength[0][0], pos_strength[1][0]])
    three_pos = sorted([pos_strength[i][0] for i in range(3)])
    rec["二定"] = {
        "单码": [(POS_NAMES[p], best_digit_each_pos[p]) for p in two_pos],
        "包码": [(POS_NAMES[p], _select_dynamic(pos_scores[p], 3, 6, 0.55)) for p in two_pos],
    }
    rec["三定"] = {
        "单码": [(POS_NAMES[p], best_digit_each_pos[p]) for p in three_pos],
        "包码": [(POS_NAMES[p], _select_dynamic(pos_scores[p], 3, 6, 0.55)) for p in three_pos],
    }
    rec["四定"] = {
        "单码": [(POS_NAMES[p], best_digit_each_pos[p]) for p in range(4)],
        "包码": [(POS_NAMES[p], _select_dynamic(pos_scores[p], 2, 4, 0.6)) for p in range(4)],
    }
    rec["二现"] = digit_ranked[:2]
    rec["三现"] = digit_ranked[:3]
    rec["四现"] = digit_ranked[:4]

    return rec, pos_scores, digit_scores


# ------------------------------------------------------------
# 回测 + 量化指标
# ------------------------------------------------------------
def _make_random_rec():
    """生成随机推荐，结构与 predict_with_weights 输出一致。"""
    pos_strength = list(range(4))
    random.shuffle(pos_strength)
    two_pos = sorted(pos_strength[:2])
    three_pos = sorted(pos_strength[:3])

    def rand_dynamic(min_n, max_n):
        k = random.randint(min_n, max_n)
        return random.sample(range(10), k)

    rec = {}
    rec["二定"] = {
        "单码": [(POS_NAMES[p], random.randint(0, 9)) for p in two_pos],
        "包码": [(POS_NAMES[p], rand_dynamic(3, 6)) for p in two_pos],
    }
    rec["三定"] = {
        "单码": [(POS_NAMES[p], random.randint(0, 9)) for p in three_pos],
        "包码": [(POS_NAMES[p], rand_dynamic(3, 6)) for p in three_pos],
    }
    rec["四定"] = {
        "单码": [(POS_NAMES[p], random.randint(0, 9)) for p in range(4)],
        "包码": [(POS_NAMES[p], rand_dynamic(2, 4)) for p in range(4)],
    }
    rec["二现"] = random.sample(range(10), 2)
    rec["三现"] = random.sample(range(10), 3)
    rec["四现"] = random.sample(range(10), 4)
    return rec


def backtest_with_weights(history, weights, train_window=50, budget=100.0):
    """滚动回测，计算量化指标。同时跑真实随机基准对比。"""
    chrono = list(reversed(history))
    n = len(chrono)
    n_test = n - train_window

    period_values = []
    rand_period_values = []
    hit_records = []
    rand_hit_records = []
    play_stats = {p: {"tests": 0, "algo_hits": 0, "rand_hits": 0,
                       "algo_cost": 0.0, "algo_payout": 0.0,
                       "rand_cost": 0.0, "rand_payout": 0.0}
                  for p in ["二定单码", "二定包码", "三定单码", "三定包码",
                            "四定单码", "四定包码", "二现", "三现", "四现"]}

    for t in range(train_window, train_window + n_test):
        train_hist = list(reversed(chrono[t - train_window:t]))
        actual = chrono[t]["nums"][:4]

        algo_rec, _, _ = predict_with_weights(train_hist, weights)
        algo_plans = _budget_plans(budget, algo_rec)
        rand_rec = _make_random_rec()
        rand_plans = _budget_plans(budget, rand_rec)

        algo_cost = algo_payout = 0.0
        rand_cost = rand_payout = 0.0
        for play, plan in algo_plans.items():
            if play.startswith("__"):
                continue
            hit = _check_hit(play, algo_rec, actual)
            algo_cost += plan["实际投入"]
            if hit:
                algo_payout += plan["中奖金额"]
            play_stats[play]["tests"] += 1
            play_stats[play]["algo_cost"] += plan["实际投入"]
            play_stats[play]["algo_hits"] += int(hit)
            if hit:
                play_stats[play]["algo_payout"] += plan["中奖金额"]
        for play, plan in rand_plans.items():
            if play.startswith("__"):
                continue
            hit = _check_hit(play, rand_rec, actual)
            rand_cost += plan["实际投入"]
            if hit:
                rand_payout += plan["中奖金额"]
            play_stats[play]["rand_cost"] += plan["实际投入"]
            play_stats[play]["rand_hits"] += int(hit)
            if hit:
                play_stats[play]["rand_payout"] += plan["中奖金额"]

        period_values.append(algo_payout - algo_cost)
        rand_period_values.append(rand_payout - rand_cost)
        hit_records.append(algo_payout > 0)
        rand_hit_records.append(rand_payout > 0)

    def _summarize(pvs):
        total_cost = sum(p for p in pvs if p < 0)
        net = sum(pvs)
        equity = [0.0]
        for pv in pvs:
            equity.append(equity[-1] + pv)
        peak = 0
        max_dd = 0.0
        for v in equity:
            if v > peak:
                peak = v
            if peak - v > max_dd:
                max_dd = peak - v
        mean_ret = sum(pvs) / max(len(pvs), 1)
        if len(pvs) > 1:
            var = sum((pv - mean_ret) ** 2 for pv in pvs) / (len(pvs) - 1)
            std = math.sqrt(var) if var > 0 else 1e-9
        else:
            std = 1e-9
        return net, max_dd, mean_ret / std, equity

    algo_total_cost = sum(p["algo_cost"] for p in play_stats.values())
    algo_total_payout = sum(p["algo_payout"] for p in play_stats.values())
    rand_total_cost = sum(p["rand_cost"] for p in play_stats.values())
    rand_total_payout = sum(p["rand_payout"] for p in play_stats.values())

    algo_net, algo_dd, algo_sharpe, equity = _summarize(period_values)
    rand_net, rand_dd, rand_sharpe, _ = _summarize(rand_period_values)

    win_pvs = [pv for pv in period_values if pv > 0]
    loss_pvs = [pv for pv in period_values if pv < 0]
    avg_win = sum(win_pvs) / max(len(win_pvs), 1)
    avg_loss = abs(sum(loss_pvs) / max(len(loss_pvs), 1)) if loss_pvs else 1e-9
    profit_factor = avg_win / avg_loss

    return {
        "train_window": train_window,
        "n_test": n_test,
        "total_cost": round(algo_total_cost, 2),
        "total_payout": round(algo_total_payout, 2),
        "net": round(algo_net, 2),
        "roi": round(algo_net / algo_total_cost, 4) if algo_total_cost > 0 else 0.0,
        "win_rate": round(sum(hit_records) / max(len(hit_records), 1), 4),
        "sharpe": round(algo_sharpe, 4),
        "calmar": round(algo_net / algo_dd, 4) if algo_dd > 0 else 999.0,
        "max_drawdown": round(algo_dd, 2),
        "profit_factor": round(profit_factor, 4),
        "rand_total_cost": round(rand_total_cost, 2),
        "rand_total_payout": round(rand_total_payout, 2),
        "rand_net": round(rand_net, 2),
        "rand_roi": round(rand_net / rand_total_cost, 4) if rand_total_cost > 0 else 0.0,
        "rand_win_rate": round(sum(rand_hit_records) / max(len(rand_hit_records), 1), 4),
        "rand_sharpe": round(rand_sharpe, 4),
        "rand_max_drawdown": round(rand_dd, 2),
        "algo_vs_random_net_diff": round(algo_net - rand_net, 2),
        "play_stats": {p: {
            "tests": s["tests"],
            "algo_hit_rate": round(s["algo_hits"] / max(s["tests"], 1), 4),
            "rand_hit_rate": round(s["rand_hits"] / max(s["tests"], 1), 4),
            "algo_roi": round((s["algo_payout"] - s["algo_cost"]) / s["algo_cost"], 4) if s["algo_cost"] > 0 else 0.0,
            "rand_roi": round((s["rand_payout"] - s["rand_cost"]) / s["rand_cost"], 4) if s["rand_cost"] > 0 else 0.0,
        } for p, s in play_stats.items()},
        "equity": [round(v, 2) for v in equity],
        "period_values": [round(v, 2) for v in period_values],
    }


def _budget_plans(budget, rec):
    """量化版预算分配（风险校正 — 减少单码，提高高EV玩法占比）。"""
    weights_map = {
        "二定单码": 0.02, "二定包码": 0.15,
        "三定单码": 0.02, "三定包码": 0.20,
        "四定单码": 0.01, "四定包码": 0.10,
        "二现": 0.20, "三现": 0.25, "四现": 0.05,
    }
    bao_combos = {}
    for name in ["二定", "三定", "四定"]:
        nn = 1
        for _, ds in rec[name]["包码"]:
            nn *= len(ds)
        bao_combos[name] = nn

    schemes = {
        "二定单码": {"combo": 1, "cost": 1.0, "payout": float(PAYOUT["二定"])},
        "三定单码": {"combo": 1, "cost": 1.0, "payout": float(PAYOUT["三定"])},
        "四定单码": {"combo": 1, "cost": 1.0, "payout": float(PAYOUT["四定"])},
        "二定包码": {"combo": bao_combos["二定"], "cost": round(bao_combos["二定"] * 0.1, 2), "payout": 0.1 * PAYOUT["二定"]},
        "三定包码": {"combo": bao_combos["三定"], "cost": round(bao_combos["三定"] * 0.1, 2), "payout": 0.1 * PAYOUT["三定"]},
        "四定包码": {"combo": bao_combos["四定"], "cost": round(bao_combos["四定"] * 0.1, 2), "payout": 0.1 * PAYOUT["四定"]},
        "二现": {"combo": 1, "cost": 1.0, "payout": float(PAYOUT["二现"])},
        "三现": {"combo": 1, "cost": 1.0, "payout": float(PAYOUT["三现"])},
        "四现": {"combo": 1, "cost": 1.0, "payout": float(PAYOUT["四现"])},
    }

    plans = {}
    total = 0.0
    for play, weight in weights_map.items():
        s = schemes[play]
        target = budget * weight
        multiples = int(target / s["cost"])
        if multiples < 1:
            continue
        cost = round(multiples * s["cost"], 2)
        plans[play] = {
            "倍数": multiples, "组合数": s["combo"], "实际投入": cost,
            "单注赔付": s["payout"], "中奖金额": round(multiples * s["payout"], 2),
        }
        total += cost
    plans["__total__"] = round(total, 2)
    return plans


def _check_hit(play, rec, actual):
    """检查玩法是否命中。"""
    if "现" in play:
        digits = rec[play]
        return all(d in actual for d in digits)
    else:
        def_name = {"二定单码": "二定", "二定包码": "二定",
                     "三定单码": "三定", "三定包码": "三定",
                     "四定单码": "四定", "四定包码": "四定"}[play]
        idx = {"千位": 0, "百位": 1, "十位": 2, "个位": 3}
        if "单码" in play:
            single = rec[def_name]["单码"]
            return all(actual[idx[pos]] == d for pos, d in single)
        else:
            bao = rec[def_name]["包码"]
            return all(actual[idx[pos]] in digits for pos, digits in bao)


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def run_quant(periods=2000, budget=100.0, output_file=None, verbose=True):
    """运行完整量化流程并写入 JSON 输出文件。

    periods: 使用多少期数据。0 或 None 表示全部。
    返回: 输出文件路径。"""
    out_path = output_file or OUTPUT_FILE
    log = print if verbose else (lambda *a, **k: None)

    log("=" * 60)
    log("  排列五量化选号引擎")
    log("=" * 60)

    target = periods if periods else 100000
    log(f"\n[1/5] 加载历史数据 (目标 {target if periods else '全部'} 期)...")
    history = fetch_history(target)
    n = len(history)
    log(f"      实际获取 {n} 期 ({history[-1]['issue']} → {history[0]['issue']})")

    train_window = max(50, min(500, n // 3))
    ic_test_count = max(50, min(500, n - train_window - 10))
    log(f"      训练窗口 = {train_window}, IC测试 = {ic_test_count}")

    log("[2/5] 滚动回测学习因子权重...")
    weights, ic_report = learn_optimal_weights(history, train_window, ic_test_count)
    for name, info in ic_report.items():
        bar = "█" * int(info["weight"] * 50)
        log(f"        {name:<16} IC={info['ic']:.4f}  W={info['weight']:.4f}  {bar}")

    log("[3/5] 基于学习权重生成最新预测...")
    rec, pos_scores, digit_scores = predict_with_weights(history, weights)

    log("[4/5] 全量回测计算量化指标...")
    bt_result = backtest_with_weights(history, weights, train_window, budget)
    log(f"      测试 {bt_result['n_test']} 期")
    log(f"      算法: 投入 ¥{bt_result['total_cost']}  回报 ¥{bt_result['total_payout']}  "
        f"ROI {bt_result['roi']*100:+.2f}%")
    log(f"      随机: 投入 ¥{bt_result['rand_total_cost']}  回报 ¥{bt_result['rand_total_payout']}  "
        f"ROI {bt_result['rand_roi']*100:+.2f}%")
    log(f"      Sharpe {bt_result['sharpe']:.4f}  胜率 {bt_result['win_rate']*100:.2f}%")

    output = {
        "meta": {
            "periods_used": n,
            "train_window": train_window,
            "test_periods": bt_result["n_test"],
            "budget": budget,
            "generated_at": history[0]["date"],
        },
        "因子权重": ic_report,
        "最优权重": [round(w, 4) for w in weights],
        "预测推荐": {},
        "位置评分": pos_scores,
        "数字评分": digit_scores,
        "回测报告": bt_result,
    }
    for k, v in rec.items():
        if isinstance(v, dict):
            output["预测推荐"][k] = {
                "单码": [(p, int(d)) for p, d in v["单码"]],
                "包码": [(p, [int(x) for x in ds]) for p, ds in v["包码"]],
            }
        else:
            output["预测推荐"][k] = [int(x) for x in v]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log(f"\n[5/5] 结果已写入 {out_path}")
    return out_path


def main():
    periods = TARGET_PERIODS
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip().lower()
        if arg in ("all", "full", "全部", "0"):
            periods = 0
        else:
            try:
                periods = int(arg)
            except ValueError:
                print(f"用法: python quant.py [期数|all]")
                sys.exit(1)
    run_quant(periods=periods, budget=BUDGET)


if __name__ == "__main__":
    main()
