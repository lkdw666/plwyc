# -*- coding: utf-8 -*-
"""排列五预测器 - Web 后端服务

启动方式: python server.py
访问 http://localhost:5173
"""
import sys
import os
import json
import uuid
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory

from predictor import (
    fetch_history, Predictor,
    make_recommendations, make_custom_recommendations,
    calculate_budget_plans, calculate_custom_budget_plans,
    run_backtest,
    PAYOUT_RATIO, PROB_XIAN, RISK_PROFILES, RISK_DESC
)

app = Flask(__name__, static_folder='.', static_url_path='')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_CONFIG_PATH = os.path.join(BASE_DIR, "custom_config.json")

# 量化任务内存存储
_quant_jobs = {}


def _default_config():
    return {
        "enabled": ["二定包码", "三定包码", "二现", "三现"],
        "bao_pos": {
            "二定": [0, 3, 0, 3],
            "三定": [3, 3, 3, 0],
            "四定": [2, 2, 2, 2],
        },
        "xian_manual": {},
        "__active__": False,
    }


def _load_custom_config():
    if not os.path.exists(CUSTOM_CONFIG_PATH):
        return _default_config()
    try:
        with open(CUSTOM_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "enabled": data.get("enabled", []),
            "bao_pos": data.get("bao_pos", _default_config()["bao_pos"]),
            "xian_manual": data.get("xian_manual", {}),
            "__active__": bool(data.get("__active__", False)),
        }
    except Exception:
        return _default_config()


def _save_custom_config(config):
    data = {
        "__active__": bool(config.get("__active__", False)),
        "enabled": list(config.get("enabled", [])),
        "bao_pos": config.get("bao_pos", {}),
        "xian_manual": config.get("xian_manual", {}),
    }
    with open(CUSTOM_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _rec_to_json(rec):
    """将推荐结果转为 JSON 可序列化格式。"""
    result = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            result[k] = {}
            for sub_k, sub_v in v.items():
                result[k][sub_k] = [
                    (p, list(ds) if isinstance(ds, list) else ds)
                    for p, ds in sub_v
                ]
        else:
            result[k] = list(v)
    return result


# ============================================================
# 静态页面
# ============================================================
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


# ============================================================
# 配置 API
# ============================================================
@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(_load_custom_config())


@app.route('/api/config', methods=['POST'])
def save_config():
    try:
        _save_custom_config(request.json)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 预测 API
# ============================================================
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        body = request.json or {}
        budget = float(body.get("budget", 100))
        risk = body.get("risk", "平衡")
        use_custom = bool(body.get("use_custom", False))

        if budget < 0:
            return jsonify({"error": "预算不能为负数"}), 400

        history = fetch_history(50)
        if not history:
            return jsonify({"error": "无法获取开奖数据"}), 500

        latest = history[0]
        predictor = Predictor(history)

        if use_custom:
            config = body.get("custom_config", _load_custom_config())
            rec, pos_scores, digit_scores, enabled = make_custom_recommendations(
                predictor, config)
            budget_plans = calculate_custom_budget_plans(budget, rec, enabled)
        else:
            rec, pos_scores, digit_scores = make_recommendations(predictor)
            budget_plans = calculate_budget_plans(budget, rec, risk)

        # 清理 budget_plans 中的特殊键
        plans_clean = {}
        for k, v in budget_plans.items():
            if k.startswith("__"):
                plans_clean[k] = v
            else:
                plans_clean[k] = {
                    "倍数": v["倍数"],
                    "组合数": v["组合数"],
                    "单份成本": v.get("单份成本", 0),
                    "实际投入": v["实际投入"],
                    "命中概率": v["命中概率"],
                    "单注赔付": v["单注赔付"],
                    "中奖金额": v["中奖金额"],
                    "净收益": v["净收益"],
                }

        recent = []
        for h in history[:10]:
            recent.append({
                "issue": h["issue"],
                "date": h["date"],
                "nums": h["nums"],
            })

        return jsonify({
            "latest": {
                "issue": latest["issue"],
                "date": latest["date"],
                "nums": latest["nums"],
            },
            "recommendations": _rec_to_json(rec),
            "budget_plans": plans_clean,
            "pos_scores": pos_scores,
            "digit_scores": digit_scores,
            "recent_history": recent,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 回测 API
# ============================================================
@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    try:
        body = request.json or {}
        budget = float(body.get("budget", 100))
        risk = body.get("risk", "平衡")

        if budget < 0:
            return jsonify({"error": "预算不能为负数"}), 400

        history = fetch_history(100)
        if not history:
            return jsonify({"error": "无法获取开奖数据"}), 500

        result = run_backtest(history, train_window=50, budget=budget, risk=risk)

        play_stats_clean = {}
        for k, v in result["play_stats"].items():
            play_stats_clean[k] = {
                "algo_bets": v["algo_bets"],
                "algo_hits": v["algo_hits"],
                "algo_cost": round(v["algo_cost"], 2),
                "algo_payout": round(v["algo_payout"], 2),
                "algo_hit_rate": v.get("algo_hit_rate", 0),
                "algo_roi": v.get("algo_roi", 0),
                "random_bets": v["random_bets"],
                "random_hits": v["random_hits"],
                "random_cost": round(v["random_cost"], 2),
                "random_payout": round(v["random_payout"], 2),
                "random_hit_rate": v.get("random_hit_rate", 0),
                "random_roi": v.get("random_roi", 0),
            }

        details_clean = []
        for d in result["details"]:
            details_clean.append({
                "issue": d["issue"],
                "actual": d["actual"],
                "algo_cost": round(d["algo_eval"]["total_cost"], 2),
                "algo_payout": round(d["algo_eval"]["total_payout"], 2),
                "random_cost": round(d["random_eval"]["total_cost"], 2),
                "random_payout": round(d["random_eval"]["total_payout"], 2),
            })

        return jsonify({
            "train_window": result["train_window"],
            "n_test": result["n_test"],
            "budget": result["budget"],
            "risk": result["risk"],
            "totals": result["totals"],
            "play_stats": play_stats_clean,
            "details": details_clean,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 量化 API（异步）
# ============================================================
@app.route('/api/quant', methods=['POST'])
def start_quant():
    try:
        periods_raw = request.json.get("periods", "2000") if request.json else "2000"
        if str(periods_raw).strip().lower() in ("全部", "all", "0", ""):
            periods = 0
        else:
            periods = int(periods_raw)

        job_id = str(uuid.uuid4())
        _quant_jobs[job_id] = {"status": "running", "result": None, "error": None}

        def run_job():
            try:
                import quant
                fd, tmp_path = tempfile.mkstemp(suffix='.json', prefix='quant_')
                os.close(fd)
                quant.run_quant(periods=periods, budget=100.0,
                                output_file=tmp_path, verbose=False)
                with open(tmp_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                _quant_jobs[job_id] = {"status": "done", "result": data, "error": None}
            except Exception as e:
                _quant_jobs[job_id] = {"status": "error", "result": None, "error": str(e)}

        threading.Thread(target=run_job, daemon=True).start()
        return jsonify({"job_id": job_id, "status": "running"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/quant/<job_id>', methods=['GET'])
def get_quant_result(job_id):
    job = _quant_jobs.get(job_id)
    if not job:
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(job)


# ============================================================
# 启动
# ============================================================
if __name__ == '__main__':
    print("=" * 50)
    print("  排列五预测器 Web 服务")
    print("  打开浏览器访问 http://localhost:5173")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5173, debug=True, threaded=True)
